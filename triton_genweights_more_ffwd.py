# Triton generated-weight experiment.
# The hot path is branch-free: procedural weights are generated inside Triton
# and immediately fed to tensor-core matmuls. Trainable block overrides are
# applied as a small correction outside the main generated-weight matmul.
import math
import os
import pickle
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise RuntimeError(
        "This script needs Triton installed in the active Python environment. "
        "If your setup supports it, install it with: pip install triton. "
        "On Windows, Triton availability depends on your Python/CUDA/PyTorch setup; "
        "if pip cannot find a compatible wheel, run this script from WSL2/Linux or "
        "keep using cuda_genwblocks3.py."
    ) from exc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    raise RuntimeError("triton_genweights.py requires a CUDA GPU.")

print('device is: ', device)

# parameters to tweak
max_iters = 20_001
eval_iters = 10
eval_interval = 5_000
n_embed = 1024
block_size = 64
batch_size = 16
learning_rate = 1e-3
n_head = 16
n_layer = 10
dropout = 0.2

# intrinsic dimension adjustments

percentweights = 0.1
percentweights_ffwd_lm = 0.4
use_fp16_generated_matmul = True

dataset_range = 80_000
vocab_size = 800
num_merges = vocab_size - 256
cache_file = f"wikitext_bpe_cache_{dataset_range}_{vocab_size}.pkl"


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


if os.path.exists(cache_file):
    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    data = cache_data['data']
    merges = cache_data['merges']
    vocab = cache_data['vocab']
else:
    print(f"Downloading and processing wikitext dataset (range: {dataset_range})...")
    textraw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    sample = textraw['train'].select(range(min(dataset_range, len(textraw['train']))))
    text = " ".join(sample["text"])

    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    def encode_internal(text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            tokens = merge(tokens, pair, merges[pair])
        return tokens

    print("Encoding dataset...")
    data = torch.tensor(encode_internal(text), dtype=torch.long)

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({'data': data, 'merges': merges, 'vocab': vocab}, f)


def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    return tokens.decode("utf-8", errors='replace')


def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens


n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def procedural_weight_values(flat_indices, in_features, seed):
    x = (flat_indices.to(torch.int64) + seed) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    limit = 1.0 / math.sqrt(in_features)
    return (x.to(torch.float32) / 4294967296.0) * (2.0 * limit) - limit


def contiguous_spans(start, length, total):
    if length <= 0:
        return []
    if start + length <= total:
        return [(start, length)]
    first = total - start
    return [(start, first), (0, length - first)]


@triton.jit
def _gen_weight(target, seed, K: tl.constexpr):
    x = (target + seed).to(tl.uint32)
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    limit = 1.0 / tl.sqrt(tl.full((), K, tl.float32))
    return (x.to(tl.float32) / 4294967296.0) * (2.0 * limit) - limit


@triton.jit
def _triton_forward(
    X, Y,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    seed,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x = tl.load(
            X + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )

        target = offs_n[None, :] * K + k_idxs[:, None]
        w = _gen_weight(target, seed, K)

        if USE_FP16:
            acc += tl.dot(x.to(tl.float16), w.to(tl.float16))
        else:
            acc += tl.dot(x, w, input_precision="tf32")

    tl.store(
        Y + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _triton_backward_dx(
    DY, DX,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    seed,
    BLOCK_M: tl.constexpr, BLOCK_K_OUT: tl.constexpr, BLOCK_N: tl.constexpr,
    USE_FP16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n
        dy = tl.load(
            DY + offs_m[:, None] * N + n_idxs[None, :],
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        )

        target = n_idxs[:, None] * K + offs_k[None, :]
        w = _gen_weight(target, seed, K)

        if USE_FP16:
            acc += tl.dot(dy.to(tl.float16), w.to(tl.float16))
        else:
            acc += tl.dot(dy, w, input_precision="tf32")

    tl.store(
        DX + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _triton_backward_override(
    DY, X, DOVR,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    num_sparse, block_start, num_elements,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_m = tl.arange(0, BLOCK_M)

    flat = (block_start + offs_s) % num_elements
    n_idx = flat // K
    k_idx = flat - n_idx * K

    dy = tl.load(
        DY + offs_m[:, None] * N + n_idx[None, :],
        mask=(offs_m[:, None] < M) & (offs_s[None, :] < num_sparse),
        other=0.0,
    )
    x = tl.load(
        X + offs_m[:, None] * K + k_idx[None, :],
        mask=(offs_m[:, None] < M) & (offs_s[None, :] < num_sparse),
        other=0.0,
    )
    grad = tl.sum(dy * x, axis=0)
    tl.store(DOVR + offs_s, grad, mask=offs_s < num_sparse)


class TritonGeneratedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_features, out_features, seed):
        x_flat = x.reshape(-1, in_features).contiguous().to(torch.float32)
        M = x_flat.shape[0]
        y = torch.empty((M, out_features), device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(M, 32), triton.cdiv(out_features, 64))
        _triton_forward[grid](
            x_flat, y,
            M, in_features, out_features,
            seed,
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=64,
            USE_FP16=use_fp16_generated_matmul,
            num_warps=4,
        )

        ctx.save_for_backward(x_flat)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.seed = seed
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return y.view(*x.shape[:-1], out_features).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x_flat,) = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1, ctx.out_features).contiguous().to(torch.float32)
        M = x_flat.shape[0]

        grad_x_flat = torch.empty((M, ctx.in_features), device=x_flat.device, dtype=torch.float32)
        grid_dx = (triton.cdiv(M, 32), triton.cdiv(ctx.in_features, 64))
        _triton_backward_dx[grid_dx](
            grad_output_flat, grad_x_flat,
            M, ctx.in_features, ctx.out_features,
            ctx.seed,
            BLOCK_M=32, BLOCK_K_OUT=64, BLOCK_N=64,
            USE_FP16=use_fp16_generated_matmul,
            num_warps=4,
        )

        return (
            grad_x_flat.view(ctx.x_shape).to(ctx.x_dtype),
            None,
            None,
            None,
        )


class TritonProceduralLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=0, sparsity=0.05, bias=False, block_index=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        num_elements = in_features * out_features
        self.num_sparse = int(num_elements * sparsity)
        self.num_elements = num_elements

        self.block_start = (block_index * self.num_sparse) % num_elements
        self.spans = []
        for flat_start, span_len in contiguous_spans(self.block_start, self.num_sparse, num_elements):
            row_start = flat_start // in_features
            row_stop = (flat_start + span_len - 1) // in_features + 1
            local_start = flat_start - row_start * in_features
            flat_indices = torch.arange(flat_start, flat_start + span_len, dtype=torch.long)
            base_values = procedural_weight_values(flat_indices, in_features, seed)
            name = f'override_values_{len(self.spans)}'
            base_name = f'override_base_{len(self.spans)}'
            self.register_parameter(name, nn.Parameter(base_values.clone()))
            self.register_buffer(base_name, base_values)
            self.spans.append((name, base_name, row_start, row_stop, local_start, span_len))

        if bias:
            self.biases = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.biases, -bound, bound)
        else:
            self.register_parameter('biases', None)

    def forward(self, x):
        out = TritonGeneratedLinearFunction.apply(
            x, self.in_features, self.out_features, self.seed
        )
        for name, base_name, row_start, row_stop, local_start, span_len in self.spans:
            override = getattr(self, name)
            base_values = getattr(self, base_name)
            rows = row_stop - row_start
            correction = torch.zeros(rows, self.in_features, device=x.device, dtype=torch.float32)
            correction_flat = correction.view(-1)
            correction_flat[local_start:local_start + span_len] = (
                override.to(torch.float32) - base_values.to(x.device, torch.float32)
            )
            correction_out = torch.zeros_like(out)
            correction_out[..., row_start:row_stop] = F.linear(
                x.to(torch.float32), correction
            ).to(out.dtype)
            out = out + correction_out
        if self.biases is not None:
            out = out + self.biases
        return out


class MultiheadAttentionBatch(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx, sparsity=0.05):
        super().__init__()
        self.num_heads = n_head
        self.head_size = n_embed // n_head

        seed_base = layer_idx * 1000
        bi = layer_idx * 6
        self.query = TritonProceduralLinear(n_embed, n_embed, seed=seed_base + 1, sparsity=sparsity, bias=False, block_index=bi)
        self.key = TritonProceduralLinear(n_embed, n_embed, seed=seed_base + 2, sparsity=sparsity, bias=False, block_index=bi + 1)
        self.value = TritonProceduralLinear(n_embed, n_embed, seed=seed_base + 3, sparsity=sparsity, bias=False, block_index=bi + 2)
        self.proj = TritonProceduralLinear(n_embed, n_embed, seed=seed_base + 4, sparsity=sparsity, bias=False, block_index=bi + 3)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, layer_idx, sparsity=0.05):
        super().__init__()
        seed_base = layer_idx * 1000 + 500
        bi = layer_idx * 6 + 4
        self.up = TritonProceduralLinear(n_embed, 4 * n_embed, seed=seed_base + 1, sparsity=sparsity, bias=False, block_index=bi)
        self.down = TritonProceduralLinear(4 * n_embed, n_embed, seed=seed_base + 2, sparsity=sparsity, bias=False, block_index=bi + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = F.relu(x)
        x = self.down(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx, sparsity_attn=0.05, sparsity_ffwd=0.05):
        super().__init__()
        self.sa = MultiheadAttentionBatch(n_embed, n_head, layer_idx, sparsity_attn)
        self.ffwd = FeedForward(n_embed, layer_idx, sparsity_ffwd)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head, i, percentweights, percentweights_ffwd_lm)
            for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = TritonProceduralLinear(n_embed, vocab_size, seed=99000, sparsity=percentweights_ffwd_lm, bias=False, block_index=n_layer * 6)

    def forward(self, idx, targets=None):
        _, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Transformer()
total_params = sum(p.numel() for p in model.parameters())
simulated_params = total_params
for module in model.modules():
    if isinstance(module, TritonProceduralLinear):
        simulated_params += module.out_features * module.in_features - module.num_sparse
print(
    'size of model (intrinsic dimensions adapted):',
    total_params,
    '| simulated full generated-weight parameters:',
    simulated_params,
)

m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("Warming up Triton kernels...")
xb, yb = get_batch('train')
_, warmup_loss = m(xb, yb)
optimizer.zero_grad(set_to_none=True)
warmup_loss.backward()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

start_time = time.time()

for iter in range(max_iters):
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")
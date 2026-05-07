
import os
import subprocess
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.cpp_extension import load_inline
import time
import pickle
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. ENVIRONMENT SETUP (Required for Windows compilation)
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
if os.path.exists(cuda_bin):
    os.add_dll_directory(cuda_bin)

# Inject VS environment (x64)
vs_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vs_path):
    try:
        output = subprocess.check_output(f'"{vs_path}" && set', shell=True, text=True)
        for line in output.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    except Exception:
        pass


print('device is: ', device)

# parameters to tweak
max_iters = 201
eval_iters = 10
eval_interval = 50
n_embed = 1024
block_size = 32
batch_size = 16 # Increased for better GPU utilization
learning_rate = 1e-3
n_head = 4
n_layer = 10  
dropout = 0.2

# intrinsic dimension adjustments
num_matrices = 2
percentweights = 0.01  # amount each layer updates of its assigned matrix

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
        if i<len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+= 2
        else:
            newids.append(ids[i])
            i+=1
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

    tokens = text.encode("utf-8")
    ids = list(map(int, tokens))
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
            idx = merges[pair]
            tokens = merge(tokens, pair, idx)
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
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

__device__ float get_procedural_weight(int n, int k, int seed, int K) {
    unsigned int x = (unsigned int)(n * K + k + seed);
    x ^= (x >> 16);
    x = (x * 0x85ebca6b);
    x ^= (x >> 13);
    x = (x * 0xc2b2ae35);
    x ^= (x >> 16);
    
    float u = (float)x / 4294967296.0f;
    float limit = 1.0f / sqrtf((float)K);
    return u * (2.0f * limit) - limit;
}

// O(1) block-contiguous override check.
// Returns the index into override_values if target is in the block, else -1.
// Handles wrap-around: if block_start + num_sparse > num_elements, the block
// wraps around to the beginning.
__device__ __forceinline__ int get_block_override_idx(
    int target, int block_start, int num_sparse, int num_elements) {
    // Compute offset from block_start with wrap-around
    int offset = target - block_start;
    if (offset < 0) offset += num_elements;
    return (offset < num_sparse) ? offset : -1;
}

__global__ void forward_kernel_optimized(
    const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ bias,
    const float* __restrict__ override_values,
    int M, int K, int N, int seed, int num_sparse,
    int block_start, int num_elements, bool has_bias) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    int row_start = blockIdx.y * 64 + ty * 4;
    int col_start = blockIdx.x * 64 + tx * 4;

    __shared__ float Xs[64][64];
    __shared__ float Ws[64][64];

    float sums[4][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };
    float limit = 1.0f / sqrtf((float)K);
    float two_limit = 2.0f * limit;

    for (int k_tile = 0; k_tile < (K + 63) / 64; ++k_tile) {
        for (int idx = tid; idx < 4096; idx += 256) {
            int local_r = idx / 64;
            int local_c = idx % 64;
            int r = blockIdx.y * 64 + local_r;
            int c = k_tile * 64 + local_c;
            Xs[local_r][local_c] = (r < M && c < K) ? X[r * K + c] : 0.0f;
        }

        for (int idx = tid; idx < 4096; idx += 256) {
            int local_k = idx / 64;
            int local_n = idx % 64;
            int w_row = blockIdx.x * 64 + local_n;
            int w_col = k_tile * 64 + local_k;

            float w_val = 0.0f;
            if (w_row < N && w_col < K) {
                int target = w_row * K + w_col;
                int ov_idx = get_block_override_idx(target, block_start, num_sparse, num_elements);
                if (ov_idx >= 0) {
                    w_val = override_values[ov_idx];
                } else {
                    unsigned int x = (unsigned int)(target + seed);
                    x ^= (x >> 16); x *= 0x85ebca6b; x ^= (x >> 13); x *= 0xc2b2ae35; x ^= (x >> 16);
                    w_val = ((float)x / 4294967296.0f) * two_limit - limit;
                }
            }
            Ws[local_k][local_n] = w_val;
        }
        
        __syncthreads();
        
        for (int k = 0; k < 64; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x_val = Xs[ty * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] += x_val * Ws[k][tx * 4 + j];
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < N) {
                float out = sums[i][j];
                if (has_bias) out += bias[c];
                Y[r * N + c] = out;
            }
        }
    }
}

__global__ void backward_dx_kernel_optimized(
    const float* __restrict__ dY, float* __restrict__ dX,
    const float* __restrict__ override_values,
    int M, int K, int N, int seed, int num_sparse,
    int block_start, int num_elements) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    int row_start = blockIdx.y * 64 + ty * 4;
    int col_start = blockIdx.x * 64 + tx * 4;

    __shared__ float sdYs[64][64];
    __shared__ float Ws[64][64];

    float sums[4][4] = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f}
    };
    float limit = 1.0f / sqrtf((float)K);
    float two_limit = 2.0f * limit;

    for (int n_tile = 0; n_tile < (N + 63) / 64; ++n_tile) {
        for (int idx = tid; idx < 4096; idx += 256) {
            int local_r = idx / 64;
            int local_n = idx % 64;
            int r = blockIdx.y * 64 + local_r;
            int n_col = n_tile * 64 + local_n;
            sdYs[local_r][local_n] = (r < M && n_col < N) ? dY[r * N + n_col] : 0.0f;
        }

        for (int idx = tid; idx < 4096; idx += 256) {
            int local_n = idx / 64;
            int local_k = idx % 64;
            int w_row = n_tile * 64 + local_n;
            int w_col = blockIdx.x * 64 + local_k;

            float w_val = 0.0f;
            if (w_row < N && w_col < K) {
                int target = w_row * K + w_col;
                int ov_idx = get_block_override_idx(target, block_start, num_sparse, num_elements);
                if (ov_idx >= 0) {
                    w_val = override_values[ov_idx];
                } else {
                    unsigned int x = (unsigned int)(target + seed);
                    x ^= (x >> 16); x *= 0x85ebca6b; x ^= (x >> 13); x *= 0xc2b2ae35; x ^= (x >> 16);
                    w_val = ((float)x / 4294967296.0f) * two_limit - limit;
                }
            }
            Ws[local_n][local_k] = w_val;
        }
        
        __syncthreads();
        
        for (int n = 0; n < 64; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float dy_val = sdYs[ty * 4 + i][n];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] += dy_val * Ws[n][tx * 4 + j];
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < K) {
                dX[r * K + c] = sums[i][j];
            }
        }
    }
}

__global__ void backward_override_kernel(
    const float* dY, const float* X,
    float* dOverride,
    int M, int K, int N, int num_sparse,
    int block_start, int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sparse) {
        // Compute flat index on-the-fly from block offset
        int flat_idx = (block_start + idx) % num_elements;
        int n = flat_idx / K;
        int k = flat_idx % K;

        float sum = 0.0f;
        int m = 0;
        for (; m + 3 < M; m += 4) {
            sum += dY[m * N + n] * X[m * K + k];
            sum += dY[(m + 1) * N + n] * X[(m + 1) * K + k];
            sum += dY[(m + 2) * N + n] * X[(m + 2) * K + k];
            sum += dY[(m + 3) * N + n] * X[(m + 3) * K + k];
        }
        for (; m < M; ++m) {
            sum += dY[m * N + n] * X[m * K + k];
        }

        dOverride[idx] = sum;
    }
}

torch::Tensor forward_impl(torch::Tensor X, torch::Tensor override_values, int in_features, int out_features, int seed, int num_sparse, int block_start, int num_elements, bool has_bias, c10::optional<torch::Tensor> bias) {
    int M = X.size(0);
    int K = in_features;
    int N = out_features;
    
    auto Y = torch::empty({M, N}, X.options());
    
    dim3 threads(16, 16);
    dim3 blocks((N + 63) / 64, (M + 63) / 64);
    
    const float* bias_ptr = nullptr;
    if (has_bias && bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    
    forward_kernel_optimized<<<blocks, threads>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), bias_ptr,
        override_values.data_ptr<float>(),
        M, K, N, seed, num_sparse,
        block_start, num_elements, has_bias
    );
    
    return Y;
}

std::vector<torch::Tensor> backward_impl(torch::Tensor dY, torch::Tensor X, torch::Tensor override_values, int in_features, int out_features, int seed, int num_sparse, int block_start, int num_elements) {
    int M = X.size(0);
    int K = in_features;
    int N = out_features;
    
    auto dX = torch::empty({M, K}, X.options());
    auto dOverride = torch::zeros({num_sparse}, override_values.options());
    
    dim3 threads(16, 16);
    dim3 blocks_dx((K + 63) / 64, (M + 63) / 64);
    
    backward_dx_kernel_optimized<<<blocks_dx, threads>>>(
        dY.data_ptr<float>(), dX.data_ptr<float>(),
        override_values.data_ptr<float>(),
        M, K, N, seed, num_sparse,
        block_start, num_elements
    );
    
    int threads_override = 256;
    int blocks_override = (num_sparse + threads_override - 1) / threads_override;
    
    if (num_sparse > 0) {
        backward_override_kernel<<<blocks_override, threads_override>>>(
            dY.data_ptr<float>(), X.data_ptr<float>(),
            dOverride.data_ptr<float>(),
            M, K, N, num_sparse,
            block_start, num_elements
        );
    }
    
    return {dX, dOverride};
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor forward_impl(torch::Tensor X, torch::Tensor override_values, int in_features, int out_features, int seed, int num_sparse, int block_start, int num_elements, bool has_bias, c10::optional<torch::Tensor> bias);
std::vector<torch::Tensor> backward_impl(torch::Tensor dY, torch::Tensor X, torch::Tensor override_values, int in_features, int out_features, int seed, int num_sparse, int block_start, int num_elements);
"""

try:
    procedural_linear_ext = load_inline(
        name='procedural_linear_ext_blocks3',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['forward_impl', 'backward_impl'],
        verbose=True,
        extra_cflags=["/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"],
        extra_cuda_cflags=[
            '-arch=sm_86', 
            '-allow-unsupported-compiler', 
            '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
        ],
    )
except Exception as e:
    print(f"Warning: Failed to compile CUDA extension: {e}")
    procedural_linear_ext = None

class ProceduralLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, override_values, bias, in_features, out_features, seed, block_start, num_elements, num_sparse):
        x_flat = x.view(-1, in_features)
        has_bias = bias is not None
        
        x_flat = x_flat.contiguous().to(torch.float32)
        override_values = override_values.contiguous().to(torch.float32)
        if has_bias:
            bias = bias.contiguous().to(torch.float32)
            
        out_flat = procedural_linear_ext.forward_impl(
            x_flat, override_values,
            in_features, out_features, seed, num_sparse,
            block_start, num_elements, has_bias, bias
        )
        
        ctx.save_for_backward(x_flat, override_values)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.seed = seed
        ctx.block_start = block_start
        ctx.num_elements = num_elements
        ctx.num_sparse = num_sparse
        ctx.has_bias = has_bias
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        
        return out_flat.view(*x.shape[:-1], out_features).to(x.dtype)
        
    @staticmethod
    def backward(ctx, grad_output):
        x_flat, override_values = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1, ctx.out_features).contiguous().to(torch.float32)
        
        grads = procedural_linear_ext.backward_impl(
            grad_output_flat, x_flat, override_values,
            ctx.in_features, ctx.out_features, ctx.seed,
            ctx.num_sparse, ctx.block_start, ctx.num_elements
        )
        
        grad_x_flat = grads[0]
        grad_override_values = grads[1]
        
        grad_x = grad_x_flat.view(ctx.x_shape).to(ctx.x_dtype)
        grad_override_values = grad_override_values.to(ctx.x_dtype)
        
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output_flat.sum(dim=0).to(ctx.x_dtype)
            
        return grad_x, grad_override_values, grad_bias, None, None, None, None, None, None

class ProceduralLinear(nn.Module):
    """
    Memory-less procedural linear layer.
    Uses a custom CUDA kernel to generate weights on the fly.
    Override positions are a contiguous block in the flattened weight matrix.
    The kernel checks membership via an O(1) range test (no hash table,
    no index buffer), and the contiguous layout maximizes GPU coalescing.
    """
    def __init__(self, in_features, out_features, seed=0, sparsity=0.05, bias=False, block_index=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self.sparsity = sparsity
        
        num_elements = in_features * out_features
        self.num_sparse = int(num_elements * sparsity)
        self.num_elements = num_elements
        
        self.override_values = nn.Parameter(torch.Tensor(self.num_sparse))
        limit = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.override_values, -limit, limit)
        
        # Block-contiguous override region: [block_start, block_start + num_sparse) mod num_elements
        self.block_start = (block_index * self.num_sparse) % num_elements
        
        if bias:
            self.biases = nn.Parameter(torch.Tensor(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.biases, -bound, bound)
        else:
            self.register_parameter('biases', None)

    def forward(self, x):
        if procedural_linear_ext is None:
            raise RuntimeError("CUDA extension failed to compile.")
        return ProceduralLinearFunction.apply(
            x, self.override_values, self.biases, 
            self.in_features, self.out_features, self.seed,
            self.block_start, self.num_elements, self.num_sparse
        )

class MultiheadAttentionBatch(nn.Module):
    """Refactored multihead attention to use individual ProceduralLinear layers."""
    def __init__(self, n_embed, n_head, layer_idx, sparsity=0.05):
        super().__init__()
        self.num_heads = n_head
        self.head_size = n_embed // n_head
        
        seed_base = layer_idx * 1000
        # Each projection gets a unique block_index so overrides don't overlap
        bi = layer_idx * 6  # 6 projections per layer (4 attn + 2 ffn)
        self.query = ProceduralLinear(n_embed, n_embed, seed=seed_base+1, sparsity=sparsity, bias=False, block_index=bi)
        self.key = ProceduralLinear(n_embed, n_embed, seed=seed_base+2, sparsity=sparsity, bias=False, block_index=bi+1)
        self.value = ProceduralLinear(n_embed, n_embed, seed=seed_base+3, sparsity=sparsity, bias=False, block_index=bi+2)
        self.proj = ProceduralLinear(n_embed, n_embed, seed=seed_base+4, sparsity=sparsity, bias=False, block_index=bi+3)
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
        bi = layer_idx * 6 + 4  # continues after the 4 attention projections
        self.up = ProceduralLinear(n_embed, 4 * n_embed, seed=seed_base+1, sparsity=sparsity, bias=False, block_index=bi)
        self.down = ProceduralLinear(4 * n_embed, n_embed, seed=seed_base+2, sparsity=sparsity, bias=False, block_index=bi+1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = F.relu(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx, sparsity=0.05):
        super().__init__()
        self.sa = MultiheadAttentionBatch(n_embed, n_head, layer_idx, sparsity)
        self.ffwd = FeedForward(n_embed, layer_idx, sparsity)
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
        
        sparsity = percentweights # using percentweights from hyperparams
        
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_head, i, sparsity) 
            for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed) 
        self.lm_head = ProceduralLinear(n_embed, vocab_size, seed=99000, sparsity=sparsity, bias=False, block_index=n_layer * 6)

    def forward(self, idx, targets=None):
        b,t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
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
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = Transformer()
total_params = sum(p.numel() for p in model.parameters())
print('size of model (intrinsic dimensions adapted):', total_params)

m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

start_time = time.time()

for iter in range(max_iters):
    if not iter % eval_interval:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

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
percentweights = 0.001  # amount each layer updates of its assigned matrix

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

__global__ void forward_kernel_optimized(
    const float* __restrict__ X, const float* __restrict__ weight, float* __restrict__ Y, const float* __restrict__ bias,
    int M, int K, int N, bool has_bias) {
    
    // Each thread computes a 4x4 tile of the output Y
    // Thread block is 16x16, computing a 64x64 tile of Y
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_start = blockIdx.y * 64 + ty * 4;
    int col_start = blockIdx.x * 64 + tx * 4;
    
    __shared__ float sX[64][64];
    __shared__ float sW[64][64];
    
    float res[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    
    for (int k_tile = 0; k_tile < (K + 63) / 64; ++k_tile) {
        // Load X into shared memory using float4
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = blockIdx.y * 64 + ty * 4 + i;
            int c_base = k_tile * 64 + tx * 4;
            if (r < M && c_base < K) {
                *(float4*)&sX[ty*4+i][tx*4] = *(float4*)&X[r * K + c_base];
            } else {
                *(float4*)&sX[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        // Load W into shared memory (W is N, K)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = blockIdx.x * 64 + ty * 4 + i; // n
            int c_base = k_tile * 64 + tx * 4;   // k
            if (r < N && c_base < K) {
                *(float4*)&sW[ty*4+i][tx*4] = *(float4*)&weight[r * K + c_base];
            } else {
                *(float4*)&sW[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        __syncthreads();
        
        // Compute 4x4 micro-tile
        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    res[i][j] += sX[ty*4+i][k] * sW[tx*4+j][k];
                }
            }
        }
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < N) {
                float val = res[i][j];
                if (has_bias) val += bias[c];
                Y[r * N + c] = val;
            }
        }
    }
}

__global__ void backward_dx_kernel_optimized(
    const float* __restrict__ dY, const float* __restrict__ weight, float* __restrict__ dX,
    int M, int K, int N) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_start = blockIdx.y * 64 + ty * 4;
    int col_start = blockIdx.x * 64 + tx * 4;
    
    __shared__ float sdY[64][64];
    __shared__ float sW[64][64];
    
    float res[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    
    for (int n_tile = 0; n_tile < (N + 63) / 64; ++n_tile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = blockIdx.y * 64 + ty * 4 + i;
            int c_base = n_tile * 64 + tx * 4;
            if (r < M && c_base < N) {
                *(float4*)&sdY[ty*4+i][tx*4] = *(float4*)&dY[r * N + c_base];
            } else {
                *(float4*)&sdY[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = n_tile * 64 + ty * 4 + i; // n
            int c_base = blockIdx.x * 64 + tx * 4; // k
            if (r < N && c_base < K) {
                // weight is (N, K)
                *(float4*)&sW[ty*4+i][tx*4] = *(float4*)&weight[r * K + c_base];
            } else {
                *(float4*)&sW[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int n = 0; n < 64; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    res[i][j] += sdY[ty*4+i][n] * sW[n][tx*4+j];
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
                dX[r * K + c] = res[i][j];
            }
        }
    }
}

__global__ void backward_dw_kernel_optimized(
    const float* __restrict__ dY, const float* __restrict__ X, float* __restrict__ dW,
    int M, int K, int N) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_start = blockIdx.y * 64 + ty * 4; // n
    int col_start = blockIdx.x * 64 + tx * 4; // k
    
    __shared__ float sdY[64][64];
    __shared__ float sX[64][64];
    
    float res[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    
    for (int m_tile = 0; m_tile < (M + 63) / 64; ++m_tile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = m_tile * 64 + ty * 4 + i; // m
            int c_base = blockIdx.y * 64 + tx * 4; // n
            if (r < M && c_base < N) {
                *(float4*)&sdY[ty*4+i][tx*4] = *(float4*)&dY[r * N + c_base];
            } else {
                *(float4*)&sdY[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = m_tile * 64 + ty * 4 + i; // m
            int c_base = blockIdx.x * 64 + tx * 4; // k
            if (r < M && c_base < K) {
                *(float4*)&sX[ty*4+i][tx*4] = *(float4*)&X[r * K + c_base];
            } else {
                *(float4*)&sX[ty*4+i][tx*4] = make_float4(0,0,0,0);
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int m = 0; m < 64; ++m) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    res[i][j] += sdY[m][ty*4+i] * sX[m][tx*4+j];
                }
            }
        }
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = row_start + i;
            int k = col_start + j;
            if (n < N && k < K) {
                dW[n * K + k] = res[i][j];
            }
        }
    }
}

torch::Tensor forward_impl(torch::Tensor X, torch::Tensor weight, int in_features, int out_features, bool has_bias, c10::optional<torch::Tensor> bias) {
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
        X.data_ptr<float>(), weight.data_ptr<float>(), Y.data_ptr<float>(), bias_ptr,
        M, K, N, has_bias
    );
    
    return Y;
}

std::vector<torch::Tensor> backward_impl(torch::Tensor dY, torch::Tensor X, torch::Tensor weight, int in_features, int out_features) {
    int M = X.size(0);
    int K = in_features;
    int N = out_features;
    
    auto dX = torch::empty({M, K}, X.options());
    auto dW = torch::empty({N, K}, weight.options());
    
    dim3 threads(16, 16);
    dim3 blocks_dx((K + 63) / 64, (M + 63) / 64);
    dim3 blocks_dw((K + 63) / 64, (N + 63) / 64);
    
    backward_dx_kernel_optimized<<<blocks_dx, threads>>>(
        dY.data_ptr<float>(), weight.data_ptr<float>(), dX.data_ptr<float>(),
        M, K, N
    );
    
    backward_dw_kernel_optimized<<<blocks_dw, threads>>>(
        dY.data_ptr<float>(), X.data_ptr<float>(), dW.data_ptr<float>(),
        M, K, N
    );
    
    return {dX, dW};
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor forward_impl(torch::Tensor X, torch::Tensor weight, int in_features, int out_features, bool has_bias, c10::optional<torch::Tensor> bias);
std::vector<torch::Tensor> backward_impl(torch::Tensor dY, torch::Tensor X, torch::Tensor weight, int in_features, int out_features);
"""

try:
    procedural_linear_ext = load_inline(
        name='procedural_linear_ext',
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
    def forward(ctx, x, weight, bias, in_features, out_features):
        x_flat = x.view(-1, in_features)
        has_bias = bias is not None
        
        x_flat = x_flat.contiguous().to(torch.float32)
        weight = weight.contiguous().to(torch.float32)
        if has_bias:
            bias = bias.contiguous().to(torch.float32)
            
        out_flat = procedural_linear_ext.forward_impl(
            x_flat, weight,
            in_features, out_features, has_bias, bias
        )
        
        ctx.save_for_backward(x_flat, weight)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.has_bias = has_bias
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        
        return out_flat.view(*x.shape[:-1], out_features).to(x.dtype)
        
    @staticmethod
    def backward(ctx, grad_output):
        x_flat, weight = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1, ctx.out_features).contiguous().to(torch.float32)
        
        grads = procedural_linear_ext.backward_impl(
            grad_output_flat, x_flat, weight,
            ctx.in_features, ctx.out_features
        )
        
        grad_x_flat = grads[0]
        grad_weight = grads[1]
        
        grad_x = grad_x_flat.view(ctx.x_shape).to(ctx.x_dtype)
        grad_weight = grad_weight.to(ctx.x_dtype)
        
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_output_flat.sum(dim=0).to(ctx.x_dtype)
            
        return grad_x, grad_weight, grad_bias, None, None

class ProceduralLinear(nn.Module):
    """
    Standard linear layer using tiled CUDA kernels.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        limit = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -limit, limit)
        
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
            x, self.weight, self.biases, 
            self.in_features, self.out_features
        )

class MultiheadAttentionBatch(nn.Module):
    """Refactored multihead attention to use individual ProceduralLinear layers."""
    def __init__(self, n_embed, n_head, layer_idx):
        super().__init__()
        self.num_heads = n_head
        self.head_size = n_embed // n_head
        
        self.query = ProceduralLinear(n_embed, n_embed, bias=False)
        self.key = ProceduralLinear(n_embed, n_embed, bias=False)
        self.value = ProceduralLinear(n_embed, n_embed, bias=False)
        self.proj = ProceduralLinear(n_embed, n_embed, bias=False)
        
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
    def __init__(self, n_embed, layer_idx):
        super().__init__()
        self.up = ProceduralLinear(n_embed, 4 * n_embed, bias=False)
        self.down = ProceduralLinear(4 * n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = F.relu(x)
        x = self.down(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_idx):
        super().__init__()
        self.sa = MultiheadAttentionBatch(n_embed, n_head, layer_idx)
        self.ffwd = FeedForward(n_embed, layer_idx)
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
            Block(n_embed, n_head, i) 
            for i in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed) 
        self.lm_head = ProceduralLinear(n_embed, vocab_size, bias=False)

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

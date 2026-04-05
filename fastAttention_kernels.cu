/*
 * fastAttention_kernels.cu
 *
 * Implements transformer attention kernels in CUDA:
 *   Week 1 – separate naive kernels (GEMM, mask, softmax, output GEMM)
 *   Week 2 – single fused kernel with shared-memory tiling + warp primitives
 *   Week 3 – mixed-precision kernel using Tensor Cores (FP16 GEMM, FP32 softmax)
 *   Week 4 – block-sparse attention via CSR mask
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE  32
#define WMMA_M     16
#define WMMA_N     16
#define WMMA_K     16

using namespace nvcuda;

/* ─────────────────────────────────────────────
   Warp-level reduction helpers
   ───────────────────────────────────────────── */

__device__ __forceinline__ float warpMax(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warpSum(float v) {
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

/* ─────────────────────────────────────────────
   WEEK 1 – Naive (separate) kernels
   ───────────────────────────────────────────── */

/*
 * naiveGEMMKernel
 * scores[i][j] = dot(Q[i], K[j]) * scale
 * Q, K are both [seq_len x embed_dim] row-major.
 */
__global__ void naiveGEMMKernel(float* scores,
                                 const float* Q, const float* K,
                                 int seq_len, int embed_dim, float scale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;

    float dot = 0.f;
    for (int d = 0; d < embed_dim; d++)
        dot += Q[row * embed_dim + d] * K[col * embed_dim + d];

    scores[row * seq_len + col] = dot * scale;
}

/*
 * applyMaskKernel – causal (lower-triangular) mask.
 * Positions where col > row are set to a large negative number
 * to drive exp() toward 0 without producing NaN.
 */
__global__ void applyMaskKernel(float* scores, int seq_len)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;

    if (col > row)
        scores[row * seq_len + col] = -1e20f;
}

/*
 * softmaxKernel – one block per row, warp-level max/sum reductions.
 * Shared memory: one float per warp for the reduce scratch.
 */
__global__ void softmaxKernel(float* scores, int seq_len)
{
    int row   = blockIdx.x;
    int tid   = threadIdx.x;
    int lane  = tid % WARP_SIZE;
    int wid   = tid / WARP_SIZE;
    int nwarp = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];  // nwarp floats

    /* pass 1: find max */
    float lmax = -1e38f;
    for (int j = tid; j < seq_len; j += blockDim.x)
        lmax = fmaxf(lmax, scores[row * seq_len + j]);
    lmax = warpMax(lmax);
    if (lane == 0) smem[wid] = lmax;
    __syncthreads();

    float rmax = (tid < nwarp) ? smem[tid] : -1e38f;
    rmax = warpMax(rmax);
    /* broadcast row max through smem so all threads agree */
    if (tid == 0) smem[0] = rmax;
    __syncthreads();
    rmax = smem[0];

    /* pass 2: exp(x - max), sum */
    float lsum = 0.f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float e = expf(scores[row * seq_len + j] - rmax);
        scores[row * seq_len + j] = e;
        lsum += e;
    }
    lsum = warpSum(lsum);
    if (lane == 0) smem[wid] = lsum;
    __syncthreads();

    float rsum = (tid < nwarp) ? smem[tid] : 0.f;
    rsum = warpSum(rsum);
    if (tid == 0) smem[0] = rsum;
    __syncthreads();
    rsum = smem[0];

    /* normalize */
    for (int j = tid; j < seq_len; j += blockDim.x)
        scores[row * seq_len + j] /= rsum;
}

/*
 * outputGEMMKernel – output[i][d] = sum_j( attn[i][j] * V[j][d] )
 */
__global__ void outputGEMMKernel(float* output,
                                   const float* attn, const float* V,
                                   int seq_len, int embed_dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= embed_dim) return;

    float acc = 0.f;
    for (int k = 0; k < seq_len; k++)
        acc += attn[row * seq_len + k] * V[k * embed_dim + col];

    output[row * embed_dim + col] = acc;
}

/* Host wrapper exposed to Python */
void naiveAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);

    float* d_scores;
    cudaMalloc(&d_scores, (size_t)seq_len * seq_len * sizeof(float));

    dim3 blk2(16, 16);
    dim3 grd2((seq_len + 15) / 16, (seq_len + 15) / 16);
    naiveGEMMKernel<<<grd2, blk2>>>(d_scores,
        Q.data_ptr<float>(), K.data_ptr<float>(), seq_len, embed_dim, scale);

    applyMaskKernel<<<grd2, blk2>>>(d_scores, seq_len);

    /* round up to next warp multiple so all warps in the block are full */
    int sft = ((min(seq_len, 256) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int nw  = sft / WARP_SIZE;
    softmaxKernel<<<seq_len, sft, nw * sizeof(float)>>>(d_scores, seq_len);

    dim3 oblk(16, 16);
    dim3 ogrd((embed_dim + 15) / 16, (seq_len + 15) / 16);
    outputGEMMKernel<<<ogrd, oblk>>>(output.data_ptr<float>(),
        d_scores, V.data_ptr<float>(), seq_len, embed_dim);

    cudaFree(d_scores);
}

/* ─────────────────────────────────────────────
   WEEK 2 – Fused kernel
   Stores all scores in shared memory (one block per query row).
   Eliminates global memory round-trips between GEMM → mask → softmax → V.

   Key fix over naive approach: all threads compute their assigned key scores
   in parallel (each thread owns seq_len/bdim keys), with only O(log bdim)
   syncs for the softmax reductions — not O(seq_len) syncs.
   ───────────────────────────────────────────── */

__global__ void fusedAttentionKernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ output,
                                      int seq_len, int embed_dim, float scale)
{
    int qi   = blockIdx.x;
    int tid  = threadIdx.x;
    int bdim = blockDim.x;

    /*
     * Shared memory layout:
     *   [0 .. seq_len-1]      : attention scores for query qi
     *   [seq_len .. seq_len+bdim-1] : per-thread reduction scratch
     */
    extern __shared__ float smem[];
    float* scores  = smem;
    float* scratch = smem + seq_len;

    /* ── Phase 1: each thread computes scores for its key positions in parallel ──
     * Thread t handles keys t, t+bdim, t+2*bdim, ...
     * Each dot product is computed independently (embed_dim loop per thread).
     * This replaces the old serial j-loop with O(seq_len) __syncthreads().
     */
    for (int kj = tid; kj < seq_len; kj += bdim) {
        if (kj <= qi) {
            float dot = 0.f;
            for (int d = 0; d < embed_dim; d++)
                dot += Q[qi * embed_dim + d] * K[kj * embed_dim + d];
            scores[kj] = dot * scale;
        } else {
            scores[kj] = -1e20f;
        }
    }
    __syncthreads();

    /* ── Phase 2: parallel softmax with tree reduction ── */
    /* find max */
    float lmax = -1e38f;
    for (int kj = tid; kj < seq_len; kj += bdim)
        lmax = fmaxf(lmax, scores[kj]);
    scratch[tid] = lmax;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }
    float rmax = scratch[0];

    /* exp and partial sum */
    float lsum = 0.f;
    for (int kj = tid; kj < seq_len; kj += bdim) {
        float e = expf(scores[kj] - rmax);
        scores[kj] = e;
        lsum += e;
    }
    scratch[tid] = lsum;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    float rsum = scratch[0];
    __syncthreads();

    /* normalize */
    for (int kj = tid; kj < seq_len; kj += bdim)
        scores[kj] /= rsum;
    __syncthreads();

    /* ── Phase 3: output[qi][d] = sum_kj( scores[kj] * V[kj][d] ) ── */
    for (int d = tid; d < embed_dim; d += bdim) {
        float acc = 0.f;
        for (int kj = 0; kj < seq_len; kj++)
            acc += scores[kj] * V[kj * embed_dim + d];
        output[qi * embed_dim + d] = acc;
    }
}

void fusedAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);

    const int BLOCK = 128;
    /* scores (seq_len) + reduction scratch (BLOCK) */
    size_t smem = ((size_t)seq_len + BLOCK) * sizeof(float);

    fusedAttentionKernel<<<seq_len, BLOCK, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), seq_len, embed_dim, scale);
}

/* ─────────────────────────────────────────────
   WEEK 3 – Mixed-precision with Tensor Cores
   FP16 for GEMM computation, FP32 for softmax and output accumulation.
   ───────────────────────────────────────────── */

/* Convert a flat float array to half in-place (new allocation) */
__global__ void fp32ToHalfKernel(__half* dst, const float* src, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}

/*
 * tcScoreKernel – computes score tile = Q_fp16 @ K_fp16^T using WMMA.
 * One warp per 16×16 output tile.
 * Grid: (ceil(seq_len/16), ceil(seq_len/16))  Threads: 32
 */
__global__ void tcScoreKernel(float* scores,
                               const __half* Q, const __half* K,
                               int seq_len, int embed_dim, float scale)
{
    int qi_start = blockIdx.y * WMMA_M;
    int kj_start = blockIdx.x * WMMA_N;
    if (qi_start >= seq_len || kj_start >= seq_len) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.f);

    /* accumulate across embed_dim in chunks of WMMA_K=16 */
    for (int d = 0; d < embed_dim; d += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> qa;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> kb;

        /* col_major B gives us K^T without an explicit transpose */
        wmma::load_matrix_sync(qa, Q + qi_start * embed_dim + d, embed_dim);
        wmma::load_matrix_sync(kb, K + kj_start * embed_dim + d, embed_dim);
        wmma::mma_sync(acc, qa, kb, acc);
    }

    /* apply scale and write – positions not yet allocated just get overwritten by mask */
    for (int i = 0; i < (int)acc.num_elements; i++)
        acc.x[i] *= scale;

    wmma::store_matrix_sync(scores + qi_start * seq_len + kj_start,
                             acc, seq_len, wmma::mem_row_major);
}

/*
 * tcOutputKernel – output tile = attn_fp16 @ V_fp16 using WMMA.
 * Grid: (ceil(embed_dim/16), ceil(seq_len/16))  Threads: 32
 */
__global__ void tcOutputKernel(float* output,
                                const __half* attn, const __half* V,
                                int seq_len, int embed_dim)
{
    int qi_start = blockIdx.y * WMMA_M;
    int d_start  = blockIdx.x * WMMA_N;
    if (qi_start >= seq_len || d_start >= embed_dim) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int kj = 0; kj < seq_len; kj += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> vf;

        wmma::load_matrix_sync(af, attn + qi_start * seq_len + kj, seq_len);
        wmma::load_matrix_sync(vf, V + kj * embed_dim + d_start, embed_dim);
        wmma::mma_sync(acc, af, vf, acc);
    }

    wmma::store_matrix_sync(output + qi_start * embed_dim + d_start,
                             acc, embed_dim, wmma::mem_row_major);
}

void tcFusedAttention(torch::Tensor Q, torch::Tensor K,
                      torch::Tensor V, torch::Tensor output)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);

    int n  = seq_len * embed_dim;
    int n2 = seq_len * seq_len;

    /* allocate FP16 copies of Q, K, V and a FP32 score buffer */
    __half *qh, *kh, *vh, *attn_h;
    float  *scores;
    cudaMalloc(&qh,     n  * sizeof(__half));
    cudaMalloc(&kh,     n  * sizeof(__half));
    cudaMalloc(&vh,     n  * sizeof(__half));
    cudaMalloc(&attn_h, n2 * sizeof(__half));
    cudaMalloc(&scores, n2 * sizeof(float));

    int blk = 256;
    fp32ToHalfKernel<<<(n + blk - 1) / blk, blk>>>(qh, Q.data_ptr<float>(), n);
    fp32ToHalfKernel<<<(n + blk - 1) / blk, blk>>>(kh, K.data_ptr<float>(), n);
    fp32ToHalfKernel<<<(n + blk - 1) / blk, blk>>>(vh, V.data_ptr<float>(), n);

    /* step 1: scores = Q_fp16 @ K_fp16^T  (FP32 accumulator) */
    dim3 sg((seq_len + WMMA_N - 1) / WMMA_N, (seq_len + WMMA_M - 1) / WMMA_M);
    tcScoreKernel<<<sg, WARP_SIZE>>>(scores, qh, kh, seq_len, embed_dim, scale);

    /* step 2: causal mask */
    dim3 blk2d(16, 16);
    dim3 grd2d((seq_len + 15) / 16, (seq_len + 15) / 16);
    applyMaskKernel<<<grd2d, blk2d>>>(scores, seq_len);

    /* step 3: softmax in FP32 */
    int sft = ((min(seq_len, 256) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int nw  = sft / WARP_SIZE;
    softmaxKernel<<<seq_len, sft, nw * sizeof(float)>>>(scores, seq_len);

    /* step 4: convert softmax output to FP16 for the tensor-core output GEMM */
    fp32ToHalfKernel<<<(n2 + blk - 1) / blk, blk>>>(attn_h, scores, n2);

    /* step 5: output = attn_fp16 @ V_fp16  (FP32 accumulator) */
    dim3 og((embed_dim + WMMA_N - 1) / WMMA_N, (seq_len + WMMA_M - 1) / WMMA_M);
    tcOutputKernel<<<og, WARP_SIZE>>>(output.data_ptr<float>(),
                                       attn_h, vh, seq_len, embed_dim);

    cudaFree(qh); cudaFree(kh); cudaFree(vh);
    cudaFree(attn_h); cudaFree(scores);
}

/* ─────────────────────────────────────────────
   WEEK 4 – Block-Sparse Attention
   Uses block-CSR mask: only non-zero blocks are computed.
   Each CUDA block handles one row of query blocks.
   Within each block-row, threads each own one query token.
   ───────────────────────────────────────────── */

/*
 * sparseAttentionKernel
 *
 * One warp per query token: lane threads parallelize the embed_dim dot product.
 * Online softmax in a single pass over non-zero CSR blocks.
 * Output accumulation in shared memory (avoids hardcoded register array).
 *
 * Grid : (n_row_blocks,)
 * Block: block_h * WARP_SIZE threads  (block_h warps, one warp per query)
 * Smem : block_h * embed_dim floats
 */
__global__ void sparseAttentionKernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ output,
                                       const int* row_ptr, const int* col_idx,
                                       int seq_len, int embed_dim, float scale,
                                       int block_h, int block_w)
{
    int qi_block = blockIdx.x;
    int warp_id  = threadIdx.x / WARP_SIZE;  /* which query within block */
    int lane     = threadIdx.x % WARP_SIZE;  /* lane within warp          */
    int qi       = qi_block * block_h + warp_id;
    if (qi >= seq_len) return;

    /* each warp owns a contiguous embed_dim slice of shared memory */
    extern __shared__ float smem[];
    float* warp_out = smem + warp_id * embed_dim;
    for (int d = lane; d < embed_dim; d += WARP_SIZE)
        warp_out[d] = 0.f;
    __syncwarp();

    int b_start = row_ptr[qi_block];
    int b_end   = row_ptr[qi_block + 1];

    /* online softmax: running max + rescaled accumulator (single pass) */
    float max_val = -1e38f;
    float sum_exp = 0.f;

    for (int b = b_start; b < b_end; b++) {
        int kj_blk = col_idx[b];
        int kj0    = kj_blk * block_w;
        int kj1    = min(kj0 + block_w, seq_len);
        for (int kj = kj0; kj < kj1; kj++) {
            if (kj > qi) continue;

            /* parallel dot product: each lane sums over d, d+32, d+64 … */
            float dot = 0.f;
            for (int d = lane; d < embed_dim; d += WARP_SIZE)
                dot += Q[qi * embed_dim + d] * K[kj * embed_dim + d];
            /* warpSum broadcasts the full dot product to every lane */
            dot = warpSum(dot) * scale;

            /* update running max and rescale accumulator if needed */
            if (dot > max_val) {
                float rescale = expf(max_val - dot);
                sum_exp *= rescale;
                for (int d = lane; d < embed_dim; d += WARP_SIZE)
                    warp_out[d] *= rescale;
                max_val = dot;
            }
            float w = expf(dot - max_val);
            sum_exp += w;
            for (int d = lane; d < embed_dim; d += WARP_SIZE)
                warp_out[d] += w * V[kj * embed_dim + d];
        }
    }

    /* normalize and write output */
    for (int d = lane; d < embed_dim; d += WARP_SIZE)
        output[qi * embed_dim + d] = warp_out[d] / sum_exp;
}

void sparseAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                     torch::Tensor output,
                     torch::Tensor row_ptr, torch::Tensor col_idx,
                     int block_h, int block_w)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);
    int n_row_blk = (seq_len + block_h - 1) / block_h;

    /* block_h warps per CUDA block, WARP_SIZE threads per warp */
    int threads = block_h * WARP_SIZE;
    size_t smem = (size_t)block_h * embed_dim * sizeof(float);

    sparseAttentionKernel<<<n_row_blk, threads, smem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(),
        row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(),
        seq_len, embed_dim, scale, block_h, block_w);
}

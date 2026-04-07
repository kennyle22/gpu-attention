#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

// warp size is always 32 on current nvidia hardware
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

// butterfly reduction helpers for warp-level ops
__device__ __forceinline__ float warpMax(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// -------------------------------------------------------
// Week 1: naive separate kernels
// -------------------------------------------------------

// compute scores[i][j] = dot(Q[i], K[j]) / sqrt(d)
// one thread per output element, straightforward 2d grid
__global__ void naiveGEMMKernel(float* scores,
                                 const float* Q, const float* K,
                                 int seq_len, int embed_dim, float scale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;

    float s = 0.f;
    for (int d = 0; d < embed_dim; d++)
        s += Q[row * embed_dim + d] * K[col * embed_dim + d];
    scores[row * seq_len + col] = s * scale;
}

// causal mask: zero out upper triangle so future tokens don't attend
// use -1e20 instead of -inf to avoid nan in exp
__global__ void applyMaskKernel(float* scores, int seq_len)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len || col >= seq_len) return;
    if (col > row)
        scores[row * seq_len + col] = -1e20f;
}

// row-wise softmax using shared mem for the warp partials
// one block per row, threads stride over columns
__global__ void softmaxKernel(float* scores, int seq_len)
{
    int row  = blockIdx.x;
    int tid  = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;
    int nw   = blockDim.x / WARP_SIZE;

    extern __shared__ float smem[];

    // find row max
    float mx = -1e38f;
    for (int j = tid; j < seq_len; j += blockDim.x)
        mx = fmaxf(mx, scores[row * seq_len + j]);
    mx = warpMax(mx);
    if (lane == 0) smem[wid] = mx;
    __syncthreads();

    float rmx = (tid < nw) ? smem[tid] : -1e38f;
    rmx = warpMax(rmx);
    if (tid == 0) smem[0] = rmx;
    __syncthreads();
    rmx = smem[0];

    // subtract max, exponentiate, sum
    float s = 0.f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float e = expf(scores[row * seq_len + j] - rmx);
        scores[row * seq_len + j] = e;
        s += e;
    }
    s = warpSum(s);
    if (lane == 0) smem[wid] = s;
    __syncthreads();

    float rs = (tid < nw) ? smem[tid] : 0.f;
    rs = warpSum(rs);
    if (tid == 0) smem[0] = rs;
    __syncthreads();
    rs = smem[0];

    for (int j = tid; j < seq_len; j += blockDim.x)
        scores[row * seq_len + j] /= rs;
}

// out[i][d] = sum_j attn[i][j] * V[j][d]
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

void naiveAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);

    float* d_scores;
    cudaMalloc(&d_scores, (size_t)seq_len * seq_len * sizeof(float));

    dim3 blk(16, 16);
    dim3 grd((seq_len + 15) / 16, (seq_len + 15) / 16);
    naiveGEMMKernel<<<grd, blk>>>(d_scores,
        Q.data_ptr<float>(), K.data_ptr<float>(), seq_len, embed_dim, scale);
    applyMaskKernel<<<grd, blk>>>(d_scores, seq_len);

    // round threads up to warp boundary
    int sft = ((min(seq_len, 256) + 31) / 32) * 32;
    softmaxKernel<<<seq_len, sft, (sft / 32) * sizeof(float)>>>(d_scores, seq_len);

    dim3 oblk(16, 16);
    dim3 ogrd((embed_dim + 15) / 16, (seq_len + 15) / 16);
    outputGEMMKernel<<<ogrd, oblk>>>(output.data_ptr<float>(),
        d_scores, V.data_ptr<float>(), seq_len, embed_dim);

    cudaFree(d_scores);
}

// -------------------------------------------------------
// Week 2: fused tiled kernel
//
// Instead of writing scores to global mem between steps, we keep
// everything in shared mem. One block owns TILE_Q query rows,
// each warp handles one row.
//
// We tile over K/V in chunks of TILE_KV rows, loading both K and V
// at once so we only pay for one global load pass per tile.
// Causal skip: if the whole KV tile is past qi_max, skip it.
// Online softmax lets us do this in one pass without storing scores.
// -------------------------------------------------------

#define TILE_Q  32
#define TILE_KV 32

__global__ void fusedAttentionKernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ output,
                                      int seq_len, int embed_dim, float scale)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int qi      = blockIdx.x * TILE_Q + warp_id;
    int qi_max  = min((blockIdx.x + 1) * TILE_Q - 1, seq_len - 1);

    extern __shared__ float smem[];
    float* Qs  = smem;
    float* Ks  = smem + TILE_Q * embed_dim;
    float* Vs  = smem + (TILE_Q + TILE_KV) * embed_dim;
    float* out = smem + (TILE_Q + 2 * TILE_KV) * embed_dim;

    // load this block's Q rows + zero output buffer
    for (int i = threadIdx.x; i < TILE_Q * embed_dim; i += blockDim.x) {
        int r = i / embed_dim, d = i % embed_dim;
        int g = blockIdx.x * TILE_Q + r;
        Qs[i]  = (g < seq_len) ? Q[g * embed_dim + d] : 0.f;
        out[i] = 0.f;
    }
    __syncthreads();

    if (qi >= seq_len) return;

    float mi = -1e38f;  // running max for online softmax
    float li = 0.f;     // running normalizer

    // iterate over KV tiles, skip anything entirely in the future
    for (int kv0 = 0; kv0 <= qi_max; kv0 += TILE_KV) {
        int tsz = min(TILE_KV, seq_len - kv0);

        // coload K and V for this tile
        for (int i = threadIdx.x; i < TILE_KV * embed_dim; i += blockDim.x) {
            int kl = i / embed_dim, d = i % embed_dim;
            int kj = kv0 + kl;
            Ks[i] = (kj < seq_len) ? K[kj * embed_dim + d] : 0.f;
            Vs[i] = (kj < seq_len) ? V[kj * embed_dim + d] : 0.f;
        }
        __syncthreads();

        for (int kl = 0; kl < tsz; kl++) {
            int kj = kv0 + kl;
            float dot = -1e20f;
            if (kj <= qi) {
                float a = 0.f;
                for (int d = lane; d < embed_dim; d += WARP_SIZE)
                    a += Qs[warp_id * embed_dim + d] * Ks[kl * embed_dim + d];
                dot = warpSum(a) * scale;
            }
            // update running max, rescale existing output if needed
            if (dot > mi) {
                float rs = expf(mi - dot);
                li *= rs;
                for (int d = lane; d < embed_dim; d += WARP_SIZE)
                    out[warp_id * embed_dim + d] *= rs;
                mi = dot;
            }
            float w = expf(dot - mi);
            li += w;
            for (int d = lane; d < embed_dim; d += WARP_SIZE)
                out[warp_id * embed_dim + d] += w * Vs[kl * embed_dim + d];
        }
        __syncthreads();
    }

    for (int d = lane; d < embed_dim; d += WARP_SIZE)
        output[qi * embed_dim + d] = out[warp_id * embed_dim + d] / li;
}

void fusedAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);

    int nblocks = (seq_len + TILE_Q - 1) / TILE_Q;
    int nthreads = TILE_Q * WARP_SIZE;
    size_t smem_sz = (size_t)(2 * TILE_Q + 2 * TILE_KV) * embed_dim * sizeof(float);

    fusedAttentionKernel<<<nblocks, nthreads, smem_sz>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), seq_len, embed_dim, scale);
}

// -------------------------------------------------------
// Week 3: tensor core mixed precision
// FP16 for the GEMMs, FP32 for softmax (needed for numerical stability)
// -------------------------------------------------------

__global__ void cvtHalfKernel(__half* dst, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Q @ K^T tile using wmma, one warp per 16x16 output tile
// using col_major for K avoids needing an explicit transpose
__global__ void tcScoreKernel(float* scores,
                               const __half* Q, const __half* K,
                               int seq_len, int embed_dim, float scale)
{
    int qi0 = blockIdx.y * WMMA_M;
    int kj0 = blockIdx.x * WMMA_N;
    if (qi0 >= seq_len || kj0 >= seq_len) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int d = 0; d < embed_dim; d += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> qa;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> kb;
        wmma::load_matrix_sync(qa, Q + qi0 * embed_dim + d, embed_dim);
        wmma::load_matrix_sync(kb, K + kj0 * embed_dim + d, embed_dim);
        wmma::mma_sync(acc, qa, kb, acc);
    }

    for (int i = 0; i < (int)acc.num_elements; i++)
        acc.x[i] *= scale;

    wmma::store_matrix_sync(scores + qi0 * seq_len + kj0,
                             acc, seq_len, wmma::mem_row_major);
}

// attn @ V tile using wmma
__global__ void tcOutputKernel(float* output,
                                const __half* attn, const __half* V,
                                int seq_len, int embed_dim)
{
    int qi0 = blockIdx.y * WMMA_M;
    int d0  = blockIdx.x * WMMA_N;
    if (qi0 >= seq_len || d0 >= embed_dim) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int k = 0; k < seq_len; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(af, attn + qi0 * seq_len + k, seq_len);
        wmma::load_matrix_sync(vf, V + k * embed_dim + d0, embed_dim);
        wmma::mma_sync(acc, af, vf, acc);
    }

    wmma::store_matrix_sync(output + qi0 * embed_dim + d0,
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

    __half *qh, *kh, *vh, *ah;
    float  *scores;
    cudaMalloc(&qh,     n  * sizeof(__half));
    cudaMalloc(&kh,     n  * sizeof(__half));
    cudaMalloc(&vh,     n  * sizeof(__half));
    cudaMalloc(&ah,     n2 * sizeof(__half));
    cudaMalloc(&scores, n2 * sizeof(float));

    int cvt_blk = 256;
    cvtHalfKernel<<<(n + cvt_blk-1)/cvt_blk, cvt_blk>>>(qh, Q.data_ptr<float>(), n);
    cvtHalfKernel<<<(n + cvt_blk-1)/cvt_blk, cvt_blk>>>(kh, K.data_ptr<float>(), n);
    cvtHalfKernel<<<(n + cvt_blk-1)/cvt_blk, cvt_blk>>>(vh, V.data_ptr<float>(), n);

    dim3 sg((seq_len + WMMA_N-1)/WMMA_N, (seq_len + WMMA_M-1)/WMMA_M);
    tcScoreKernel<<<sg, WARP_SIZE>>>(scores, qh, kh, seq_len, embed_dim, scale);

    dim3 mblk(16, 16);
    dim3 mgrd((seq_len+15)/16, (seq_len+15)/16);
    applyMaskKernel<<<mgrd, mblk>>>(scores, seq_len);

    int sft = ((min(seq_len, 256) + 31) / 32) * 32;
    softmaxKernel<<<seq_len, sft, (sft/32)*sizeof(float)>>>(scores, seq_len);

    // convert softmax result to fp16 for output gemm
    cvtHalfKernel<<<(n2 + cvt_blk-1)/cvt_blk, cvt_blk>>>(ah, scores, n2);

    dim3 og((embed_dim + WMMA_N-1)/WMMA_N, (seq_len + WMMA_M-1)/WMMA_M);
    tcOutputKernel<<<og, WARP_SIZE>>>(output.data_ptr<float>(), ah, vh, seq_len, embed_dim);

    cudaFree(qh); cudaFree(kh); cudaFree(vh);
    cudaFree(ah); cudaFree(scores);
}

// -------------------------------------------------------
// Week 4: block-sparse attention via CSR mask
//
// Only compute attention over non-zero blocks defined by the CSR structure.
// One warp per query token so the embed_dim dot product is parallelized
// across 32 lanes instead of being fully serial.
// Online softmax handles the single-pass weighted sum.
//
// TODO: could try loading K/V blocks into smem to reduce global traffic
// -------------------------------------------------------

__global__ void sparseAttentionKernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ output,
                                       const int* row_ptr, const int* col_idx,
                                       int seq_len, int embed_dim, float scale,
                                       int block_h, int block_w)
{
    int qblk    = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int qi      = qblk * block_h + warp_id;
    if (qi >= seq_len) return;

    extern __shared__ float smem[];
    float* acc_out = smem + warp_id * embed_dim;
    for (int d = lane; d < embed_dim; d += WARP_SIZE)
        acc_out[d] = 0.f;
    __syncwarp();

    float mval = -1e38f;
    float zsum = 0.f;

    int b0 = row_ptr[qblk];
    int b1 = row_ptr[qblk + 1];

    for (int b = b0; b < b1; b++) {
        int cblk = col_idx[b];
        int kj0  = cblk * block_w;
        int kj1  = min(kj0 + block_w, seq_len);

        for (int kj = kj0; kj < kj1; kj++) {
            if (kj > qi) continue;

            float dot = 0.f;
            for (int d = lane; d < embed_dim; d += WARP_SIZE)
                dot += Q[qi * embed_dim + d] * K[kj * embed_dim + d];
            dot = warpSum(dot) * scale;

            if (dot > mval) {
                float rs = expf(mval - dot);
                zsum *= rs;
                for (int d = lane; d < embed_dim; d += WARP_SIZE)
                    acc_out[d] *= rs;
                mval = dot;
            }
            float w = expf(dot - mval);
            zsum += w;
            for (int d = lane; d < embed_dim; d += WARP_SIZE)
                acc_out[d] += w * V[kj * embed_dim + d];
        }
    }

    for (int d = lane; d < embed_dim; d += WARP_SIZE)
        output[qi * embed_dim + d] = acc_out[d] / zsum;
}

void sparseAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                     torch::Tensor output,
                     torch::Tensor row_ptr, torch::Tensor col_idx,
                     int block_h, int block_w)
{
    int seq_len   = Q.size(0);
    int embed_dim = Q.size(1);
    float scale   = 1.f / sqrtf((float)embed_dim);
    int nrb = (seq_len + block_h - 1) / block_h;

    int nth  = block_h * WARP_SIZE;
    size_t smem_sz = (size_t)block_h * embed_dim * sizeof(float);

    sparseAttentionKernel<<<nrb, nth, smem_sz>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(),
        row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(),
        seq_len, embed_dim, scale, block_h, block_w);
}

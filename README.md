# gpu-attention

Four CUDA implementations of causal self-attention written for a GPU programming course at UCI. Each week builds on the previous one.

## What's in here

Week 1: separate kernels for Q×K^T, causal masking, row-wise softmax, and attn×V. Four global memory round trips. Correct, not fast.

Week 2: fused into a single kernel. Q, K, V tile into shared memory; softmax runs online so the full score matrix never materializes in global memory. One block per TILE_Q query rows, one warp per row.

Week 3: FP16 GEMMs via WMMA tensor core intrinsics, FP32 softmax for numerical stability. Requires sm_70 or later.

Week 4: block-sparse. Takes a CSR-format block mask and skips fully masked blocks. Causal mask by default; random sparsity supported for testing.

## Build

CUDA and PyTorch required. Week 3 needs a Volta or newer GPU (sm_70 minimum).

```bash
source /path/to/your/venv/bin/activate
bash build.sh
```

Or directly:
```bash
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0" python setup.py install
```

`build.sh` has a hardcoded course venv path. Update it for your environment.

## Run

```bash
python test_driver.py --seq_len 1024 --embed_dim 64
python test_driver.py --seq_len 2048 --embed_dim 64 --profile
```

The test driver checks each variant against PyTorch's math backend (not FlashAttention) and prints timing and speedup numbers for all four.

## Known issue: Week 3 numerical accuracy

The tensor core variant has a numerical discrepancy I didn't resolve. The error tolerance is set to 1e-2, which is looser than the FP32 kernels (1e-4 and 1e-3). FP16 accumulation in the WMMA tiles drifts more than expected against the float reference. It passes the check but the margin is wider than it should be.

## Implementation notes

Softmax uses a two-level butterfly reduction: warpMax for the row max, then warpSum for normalization. The Week 2 kernel uses the online softmax formulation (Milakov & Gimelshein) to handle everything in one pass over K/V.

The Week 4 sparse kernel reads K and V directly from global memory per non-zero key. Staging into shared memory first would reduce bandwidth pressure — that's the next thing to try.

setup.py targets sm_70 (V100), sm_75 (T4), sm_80 (A100).

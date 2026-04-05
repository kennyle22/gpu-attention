"""
test_driver.py – benchmark and correctness harness for all attention variants.

Usage:
    python test_driver.py --seq_len 1024 --embed_dim 64
    python test_driver.py --seq_len 2048 --embed_dim 64
    python test_driver.py --seq_len 4096 --embed_dim 64
"""

import argparse
import math
import numpy as np
import torch
import torch.profiler as profiler
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import fastAttention


# ──────────────────────────────────────────────────────────────
#  Block-sparse mask utilities
# ──────────────────────────────────────────────────────────────

class CSRMatrix:
    """Block-CSR mask: row_ptr tracks non-zero block counts per block-row."""
    def __init__(self, row_ptr, col_idx, n_row_blocks, n_col_blocks):
        self.row_ptr     = row_ptr
        self.col_idx     = col_idx
        self.n_row_blocks = n_row_blocks
        self.n_col_blocks = n_col_blocks


def gen_causal_mask(seq_len, block_h, block_w):
    """
    Generate a block-causal CSR mask: block (r, c) is non-zero iff c <= r.
    This is the lower-triangular pattern in block space, matching the
    element-level causal mask we use in the dense kernels.
    """
    n_row_blk = (seq_len + block_h - 1) // block_h
    n_col_blk = (seq_len + block_w - 1) // block_w

    row_ptr_list = [0]
    col_idx_list = []
    for r in range(n_row_blk):
        cnt = 0
        for c in range(n_col_blk):
            # include block only if it can contain valid (causal) positions
            if c <= r:
                col_idx_list.append(c)
                cnt += 1
        row_ptr_list.append(row_ptr_list[-1] + cnt)

    row_ptr = torch.IntTensor(row_ptr_list)
    col_idx = torch.IntTensor(col_idx_list)
    return CSRMatrix(row_ptr, col_idx, n_row_blk, n_col_blk)


def gen_random_mask(seq_len, sparsity, block_h, block_w):
    """
    Generate a random block-sparse mask in CSR format.
    sparsity = fraction of blocks that are NON-zero (so 0.5 means half computed).
    """
    assert 0.0 < sparsity < 1.0
    n_row_blk = math.ceil(seq_len / block_h)
    n_col_blk = math.ceil(seq_len / block_w)
    total     = n_row_blk * n_col_blk

    kept = np.random.choice(total, int(total * sparsity), replace=False)
    kept = np.sort(kept)

    row_indices = kept // n_col_blk
    col_indices = kept %  n_col_blk

    row_ptr = np.zeros(n_row_blk + 1, dtype=np.int32)
    for r in row_indices:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    return CSRMatrix(
        torch.IntTensor(row_ptr),
        torch.IntTensor(col_indices),
        n_row_blk, n_col_blk
    )


def csr_to_dense_mask(mask, seq_len, block_h, block_w):
    """Convert a block-CSR mask to a float additive mask for PyTorch reference."""
    n_row_blk = mask.row_ptr.size(0) - 1
    # start with -inf everywhere; zero out included blocks
    dense = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device='cuda')
    for r in range(n_row_blk):
        for b in range(mask.row_ptr[r], mask.row_ptr[r + 1]):
            c    = mask.col_idx[b].item()
            r0, r1 = r * block_h, min((r + 1) * block_h, seq_len)
            c0, c1 = c * block_w, min((c + 1) * block_w, seq_len)
            dense[r0:r1, c0:c1] = 0.0   # 0 = keep, -inf = mask out
    return dense


# ──────────────────────────────────────────────────────────────
#  PyTorch reference (math backend so timing is fair)
# ──────────────────────────────────────────────────────────────

def pytorch_reference(Q, K, V, attn_mask=None, warmup=5, niters=20):
    """
    Run F.scaled_dot_product_attention with the MATH backend (no FlashAttn).
    Returns (output, avg_ms).
    """
    Qu = Q.unsqueeze(0).unsqueeze(0)
    Ku = K.unsqueeze(0).unsqueeze(0)
    Vu = V.unsqueeze(0).unsqueeze(0)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        for _ in range(warmup):
            out = F.scaled_dot_product_attention(Qu, Ku, Vu,
                                                  attn_mask=attn_mask,
                                                  dropout_p=0.0,
                                                  scale=None)  # scale = 1/sqrt(d)
        start.record()
        for _ in range(niters):
            out = F.scaled_dot_product_attention(Qu, Ku, Vu,
                                                  attn_mask=attn_mask,
                                                  dropout_p=0.0,
                                                  scale=None)
        end.record()

    end.synchronize()
    ms = start.elapsed_time(end) / niters
    return out.squeeze(0).squeeze(0), ms


# ──────────────────────────────────────────────────────────────
#  Timing helper
# ──────────────────────────────────────────────────────────────

def time_kernel(fn, warmup=5, niters=20):
    """Returns average kernel time in milliseconds using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(niters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / niters


def rel_error(a, b):
    return (torch.norm(a - b) / torch.norm(b)).item()


def throughput(seq_len, ms):
    """sequences / second"""
    return 1000.0 / ms   # 1 sequence per ms, convert to /s


# ──────────────────────────────────────────────────────────────
#  Per-week test functions
# ──────────────────────────────────────────────────────────────

def test_week1(Q, K, V, ref, ref_ms, seq_len):
    print("\n── Week 1: Naive Attention ──")
    out = fastAttention.naive_attention(Q, K, V)
    print("Naive Attention Output (corner):")
    print(out[:3, :4])
    err = rel_error(out, ref)
    print(f"Relative error vs. reference: {err:.4e}")
    assert err < 1e-4, f"Week 1 correctness FAILED (err={err:.4e})"
    print("Correctness: PASSED")

    ms = time_kernel(lambda: fastAttention.naive_attention(Q, K, V))
    tp = throughput(seq_len, ms)
    speedup = ref_ms / ms
    print(f"Avg CUDA time : {ms:.4f} ms")
    print(f"Throughput    : {tp:.1f} seq/s")
    print(f"PyTorch time  : {ref_ms:.4f} ms")
    print(f"Speedup       : {speedup:.4f}x")
    return ms


def test_week2(Q, K, V, ref, ref_ms, seq_len, naive_ms):
    print("\n── Week 2: Fused Attention ──")
    out = fastAttention.fused_attention(Q, K, V)
    print("Fused Attention Output (corner):")
    print(out[:3, :4])
    err = rel_error(out, ref)
    print(f"Relative error vs. reference: {err:.4e}")
    assert err < 1e-3, f"Week 2 correctness FAILED (err={err:.4e})"
    print("Correctness: PASSED")

    ms = time_kernel(lambda: fastAttention.fused_attention(Q, K, V))
    tp = throughput(seq_len, ms)
    print(f"Avg CUDA time         : {ms:.4f} ms")
    print(f"Throughput            : {tp:.1f} seq/s")
    print(f"Speedup vs. PyTorch   : {ref_ms / ms:.4f}x")
    print(f"Speedup vs. naive     : {naive_ms / ms:.4f}x")
    return ms


def test_week3(Q, K, V, ref, ref_ms, seq_len):
    print("\n── Week 3: Tensor Core (Mixed-Precision) Attention ──")
    out = fastAttention.tc_fused_attention(Q, K, V)
    print("TC Attention Output (corner):")
    print(out[:3, :4])
    err = rel_error(out, ref)
    print(f"Relative error vs. reference: {err:.4e}")
    assert err < 1e-2, f"Week 3 correctness FAILED (err={err:.4e})"
    print("Correctness: PASSED")

    ms = time_kernel(lambda: fastAttention.tc_fused_attention(Q, K, V))
    tp = throughput(seq_len, ms)
    print(f"Avg CUDA time       : {ms:.4f} ms")
    print(f"Throughput          : {tp:.1f} seq/s")
    print(f"Speedup vs. PyTorch : {ref_ms / ms:.4f}x")
    return ms


def test_week4(Q, K, V, seq_len, naive_ms, fused_ms):
    print("\n── Week 4: Block-Sparse Attention ──")

    block_h, block_w = 16, 16
    mask = gen_causal_mask(seq_len, block_h, block_w)

    # Dense causal reference (same pattern as block-causal at token level)
    causal_additive = torch.zeros(seq_len, seq_len, device='cuda')
    causal_additive = causal_additive.masked_fill(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device='cuda').triu(diagonal=1),
        float('-inf')
    )
    ref_sparse, ref_ms = pytorch_reference(Q, K, V, attn_mask=causal_additive)

    row_ptr = mask.row_ptr.cuda()
    col_idx = mask.col_idx.cuda()
    out = fastAttention.sparse_attention(Q, K, V, row_ptr, col_idx, block_h, block_w)

    print("Sparse Attention Output (corner):")
    print(out[:3, :4])
    err = rel_error(out, ref_sparse)
    print(f"Relative error vs. causal reference: {err:.4e}")
    assert err < 1e-3, f"Week 4 correctness FAILED (err={err:.4e})"
    print("Correctness: PASSED")

    ms = time_kernel(lambda: fastAttention.sparse_attention(
        Q, K, V, row_ptr, col_idx, block_h, block_w))
    tp = throughput(seq_len, ms)
    print(f"Avg CUDA time         : {ms:.4f} ms")
    print(f"Throughput            : {tp:.1f} seq/s")
    print(f"Speedup vs. naive     : {naive_ms / ms:.4f}x  (target ≥2×)")
    print(f"Speedup vs. fused     : {fused_ms / ms:.4f}x")
    print(f"Speedup vs. PyTorch   : {ref_ms / ms:.4f}x")
    return ms


# ──────────────────────────────────────────────────────────────
#  Torch profiler helper (single pass, detailed trace)
# ──────────────────────────────────────────────────────────────

def profile_variant(label, fn):
    """Run one warmup then profile a single iteration with torch.profiler."""
    fn()
    torch.cuda.synchronize()
    print(f"\nProfiling {label}:")
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=False
    ) as prof:
        fn()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main(args):
    seq_len   = args.seq_len
    embed_dim = args.embed_dim
    scale     = 1.0 / math.sqrt(embed_dim)

    print(f"seq_len={seq_len}  embed_dim={embed_dim}  scale={scale:.4f}")
    torch.manual_seed(42)

    Q = torch.randn(seq_len, embed_dim, dtype=torch.float32, device='cuda')
    K = torch.randn(seq_len, embed_dim, dtype=torch.float32, device='cuda')
    V = torch.randn(seq_len, embed_dim, dtype=torch.float32, device='cuda')

    # causal additive mask for the reference
    causal_mask = torch.zeros(seq_len, seq_len, device='cuda')
    causal_mask.masked_fill_(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device='cuda').triu(diagonal=1),
        float('-inf')
    )

    print("\nRunning PyTorch reference (causal)…")
    ref, ref_ms = pytorch_reference(Q, K, V, attn_mask=causal_mask)
    print(f"Reference time: {ref_ms:.4f} ms  |  throughput: {throughput(seq_len, ref_ms):.1f} seq/s")
    print("Reference Output (corner):")
    print(ref[:3, :4])

    naive_ms = test_week1(Q, K, V, ref, ref_ms, seq_len)
    fused_ms = test_week2(Q, K, V, ref, ref_ms, seq_len, naive_ms)
    tc_ms    = test_week3(Q, K, V, ref, ref_ms, seq_len)
    sparse_ms = test_week4(Q, K, V, seq_len, naive_ms, fused_ms)

    # ── summary table ──
    print("\n" + "=" * 60)
    print(f"{'Variant':<25} {'Time (ms)':>10} {'Seq/s':>12} {'Speedup':>10}")
    print("-" * 60)
    for label, ms in [("PyTorch (reference)", ref_ms),
                       ("Naive (Week 1)",      naive_ms),
                       ("Fused (Week 2)",      fused_ms),
                       ("TC Mixed-prec (Wk3)", tc_ms),
                       ("Block-Sparse (Wk4)",  sparse_ms)]:
        print(f"{label:<25} {ms:>10.4f} {throughput(seq_len, ms):>12.1f} {ref_ms/ms:>10.3f}x")
    print("=" * 60)

    if args.profile:
        profile_variant("naive_attention",
                         lambda: fastAttention.naive_attention(Q, K, V))
        profile_variant("fused_attention",
                         lambda: fastAttention.fused_attention(Q, K, V))
        profile_variant("tc_fused_attention",
                         lambda: fastAttention.tc_fused_attention(Q, K, V))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", "-e", type=int, default=64)
    parser.add_argument("--seq_len",   "-s", type=int, default=1024)
    parser.add_argument("--profile",   action="store_true",
                        help="run torch.profiler on each variant")
    args = parser.parse_args()
    main(args)

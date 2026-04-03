/*
 * fastAttention.cpp – PyTorch C++ extension bindings
 *
 * Exposes each attention variant to Python via pybind11.
 * The actual kernels live in fastAttention_kernels.cu.
 */

#include <torch/extension.h>
#include <vector>

/* Forward declarations – defined in fastAttention_kernels.cu */
void naiveAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output);

void fusedAttention(torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V, torch::Tensor output);

void tcFusedAttention(torch::Tensor Q, torch::Tensor K,
                      torch::Tensor V, torch::Tensor output);

void sparseAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                     torch::Tensor output,
                     torch::Tensor row_ptr, torch::Tensor col_idx,
                     int block_h, int block_w);

/* ── thin wrapper that allocates the output tensor and calls the kernel ── */

torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto output = torch::zeros({Q.size(0), Q.size(1)},
                               torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));
    naiveAttention(Q, K, V, output);
    return output;
}

torch::Tensor fused_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto output = torch::zeros({Q.size(0), Q.size(1)},
                               torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));
    fusedAttention(Q, K, V, output);
    return output;
}

torch::Tensor tc_fused_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto output = torch::zeros({Q.size(0), Q.size(1)},
                               torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));
    tcFusedAttention(Q, K, V, output);
    return output;
}

torch::Tensor sparse_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                torch::Tensor row_ptr, torch::Tensor col_idx,
                                int block_h, int block_w) {
    auto output = torch::zeros({Q.size(0), Q.size(1)},
                               torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));
    sparseAttention(Q, K, V, output, row_ptr, col_idx, block_h, block_w);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention",    &naive_attention,    "Week 1 – naive attention (separate kernels)");
    m.def("fused_attention",    &fused_attention,    "Week 2 – fused attention (shared-mem + warp primitives)");
    m.def("tc_fused_attention", &tc_fused_attention, "Week 3 – mixed-precision attention (FP16 GEMM, FP32 softmax)");
    m.def("sparse_attention",   &sparse_attention,   "Week 4 – block-sparse attention (CSR mask)");
}

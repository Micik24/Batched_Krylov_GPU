module BatchedKrylovGPU

using CUDA
using LinearAlgebra

CUDA.allowscalar(false)

include("gemm_wrappers.jl")
include("operators.jl")
include("utils.jl")
include("arnoldi_cgs2.jl")

export
    gemm_strided_batched!,
    apply_diagonal_operator!,
    batched_arnoldi_cgs2_gpu

end

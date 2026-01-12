using CUDA

"""
    gemm_strided_batched!(
        transA, transB,
        α,
        A, lda, strideA,
        B, ldb, strideB,
        β,
        C, ldc, strideC,
        batchCount
    )

Thin wrapper around cuBLAS strided-batched GEMM.

Assumes A, B, C are 3D CuArrays with batch dimension last.
"""
function gemm_strided_batched!(
    transA::Char, transB::Char,
    α::T,
    A::AbstractArray{T,3}, lda::Int, strideA::Int,
    B::AbstractArray{T,3}, ldb::Int, strideB::Int,
    β::T,
    C::AbstractArray{T,3}, ldc::Int, strideC::Int,
    batchCount::Int
) where {T<:Union{Float32,Float64}}

    CUDA.CUBLAS.gemmStridedBatchedEx!(transA, transB, α, A, B, β, C)
    return C
end

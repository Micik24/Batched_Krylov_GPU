using CUDA
using LinearAlgebra

using .BatchedKrylovGPU:
    gemm_strided_batched!,
    apply_diagonal_operator!,
    column_norms

"""
    batched_arnoldi_cgs2_gpu(
        A0, d, B;
        kmax=30,
        tol=1e-14,
        transA0=false,
        α=1
    )

Batched Arnoldi method with CGS2 reorthogonalization on GPU.

- A = A0 + diag(d)
- B contains multiple RHS (columns)
- Masked breakdown handling
"""
function batched_arnoldi_cgs2_gpu(
    A0::CuArray{T,2},
    d::CuArray{T,1},
    B::CuArray{T,2};
    kmax::Int = 30,
    tol::T = T(1e-14),
    transA0::Bool = false,
    α::T = one(T),
) where {T<:Union{Float32,Float64}}

    n, S = size(B)
    k1 = kmax + 1

    V = CUDA.zeros(T, n, k1, S)
    H = CUDA.zeros(T, k1, kmax, S)

    Vj = CUDA.zeros(T, n, S)
    Q  = CUDA.zeros(T, n, S)

    h1 = CUDA.zeros(T, k1, 1, S)
    h2 = CUDA.zeros(T, k1, 1, S)

    β0 = column_norms(B)
    β  = CUDA.zeros(T, S)

    active = β0 .> tol
    mask = reshape(active, 1, :)
    scale = reshape(max.(β0, eps(T)), 1, :)

    Vj .= B ./ scale
    Vj .*= mask

    Q3 = reshape(Q, n, 1, S)

    strideV = n * k1
    strideQ = n
    strideh = k1

    for j in 1:kmax
        @views V[:, j, :] .= Vj
        r = j

        apply_diagonal_operator!(Q, A0, d, Vj; trans=transA0, α=α)
        Q .*= mask

        Vr  = @view V[:, 1:r, :]
        h1r = @view h1[1:r, 1, :]
        h2r = @view h2[1:r, 1, :]

        fill!(h1r, zero(T))

        gemm_strided_batched!(
            'T','N',
            one(T),
            Vr, n, strideV,
            Q3, n, strideQ,
            zero(T),
            @view(h1[1:r,1:1,:]), k1, strideh,
            S
        )

        gemm_strided_batched!(
            'N','N',
            -one(T),
            Vr, n, strideV,
            @view(h1[1:r,1:1,:]), k1, strideh,
            one(T),
            Q3, n, strideQ,
            S
        )

        fill!(h2r, zero(T))

        gemm_strided_batched!(
            'T','N',
            one(T),
            Vr, n, strideV,
            Q3, n, strideQ,
            zero(T),
            @view(h2[1:r,1:1,:]), k1, strideh,
            S
        )

        gemm_strided_batched!(
            'N','N',
            -one(T),
            Vr, n, strideV,
            @view(h2[1:r,1:1,:]), k1, strideh,
            one(T),
            Q3, n, strideQ,
            S
        )

        h1r .+= h2r

        β .= column_norms(Q)

        @views H[1:r, j, :] .= h1r
        @views H[r+1, j, :] .= β

        active .= active .& (β .> tol)
        mask = reshape(active, 1, :)

        scale = reshape(max.(β, eps(T)), 1, :)
        Vj .= Q ./ scale
        Vj .*= mask

        @views V[:, j+1, :] .= Vj
    end

    return V, H, β0, active
end

using CUDA
using LinearAlgebra

"""
    apply_diagonal_operator!(Q, A0, d, V; trans=false, α=1)

Applies:
    Q = α * (A0 + diag(d)) * V
or, if `trans=true`,
    Q = α * (A0' + diag(d)) * V

Designed for Krylov methods without materializing A.
"""
@inline function apply_diagonal_operator!(
    Q::CuArray{T,2},
    A0::CuArray{T,2},
    d::CuArray{T,1},
    V::CuArray{T,2};
    trans::Bool = false,
    α::T = one(T),
) where {T<:Union{Float32,Float64}}

    if !trans
        mul!(Q, A0, V)
    else
        mul!(Q, transpose(A0), V)
    end

    Q .+= reshape(d, :, 1) .* V

    if α != one(T)
        Q .*= α
    end

    return Q
end

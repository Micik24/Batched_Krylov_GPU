using Test
using LinearAlgebra
using BatchedKrylovGPU

@testset "Batched Arnoldi – basic correctness" begin
    
    if !CUDA.functional()
        @info "CUDA not available, skipping GPU tests"
        return
    end

    n = 10
    m = 5
    batch = 3

    A0 = randn(Float32, n, n)
    d  = randn(Float32, n)
    A  = A0 + Diagonal(d)

    X0 = randn(Float32, n, batch)

    Vall, Hall, beta0, active =
        batched_arnoldi(A, X0; m=m)

    @test size(Vall, 1) == n
    @test size(Vall, 2) == m + 1
    @test size(Vall, 3) == batch

    @test size(Hall, 1) == m + 1
    @test size(Hall, 2) == m
    @test size(Hall, 3) == batch

    # Orthogonality test
    for b in 1:batch
        V = Vall[:, :, b]
        G = V' * V
        @test norm(G - I, Inf) < 1e-4
    end

    @test length(active) == batch
end

using CUDA
using BatchedKrylovGPU

n = 512
S = 64

A0 = CUDA.randn(Float64, n, n)
A0 .= A0 .- Diagonal(diag(A0))
d = -CUDA.ones(Float64, n)

B = CUDA.randn(Float64, n, S)

V, H, β, active = batched_arnoldi_cgs2_gpu(A0, d, B; kmax=30)

println("Fraction active: ", mean(Array(active)))

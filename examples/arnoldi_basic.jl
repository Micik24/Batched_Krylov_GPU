using CUDA
using BatchedKrylovGPU

n = 256
S = 16
k = 20

A0 = CUDA.randn(Float64, n, n)
A0 .= A0 .- Diagonal(diag(A0))
d  = -abs.(CUDA.randn(Float64, n))

B = CUDA.randn(Float64, n, S)

V, H, β, active = batched_arnoldi_cgs2_gpu(A0, d, B; kmax=k)

println("Active trajectories: ", sum(Array(active)))

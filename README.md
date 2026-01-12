# BatchedKrylovGPU.jl

GPU-resident batched Arnoldi / Krylov methods implemented in Julia using CUDA.jl.

## Features
- Batched Arnoldi with CGS2 reorthogonalization
- Fully GPU-resident (no CPU fallback)
- Masked breakdown handling
- Custom linear operators of the form A₀ + diag(d)
- Built on strided-batched cuBLAS GEMM

## Motivation
Designed for problems involving many independent Krylov projections,
such as exponential integrators, adjoint methods, and inverse problems.

## Status
Research code. API may change.

# Batched Krylov GPU

**GPU-resident batched Arnoldi / Krylov subspace methods implemented in Julia using CUDA.**

This package is designed for efficient handling of many right-hand sides in parallel, with all core linear algebra operations executed on the GPU. This project focuses on **numerical linear algebra for high-performance computing**, with applications in time integration, model reduction, and large-scale optimization.

---

## Key Features

- **Batched Arnoldi Method:** Implements CGS2 reorthogonalization for numerical stability.
- **Fully GPU-Resident:** All operations stay on the device (no expensive CPU round-trips).
- **High Performance:** Utilizes **Strided-batched cuBLAS GEMM** backend.
- **Operator Support:** Specialized support for operators of the form:
  $$A = A_0 + \mathrm{diag}(d)$$
- **Robust:** Features **active trajectory masking** and graceful breakdown handling.
- **Research Ready:** Designed for research, experimentation, and easy extension.

---

## Installation

You can install the package directly from GitHub:

```julia
using Pkg
Pkg.add(url="[https://github.com/Micik24/Batched_Krylov_GPU.jl](https://github.com/Micik24/Batched_Krylov_GPU.jl)")
```

---

## Quick Start

Here is a simple example demonstrating how to run the batched Arnoldi process.

```julia
using BatchedKrylovGPU
using CUDA

# 1. Define Problem Dimensions
n     = 100      # Matrix size
m     = 20       # Krylov subspace dimension
batch = 8        # Number of right-hand sides

# 2. Setup Operator A = A0 + diag(d)
A0 = CUDA.randn(Float32, n, n)
d  = CUDA.randn(Float32, n)

# 3. Initialize Vectors
X0 = CUDA.randn(Float32, n, batch)

# 4. Run Batched Arnoldi
Vall, Hall, beta0, active = batched_arnoldi(A0, d, X0; m = m)
```

### Outputs

- **`Vall`**: Arnoldi basis vectors.
  *Shape:* `(n, m+1, batch)`
- **`Hall`**: Upper Hessenberg matrices.
  *Shape:* `(m+1, m, batch)`
- **`beta0`**: Initial normalization coefficients for each trajectory.
- **`active`**: Boolean mask indicating which trajectories remained active throughout the iteration.

---

## Requirements

- **Julia:** ≥ 1.9
- **Packages:** `CUDA.jl`
- **Hardware:** NVIDIA GPU with CUDA support

> **Note:** Continuous Integration (CI) runs CPU-only correctness tests. Full functionality and performance benefits require a CUDA-capable GPU.

---

## Testing

Basic correctness tests are included and automatically run via GitHub Actions. The test suite verifies:

- API stability
- Output dimensions
- Orthogonality of the Arnoldi basis
- Correct handling of batched trajectories

---

## Project Status

**This is research code.** The API may change, and performance tuning is ongoing.

Contributions, discussions, and experimentation are welcome!

---

## License

MIT License
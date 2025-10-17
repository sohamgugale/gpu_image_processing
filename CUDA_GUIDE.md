# CUDA Implementation Guide
- GPU operations use CuPy (NumPy-compatible).
- Can later replace `_convolve()` with custom CUDA kernels via RawKernel or Numba.cuda.jit.

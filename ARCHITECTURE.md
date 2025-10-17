# Architecture Overview
- Modular design with separate CPU/GPU backends.
- Each image operation (blur, edge detection) uses `_convolve()` kernel internally.

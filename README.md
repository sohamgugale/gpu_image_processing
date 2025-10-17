# GPU Image Processing 🚀

A CUDA-accelerated image processing pipeline using **CuPy** for GPU computation.

## 📦 Features
- Grayscale conversion
- Gaussian-like blur filter
- Edge detection (Sobel operator)
- CPU (NumPy) ↔ GPU (CuPy) switch
- Performance benchmarking

## 🧠 Requirements
```bash
pip install -r requirements.txt
```

## 🖥️ Run
```bash
python main.py
```

## ⚡ Benchmark
```bash
python benchmark.py
```

## 📂 Folder Structure
```
gpu-image-processing/
├── main.py
├── image_processing_pipeline.py
├── check_cuda.py
├── benchmark.py
├── requirements.txt
├── README.md
├── LICENSE
├── images/
│   ├── original.jpg
│   ├── blur.jpg
│   └── edge.jpg
└── tests/
```

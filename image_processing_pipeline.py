import numpy as np
import cv2

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


class ImageProcessingPipeline:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        print(f"Pipeline using {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'} backend.")

    def to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def blur(self, img, kernel_size=5):
        kernel = self.xp.ones((kernel_size, kernel_size), dtype=self.xp.float32)
        kernel /= kernel.size
        return self._convolve(img, kernel)

    def edge_detect(self, img):
        Kx = self.xp.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=self.xp.float32)
        Ky = self.xp.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=self.xp.float32)

        Gx = self._convolve(img, Kx)
        Gy = self._convolve(img, Ky)

        mag = self.xp.sqrt(Gx**2 + Gy**2)
        mag = (mag / mag.max() * 255).astype(np.uint8)
        return cp.asnumpy(mag) if self.use_gpu else mag

    def _convolve(self, img, kernel):
        xp = self.xp
        pad = kernel.shape[0] // 2
        img_padded = xp.pad(img, pad, mode='reflect')

        H, W = img.shape
        result = xp.zeros_like(img, dtype=xp.float32)

        for i in range(H):
            for j in range(W):
                region = img_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                result[i, j] = xp.sum(region * kernel)

        if self.use_gpu:
            result = cp.asnumpy(result)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

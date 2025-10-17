import cv2
from image_processing_pipeline import ImageProcessingPipeline

def test_grayscale():
    img = cv2.imread("images/original.jpg")
    pipeline = ImageProcessingPipeline(use_gpu=False)
    gray = pipeline.to_grayscale(img)
    assert len(gray.shape) == 2

def test_blur():
    import numpy as np
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    pipeline = ImageProcessingPipeline(use_gpu=False)
    blurred = pipeline.blur(img)
    assert blurred.shape == img.shape

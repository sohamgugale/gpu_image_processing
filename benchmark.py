import cv2
import time
from image_processing_pipeline import ImageProcessingPipeline

def benchmark():
    img = cv2.imread("images/original.jpg", cv2.IMREAD_GRAYSCALE)
    print("Running benchmarks...\n")

    for gpu in [False, True]:
        start = time.time()
        pipeline = ImageProcessingPipeline(use_gpu=gpu)
        blur = pipeline.blur(img)
        edges = pipeline.edge_detect(img)
        end = time.time()
        print(f"{'GPU' if gpu else 'CPU'} time: {end - start:.3f} s")

if __name__ == "__main__":
    benchmark()

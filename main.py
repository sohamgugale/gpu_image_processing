import cv2
from image_processing_pipeline import ImageProcessingPipeline
from check_cuda import is_cuda_available

def main():
    print("🚀 GPU Image Processing Pipeline\n")

    cuda_status = is_cuda_available()
    print(f"CUDA Available: {cuda_status}\n")

    # Initialize pipeline
    pipeline = ImageProcessingPipeline(use_gpu=cuda_status)

    # Input image
    img_path = "images/original.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found. Please place an image at 'images/original.jpg'.")
        return

    # Process
    gray = pipeline.to_grayscale(img)
    blur = pipeline.blur(gray)
    edges = pipeline.edge_detect(gray)

    # Save outputs
    cv2.imwrite("images/blur.jpg", blur)
    cv2.imwrite("images/edge.jpg", edges)

    print("✅ Processing complete! Results saved in 'images/'.")

if __name__ == "__main__":
    main()

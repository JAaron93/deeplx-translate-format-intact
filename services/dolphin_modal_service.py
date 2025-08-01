"""Modal Labs deployment service for ByteDance Dolphin OCR.

This service replaces the HuggingFace Spaces API approach with a direct
Modal deployment of the Dolphin OCR model for better performance and control.
"""

from typing import Any, Dict, List

import modal

# Modal App Configuration
app = modal.App("dolphin-ocr-service")

# Model and cache configuration
MODEL_CACHE_PATH = "/models"
DOLPHIN_MODEL_ID = "ByteDance/Dolphin"  # Adjust based on actual model ID
PDF2IMAGE_DPI = 300

# Define the container image with all dependencies
dolphin_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "poppler-utils",  # For pdf2image
        "libgl1-mesa-glx",  # For OpenCV
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "wget",
        "curl",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.30.0",
        "accelerate",
        "pdf2image>=3.1.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "opencv-python-headless",
        "huggingface_hub>=0.16.0",
        "hf-transfer>=0.1.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "TRANSFORMERS_CACHE": MODEL_CACHE_PATH,
            "TORCH_HOME": MODEL_CACHE_PATH,
        }
    )
)

# Create a persistent volume for model storage
model_volume = modal.Volume.from_name("dolphin-ocr-models", create_if_missing=True)


@app.function(
    image=dolphin_image,
    gpu="T4",  # Start with T4, can upgrade to A10G or L40S if needed
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=300,
    memory=8192,  # 8GB RAM
    container_idle_timeout=300,  # 5 minutes
)
def download_model():
    """Download and cache the Dolphin OCR model."""
    from huggingface_hub import snapshot_download

    print("Downloading Dolphin OCR model...")

    # Download the model to the persistent volume
    model_path = snapshot_download(
        repo_id=DOLPHIN_MODEL_ID,
        cache_dir=MODEL_CACHE_PATH,
        local_dir=f"{MODEL_CACHE_PATH}/dolphin",
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to: {model_path}")
    return model_path


@app.function(
    image=dolphin_image,
    gpu="T4",  # Adjust based on performance needs
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=600,  # 10 minutes for processing
    memory=8192,
    container_idle_timeout=600,
)
def process_pdf_with_dolphin(pdf_bytes: bytes) -> Dict[str, Any]:
    """Process a PDF with Dolphin OCR and return structured layout data.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Dictionary containing OCR results with layout information
    """
    import torch
    from pdf2image import convert_from_bytes
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("Loading Dolphin OCR model...")

    # Load the model and processor
    model_path = f"{MODEL_CACHE_PATH}/dolphin"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Converting PDF to images...")

    # Convert PDF to images
    images = convert_from_bytes(
        pdf_bytes,
        dpi=PDF2IMAGE_DPI,
        fmt="RGB",
        thread_count=2,
    )

    print(f"Processing {len(images)} pages...")

    results = []

    for page_num, image in enumerate(images):
        print(f"Processing page {page_num + 1}/{len(images)}")

        # Process the image with Dolphin
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,
            )

        # Decode the output
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse the Dolphin output (this will need to be adapted based on actual format)
        page_result = {
            "page_number": page_num + 1,
            "width": image.width,
            "height": image.height,
            "text_blocks": parse_dolphin_output(generated_text),
            "raw_output": generated_text,
        }

        results.append(page_result)

    return {
        "pages": results,
        "total_pages": len(results),
        "processing_metadata": {
            "processor": "dolphin_ocr_modal",
            "model_id": DOLPHIN_MODEL_ID,
            "dpi": PDF2IMAGE_DPI,
        },
    }


def parse_dolphin_output(raw_output: str) -> List[Dict[str, Any]]:
    """Parse Dolphin OCR output into structured text blocks.

    This function will need to be implemented based on the actual
    output format of the Dolphin model.
    """
    # TODO: Implement actual parsing logic based on Dolphin's output format
    # For now, return a placeholder structure
    return [
        {
            "text": raw_output,
            "bbox": [0, 0, 100, 100],  # [x1, y1, x2, y2]
            "confidence": 0.95,
            "block_type": "text",
        }
    ]


@app.function(
    image=dolphin_image,
    timeout=30,
)
@modal.web_endpoint(method="POST", docs=True)
def dolphin_ocr_endpoint(
    pdf_file: bytes = modal.web_endpoint.FileUpload(),
) -> Dict[str, Any]:
    """HTTP endpoint for Dolphin OCR processing.
    Compatible with the existing dolphin_client.py interface.
    """
    try:
        # Process the PDF
        result = process_pdf_with_dolphin.remote(pdf_file)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
        }


@app.function()
def setup_dolphin_service():
    """Initialize the Dolphin service by downloading the model."""
    print("Setting up Dolphin OCR service...")
    model_path = download_model.remote()
    print(f"Dolphin service ready! Model at: {model_path}")
    return model_path


if __name__ == "__main__":
    # For local testing
    with app.run():
        setup_dolphin_service.remote()

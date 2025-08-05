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
    # Create an instance of the class and call the method
    processor = DolphinOCRProcessor()
    return processor.process_pdf(pdf_bytes)Download and cache the Dolphin OCR model."""
    from huggingface_hub import snapshot_download

    print("Downloading Dolphin OCR model...")

    # Download the model to the persistent volume
    try:
        model_path = snapshot_download(
            repo_id=DOLPHIN_MODEL_ID,
            cache_dir=MODEL_CACHE_PATH,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise

    print(f"Model downloaded to: {model_path}")
    return model_path


@app.cls(
    image=dolphin_image,
    gpu="T4",  # Adjust based on performance needs
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=600,  # 10 minutes for processing
    memory=8192,
    container_idle_timeout=600,
)
class DolphinOCRProcessor:
    """Cached Dolphin OCR processor that loads model once and reuses it."""

    def __init__(self):
        """Initialize the processor and load the model once."""
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print("Initializing Dolphin OCR processor...")
        print("Loading Dolphin OCR model and processor (one-time initialization)...")

        # Load the model and processor once during initialization
        # Use the path returned by snapshot_download or a consistent location
        # Use the actual model ID for loading
        self.processor = AutoProcessor.from_pretrained(DOLPHIN_MODEL_ID, cache_dir=MODEL_CACHE_PATH)
        self.model = AutoModelForVision2Seq.from_pretrained(
            DOLPHIN_MODEL_ID,
            cache_dir=MODEL_CACHE_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("✅ Dolphin OCR model and processor loaded successfully!")
        print(f"Model device: {self.model.device}")

    @modal.method()
    def process_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Process a PDF with Dolphin OCR using cached model and processor.

        Args:
            pdf_bytes: Raw PDF file bytes

        Returns:
            Dictionary containing OCR results with layout information
        """
        import torch
        from pdf2image import convert_from_bytes

        print("Processing PDF with cached Dolphin OCR model...")

        # Convert PDF to images
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=PDF2IMAGE_DPI,
                fmt="RGB",
                thread_count=2,
            )
        except Exception as e:
            print(f"Failed to convert PDF: {e}")
            raise ValueError(f"PDF conversion failed: {str(e)}")

        print(f"Processing {len(images)} pages...")

        results = []

        for page_num, image in enumerate(images):
            print(f"Processing page {page_num + 1}/{len(images)}")

            # Process the image with Dolphin using cached model and processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                )

            # Decode the output
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse the Dolphin output (this will need to be adapted based on actual format)
            page_result = {
                "page_number": page_num + 1,
                "width": image.width,
                "height": image.height,
                "text_blocks": [{"text": generated_text, "bbox": None, "confidence": 1.0, "block_type": "raw"}],  # TODO: Implement proper parsing
                "raw_output": generated_text,
            }

            results.append(page_result)

        return {
            "pages": results,
            "total_pages": len(results),
            "processing_metadata": {
                "processor": "dolphin_ocr_modal_cached",
                "model_id": DOLPHIN_MODEL_ID,
                "dpi": PDF2IMAGE_DPI,
            },
        }


# Create a global instance for backward compatibility
dolphin_processor = DolphinOCRProcessor()


@app.function(
    image=dolphin_image,
    gpu="T4",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=600,
    memory=8192,
    container_idle_timeout=600,
)
def process_pdf_with_dolphin(pdf_bytes: bytes) -> Dict[str, Any]:
    """Process a PDF with Dolphin OCR using cached model (backward compatibility wrapper).

    This function maintains backward compatibility while using the cached processor.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Dictionary containing OCR results with layout information
    """
    return dolphin_processor.process_pdf.remote(pdf_bytes)


def parse_dolphin_output(raw_output: str) -> List[Dict[str, Any]]:
    """Parse Dolphin OCR output into structured text blocks.

    This function needs to be implemented based on the actual output format
    of the Dolphin model. The expected format should include structured text
    blocks with bounding boxes, confidence scores, and element types.

    Args:
        raw_output: Raw text output from Dolphin OCR model

    Returns:
        List of structured text blocks with layout information

    Raises:
        NotImplementedError: This function is not yet implemented and requires
            knowledge of the actual Dolphin OCR output format.

    Note:
        Once the Dolphin model output format is known, this function should
        parse the raw_output into a list of dictionaries with the following
        structure:
        [
            {
                "text": str,           # Extracted text content
                "bbox": [x1, y1, x2, y2],  # Bounding box coordinates
                "confidence": float,   # OCR confidence score (0.0-1.0)
                "block_type": str,     # Element type (text, title, table, etc.)
    # Temporary implementation - returns raw output as a single text block
    # TODO: Implement proper parsing once Dolphin output format is known
    import logging
    logging.warning("Using temporary parse_dolphin_output implementation")
    return [{
        "text": raw_output,
        "bbox": [0, 0, 0, 0],  # Placeholder coordinates
        "confidence": 1.0,
        "block_type": "raw_text"
    }]
        "This function requires knowledge of the actual Dolphin OCR output format. "
        f"Raw output received: {raw_output[:100]}{'...' if len(raw_output) > 100 else ''}"
    )


@app.function(
    image=dolphin_image,
    timeout=630,  # Slightly longer than processing function timeout
)
@modal.web_endpoint(method="POST", docs=True)
def dolphin_ocr_endpoint(
    pdf_file: bytes = modal.web_endpoint.FileUpload(),
) -> Dict[str, Any]:
    """HTTP endpoint for Dolphin OCR processing.
    Compatible with the existing dolphin_client.py interface.
    """
    # Validate file size (example: 50MB limit)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    if len(pdf_file) > MAX_FILE_SIZE:
        return {
            "error": "File too large. Maximum size is 50MB.",
            "status": "failed",
        }

    try:
        # Process the PDF using cached processor for better performance
        # Properly invoke the Modal function
        result = process_pdf_with_dolphin.remote(pdf_file)
        return result
    except ValueError as e:
        return {
            "error": f"Invalid input: {str(e)}",
            "status": "failed",
        }
    except Exception as e:
        print(f"Unexpected error in OCR endpoint: {e}")
        return {
            "error": "Internal server error during OCR processing",
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

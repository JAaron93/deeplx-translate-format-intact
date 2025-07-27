"""Professional Document Translator with Advanced Formatting Preservation
Refactored main application entry point
"""

import logging
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import api_router, app_router

# Import refactored components
from ui.gradio_interface import create_gradio_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Advanced Document Translator",
    description=(
        "Professional document translation with advanced formatting " "preservation"
    ),
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(app_router)  # Root and philosophy routes without prefix
app.include_router(api_router, prefix="/api/v1")  # API routes with versioning

# Static files mount
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Create required directories on startup"""
    required_directories = [
        "static",
        "uploads",
        "downloads",
        ".layout_backups",
        "templates",
    ]

    for directory in required_directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    logger.info("All required directories created successfully")


def main() -> None:
    """Main application entry point"""
    logger.info("Starting Advanced Document Translator")

    # Create Gradio interface
    gradio_app = create_gradio_interface()

    # Mount Gradio app to FastAPI
    app_with_gradio = gr.mount_gradio_app(app, gradio_app, path="/ui")

    # Start server with Uvicorn
    # Note: Default to localhost for security, use environment variable to override
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app_with_gradio, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

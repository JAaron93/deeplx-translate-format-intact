"""
Professional Document Translator with Advanced Formatting Preservation
Based on amazon-translate-pdf approach with comprehensive layout preservation
"""

from __future__ import annotations

import os
import asyncio
import logging
import tempfile
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import mimetypes

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import our enhanced translation modules
from services.translation_service import TranslationService
from services.enhanced_document_processor import EnhancedDocumentProcessor
from services.language_detector import LanguageDetector
from utils.file_handler import FileHandler
from utils.validators import FileValidator
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize services with enhanced processing
settings = Settings()
translation_service = TranslationService()
document_processor = EnhancedDocumentProcessor(
    dpi=settings.get('PDF_DPI', 300),
    preserve_images=settings.get('PRESERVE_IMAGES', True)
)
language_detector = LanguageDetector()
file_handler = FileHandler()
file_validator = FileValidator()

# FastAPI app
app = FastAPI(
    title="Advanced Document Translator",
    description="Professional document translation with advanced formatting preservation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("downloads", exist_ok=True)
os.makedirs(".layout_backups", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state for translation jobs
translation_jobs: Dict[str, Dict[str, Any]] = {}

class AdvancedTranslationState:
    """Enhanced translation state management with comprehensive processing info"""
    
    def __init__(self):
        self.current_file = None
        self.current_content = None
        self.source_language = None
        self.target_language = None
        self.translation_progress = 0
        self.translation_status = "idle"
        self.error_message = ""
        self.job_id = None
        self.output_file = None
        self.processing_info = {}
        self.backup_path = None
        self.max_pages: int = 0  # 0 means translate all pages

# Global state instance
state = AdvancedTranslationState()

async def process_file_upload(file) -> Tuple[str, str, str, str]:
    """Process uploaded file with advanced content extraction"""
    try:
        if file is None:
            return "", "No file uploaded", "", ""
        
        # Handle Gradio file object (which can be a string path or file-like object)
        if isinstance(file, str):
            # File is a path string (modern Gradio)
            file_path = file
            file_name = os.path.basename(file_path)
            file_path = file
            file_name = os.path.basename(file_path)
            try:
                file_size = os.path.getsize(file_path)
            except OSError as e:
                logger.error(f"Failed to get file size: {e}")
                return "", f"❌ Failed to process file: {str(e)}", "", ""
            # File is a file-like object (older Gradio)
            file_name = getattr(file, 'name', 'unknown')
            
            # Get file size safely
            if hasattr(file, 'read'):
                current_pos = file.tell() if hasattr(file, 'tell') else 0
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(current_pos)  # Reset to original position
            else:
                file_size = 0
            
            # Save uploaded file
            file_path = file_handler.save_uploaded_file(file)
        
        # Validate file
        validation_result = file_validator.validate_file(file_name, file_size)
        if not validation_result["valid"]:
            return "", f"❌ {validation_result['error']}", "", ""
        
        state.current_file = file_path
        
        logger.info(f"Starting advanced processing of: {file_name}")
        
        # Extract content with advanced processing
        content = document_processor.extract_content(file_path)
        state.current_content = content
        
        # Generate preview
        preview = content.get('preview', 'Preview not available')
        
        # Detect language based on content type
        if content['type'] == 'pdf_advanced':
            # Use first page text for language detection
            first_page_texts = content['text_by_page'].get(0, [])
            sample_text = ' '.join(first_page_texts[:5])  # First 5 text elements
        else:
            sample_text = content.get('text_content', '')[:1000]
        
        detected_lang = language_detector.detect_language_from_text(sample_text)
        state.source_language = detected_lang
        
        # Store processing info
        metadata = content.get('metadata')
        if metadata:
            state.processing_info = {
                'file_type': metadata.file_type,
                'total_pages': metadata.total_pages,
                'total_elements': metadata.total_text_elements,
                'file_size_mb': metadata.file_size_mb,
                'processing_time': metadata.processing_time,
                'dpi': getattr(metadata, 'dpi', 'N/A')
            }
        
        # Create status message with detailed info
        status_parts = [
            f"✅ File processed successfully",
            f"📄 Type: {Path(file_name).suffix.upper()}",
            f"🌐 Detected language: {detected_lang}"
        ]
        
        if metadata:
            status_parts.extend([
                f"📊 Pages: {metadata.total_pages}",
                f"📝 Text elements: {metadata.total_text_elements}",
                f"💾 Size: {metadata.file_size_mb:.2f} MB",
                f"⏱️ Processing: {metadata.processing_time:.2f}s"
            ])
            
            if hasattr(metadata, 'dpi'):
                status_parts.append(f"🖼️ Resolution: {metadata.dpi} DPI")
        
        status = "\n".join(status_parts)
        
        # Processing details for display
        processing_details = json.dumps(state.processing_info, indent=2)
        
        logger.info(f"Advanced processing complete: {file_name}")
        return preview, status, detected_lang, processing_details
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return "", f"❌ Upload failed: {str(e)}", "", ""

async def start_translation(target_language: str, max_pages: int) -> Tuple[str, str, bool]:
    """Start the advanced translation process"""
    try:
        if not state.current_file or not state.current_content:
            return "❌ No file processed", "", False
        
        if not target_language:
            return "❌ Please select target language", "", False
        
        state.target_language = target_language
        state.job_id = str(uuid.uuid4())
        state.translation_status = "starting"
        state.translation_progress = 0
        state.error_message = ""
        state.max_pages = max_pages
        
        # Start translation in background
        asyncio.create_task(perform_advanced_translation())
        
        return "🚀 Advanced translation started...", "Translation in progress with format preservation", False
        
    except Exception as e:
        logger.error(f"Translation start error: {str(e)}")
        return f"❌ Failed to start translation: {str(e)}", "", False

async def perform_advanced_translation() -> None:
    """Perform the advanced translation process with format preservation"""
    try:
        state.translation_status = "processing"
        logger.info(f"Starting advanced translation: {state.source_language} -> {state.target_language}")
        
        content = state.current_content
        
        # Extract text for translation based on content type
        state.translation_progress = 10
        if content['type'] == 'pdf_advanced':
            text_by_page = content['text_by_page']
        elif content['type'] in ['docx', 'txt']:
            # Convert to page-based format for consistency
            if content['type'] == 'docx':
                texts = [para['text'] for para in content['paragraphs'] if para['text'].strip()]
            else:
                texts = content['lines']
            text_by_page = {0: texts}
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")
        
        # Translate content page by page respecting max_pages limit
        state.translation_progress = 30
        translated_by_page = {}

        max_pages = state.max_pages if state.max_pages > 0 else None
        pages_to_process = {k: v for k, v in text_by_page.items() if (max_pages is None or k < max_pages)}

        total_pages = len(pages_to_process)
        for page_num, page_texts in pages_to_process.items():
            if not page_texts:
                translated_by_page[page_num] = []
                continue
            
            logger.info(
                f"Translating page {page_num + 1}/{total_pages} (" f"{len(page_texts)} elements)"
            )

            # Batch translate text elements for efficiency
            batch_size = 20
            translated_texts: List[str] = [None] * len(page_texts)

            # Identify non-empty text indices to translate
            indices_to_translate = [idx for idx, txt in enumerate(page_texts) if txt.strip()]

            for start in range(0, len(indices_to_translate), batch_size):
                batch_indices = indices_to_translate[start : start + batch_size]
                batch_texts = [page_texts[idx] for idx in batch_indices]

                try:
                    batch_translated = await translation_service.translate_batch(
                        batch_texts,
                        state.source_language,
                        state.target_language,
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch translation failed for page {page_num} (indices {batch_indices}): {e}"
                    )
                    batch_translated = batch_texts  # Fallback to original

                # Map translated texts back to their original positions
                for idx, translated in zip(batch_indices, batch_translated):
                    translated_texts[idx] = translated

            # Fill untranslated (empty) slots with the original text
            for idx, original in enumerate(page_texts):
                if translated_texts[idx] is None:
                    translated_texts[idx] = original

            translated_by_page[page_num] = translated_texts
            
            # Update progress
            page_progress = 30 + (page_num + 1) / total_pages * 50
            state.translation_progress = int(page_progress)
        
        # Generate output file with preserved formatting
        state.translation_progress = 85
        output_filename = generate_output_filename(
            Path(state.current_file).name,
            state.target_language
        )
        
        logger.info(f"Creating translated document: {output_filename}")
        state.output_file = document_processor.create_translated_document(
            content,
            translated_by_page,
            output_filename
        )
        
        state.translation_progress = 100
        state.translation_status = "completed"
        
        logger.info(f"Advanced translation completed: {state.output_file}")
        
    except Exception as e:
        logger.error(f"Advanced translation error: {str(e)}")
        state.translation_status = "error"
        state.error_message = str(e)

def update_translation_progress(progress: int) -> None:
    """Update translation progress"""
    state.translation_progress = min(30 + (progress * 0.6), 90)

def generate_output_filename(original_filename: str, target_language: str) -> str:
    """Generate output filename with proper format"""
    name = Path(original_filename).stem
    ext = Path(original_filename).suffix
    return f"translated_{name}_{target_language.lower()}_advanced{ext}"

def get_translation_status() -> Tuple[str, int, bool]:
    """Get current translation status with detailed info"""
    if state.translation_status == "idle":
        return "Ready for advanced translation", 0, False
    elif state.translation_status == "starting":
        return "🚀 Initializing advanced translation...", 5, False
    elif state.translation_status == "processing":
        return f"🔄 Advanced translation in progress... ({state.translation_progress}%)", state.translation_progress, False
    elif state.translation_status == "completed":
        return "✅ Advanced translation completed with format preservation!", 100, True
    elif state.translation_status == "error":
        return f"❌ Translation failed: {state.error_message}", 0, False
    else:
        return "Unknown status", 0, False

def download_translated_file(output_format: str) -> str:
    """Prepare file for download with format conversion if needed"""
    try:
        if not state.output_file or state.translation_status != "completed":
            return "❌ No translated file available"
        
        # Convert to requested format if needed
        if output_format != Path(state.output_file).suffix[1:].upper():
            converted_file = document_processor.convert_format(
                state.output_file,
                output_format.lower()
            )
            return converted_file
        
        return state.output_file
        
    except Exception as e:
        logger.error(f"Download preparation error: {str(e)}")
        return f"❌ Download failed: {str(e)}"

# Gradio Interface with Enhanced Features
def create_gradio_interface() -> gr.Blocks:
    """Create the advanced Gradio web interface"""
    
    with gr.Blocks(
        title="Advanced Document Translator",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container {
                max-width: 1400px !important;
                margin: auto !important;
            }
            .upload-area {
                border: 2px dashed #007bff;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f8f9fa;
            }
            .status-success {
                color: #28a745;
                font-weight: bold;
            }
            .status-error {
                color: #dc3545;
                font-weight: bold;
            }
            .progress-bar {
                background: linear-gradient(90deg, #007bff, #28a745);
            }
            .info-panel {
                background-color: #e9ecef;
                border-radius: 8px;
                padding: 15px;
                font-family: monospace;
                font-size: 12px;
            }
            """
    ) as interface:
        
        gr.Markdown(
            """
            # 📄 Advanced Document Translator
            
            Professional document translation with **comprehensive formatting preservation**.
            
            🎯 **Features:**
            - Advanced PDF processing with image-text overlay technique
            - Precise text positioning preservation
            - High-resolution rendering (300 DPI)
            - Support for complex layouts and embedded images
            - DOCX and TXT format support
            - Automatic language detection
            
            📊 **Supported Files:** PDF, DOCX, TXT (up to 10MB)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # File Upload Section
                gr.Markdown("## 📤 Upload Document")
                
                file_upload = gr.File(
                    label="Choose File (PDF, DOCX, TXT)",
                    file_types=[".pdf", ".docx", ".txt"],
                    file_count="single",
                    elem_classes=["upload-area"]
                )
                
                upload_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=8,
                    max_lines=10
                )
                
                # Advanced Processing Info
                gr.Markdown("## 🔍 Processing Details")
                processing_info = gr.Textbox(
                    label="Processing Information",
                    interactive=False,
                    lines=8,
                    elem_classes=["info-panel"]
                )
                
            with gr.Column(scale=2):
                # Preview Section
                gr.Markdown("## 👀 Document Preview")
                
                document_preview = gr.Textbox(
                    label="Content Preview",
                    lines=12,
                    interactive=False,
                    placeholder="Upload a document to see preview with advanced processing info..."
                )
                
                # Language and Translation Section  
                with gr.Row():
                    with gr.Column():
                        detected_language = gr.Textbox(
                            label="Detected Source Language",
                            interactive=False
                        )
                    
                    with gr.Column():
                        target_language = gr.Dropdown(
                            label="Target Language",
                            choices=[
                                "English", "Spanish", "French", "German", "Italian",
                                "Portuguese", "Russian", "Chinese", "Japanese", "Korean",
                                "Arabic", "Hindi", "Dutch", "Swedish", "Norwegian",
                                "Danish", "Finnish", "Polish", "Czech", "Hungarian"
                            ],
                            value="English"
                        )
                
                # Page limit slider
                pages_slider = gr.Slider(minimum=1, maximum=200, step=1, value=20, label="Pages to translate")
                # Translation Controls
                translate_btn = gr.Button(
                    "🚀 Start Advanced Translation",
                    variant="primary",
                    size="lg"
                )
                
                # Progress Section
                gr.Markdown("## 📊 Translation Progress")
                
                with gr.Row():
                    progress_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=4
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                
                progress_bar = gr.Progress()
                
                # Export Section
                gr.Markdown("## 💾 Download Translated Document")
                
                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output Format",
                        choices=["PDF", "DOCX", "TXT"],
                        value="PDF"
                    )
                    
                    download_btn = gr.Button(
                        "📥 Download",
                        variant="secondary",
                        interactive=False
                    )
                
                download_file = gr.File(
                    label="Download File",
                    visible=False
                )
        
        # Event Handlers
        file_upload.change(
            fn=process_file_upload,
            inputs=[file_upload],
            outputs=[document_preview, upload_status, detected_language, processing_info]
        )
        
        translate_btn.click(
            fn=start_translation,
            inputs=[target_language, pages_slider],
            outputs=[progress_status, upload_status, download_btn]
        )
        
        # Status update function for manual refresh
        def update_status():
            status, progress, download_ready = get_translation_status()
            return status, gr.update(interactive=download_ready)
        
        # Connect refresh button to status update
        # Manual refresh
        refresh_btn.click(
            fn=update_status,
            outputs=[progress_status, download_btn]
        )

        # Automatic periodic status updates every 2 seconds
        interface.load(
            fn=update_status,
            outputs=[progress_status, download_btn],
            every=2
        )
        
        download_btn.click(
            fn=download_translated_file,
            inputs=[output_format],
            outputs=[download_file]
        )
    
    return interface

# FastAPI Routes (Enhanced)
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "Advanced Document Translator API", 
        "version": "2.0.0",
        "features": [
            "Advanced PDF processing",
            "Image-text overlay preservation",
            "High-resolution rendering",
            "Comprehensive format support"
        ]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Enhanced upload endpoint with advanced processing"""
    try:
        # Validate file
        validation_result = file_validator.validate_upload_file(file)
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Save file
        file_path = file_handler.save_upload_file(file)
        
        # Process with advanced extraction
        content = document_processor.extract_content(file_path)
        
        # Detect language
        if content['type'] == 'pdf_advanced':
            first_page_texts = content['text_by_page'].get(0, [])
            sample_text = ' '.join(first_page_texts[:5])
        else:
            sample_text = content.get('text_content', '')[:1000]
            
        detected_lang = language_detector.detect_language_from_text(sample_text)
        
        return {
            "message": "File processed with advanced extraction",
            "filename": file.filename,
            "detected_language": detected_lang,
            "file_path": file_path,
            "content_type": content['type'],
            "metadata": content.get('metadata').__dict__ if content.get('metadata') else None
        }
        
    except Exception as e:
        logger.error(f"Enhanced upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_document(
    background_tasks: BackgroundTasks,
    file_path: str,
    source_language: str,
    target_language: str
):
    """Enhanced translation endpoint"""
    try:
        job_id = str(uuid.uuid4())
        
        # Create job entry with enhanced info
        translation_jobs[job_id] = {
            "status": "started",
            "progress": 0,
            "file_path": file_path,
            "source_language": source_language,
            "target_language": target_language,
            "created_at": datetime.now(),
            "output_file": None,
            "error": None,
            "processing_type": "advanced",
            "format_preservation": True
        }
        
        # Start background translation with advanced processing
        background_tasks.add_task(
            process_advanced_translation_job,
            job_id,
            file_path,
            source_language,
            target_language
        )
        
        return {"job_id": job_id, "status": "started", "type": "advanced"}
        
    except Exception as e:
        logger.error(f"Enhanced translation start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get enhanced job status"""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return translation_jobs[job_id]

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download translated file with enhanced metadata"""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = translation_jobs[job_id]
    if job["status"] != "completed" or not job["output_file"]:
        raise HTTPException(status_code=400, detail="Translation not completed")
    
    return FileResponse(
        job["output_file"],
        media_type="application/octet-stream",
        filename=Path(job["output_file"]).name,
        headers={"X-Processing-Type": "advanced", "X-Format-Preserved": "true"}
    )

async def process_advanced_translation_job(job_id: str, file_path: str, source_lang: str, target_lang: str) -> None:
    """Process translation job with advanced formatting preservation"""
    try:
        job = translation_jobs[job_id]
        job["status"] = "processing"
        
        # Extract content with advanced processing
        job["progress"] = 20
        content = document_processor.extract_content(file_path)
        
        # Extract text for translation
        job["progress"] = 40
        if content['type'] == 'pdf_advanced':
            text_by_page = content['text_by_page']
        elif content['type'] in ['docx', 'txt']:
            if content['type'] == 'docx':
                texts = [para['text'] for para in content['paragraphs'] if para['text'].strip()]
            else:
                texts = content['lines']
            text_by_page = {0: texts}
        
        # Translate with progress tracking
        job["progress"] = 60
        translated_by_page = {}
        for page_num, page_texts in text_by_page.items():
            translated_texts = []
            for text in page_texts:
                if text.strip():
                    translated_text = await translation_service.translate_text(
                        text, source_lang, target_lang
                    )
                    translated_texts.append(translated_text)
                else:
                    translated_texts.append(text)
            translated_by_page[page_num] = translated_texts
        
        # Create output with preserved formatting
        job["progress"] = 80
        output_filename = generate_output_filename(
            Path(file_path).name, target_lang
        )
        
        output_file = document_processor.create_translated_document(
            content, translated_by_page, output_filename
        )
        
        job["progress"] = 100
        job["status"] = "completed"
        job["output_file"] = output_file
        
        logger.info(f"Advanced translation job completed: {job_id}")
    except Exception as e:
        logger.error(f"Error in advanced translation job {job_id}: {str(e)}")
        job["status"] = "failed"
        job["error"] = str(e)
        
def main() -> None:
    """Main application entry point"""
    logger.info("Starting Advanced Document Translator")
    
    # Create Gradio interface
    gradio_app = create_gradio_interface()
    
    # Mount Gradio app to FastAPI
    app_with_gradio = gr.mount_gradio_app(app, gradio_app, path="/")
    
    # Start server with Uvicorn
    uvicorn.run(
        app_with_gradio,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        quiet=False
    )

if __name__ == "__main__":
    main()
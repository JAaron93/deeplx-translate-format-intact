"""Translation handling business logic."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

from config.settings import Settings
from core.state_manager import state, translation_jobs
from services.enhanced_document_processor import EnhancedDocumentProcessor
from services.language_detector import LanguageDetector
from services.neologism_detector import NeologismDetector
from services.philosophy_enhanced_translation_service import (
    PhilosophyEnhancedTranslationService,
)
from services.translation_service import TranslationService
from services.user_choice_manager import UserChoiceManager
from utils.file_handler import FileHandler
from utils.language_utils import extract_text_sample_for_language_detection
from utils.validators import FileValidator

# Configure logging
logger = logging.getLogger(__name__)

# Initialize services
settings = Settings()
translation_service = TranslationService()
document_processor = EnhancedDocumentProcessor(
    dpi=getattr(settings, "PDF_DPI", 300),
    preserve_images=getattr(settings, "PRESERVE_IMAGES", True),
)

# Initialize philosophy-enhanced services
neologism_detector = NeologismDetector()
user_choice_manager = UserChoiceManager()
philosophy_translation_service = PhilosophyEnhancedTranslationService(
    translation_service=translation_service,
    neologism_detector=neologism_detector,
    user_choice_manager=user_choice_manager,
)

language_detector = LanguageDetector()
file_handler = FileHandler()
file_validator = FileValidator()


def extract_file_info(file) -> Tuple[str, str, int]:
    """Extract file information from various Gradio file formats.

    Args:
        file: File object from Gradio (can be str, FileData, or
              file-like object)

    Returns:
        Tuple of (file_path, file_name, file_size)

    Raises:
        ValueError: If file type is unsupported, file is unreadable,
                   or any other file-related error occurs
    """
    if file is None:
        raise ValueError("No file uploaded")

    file_path: str = ""
    file_name: str = ""
    file_size: int = 0

    # Case 1: Newer Gradio returns a path string
    if isinstance(file, str):
        file_path = file
        file_name = os.path.basename(file_path)
        try:
            file_size = os.path.getsize(file_path)
        except OSError as e:
            logger.error(f"Failed to stat uploaded file: {e}")
            raise ValueError(f"Could not read uploaded file: {e!s}")

    # Case 2: FileData (dict-like) with .name / .size / .data attributes
    elif hasattr(file, "size") and hasattr(file, "name"):
        file_name = os.path.basename(file.name)
        file_size = file.size or 0
        try:
            file_path = file_handler.save_uploaded_file(file)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise ValueError(f"Could not save uploaded file: {e!s}")

    # Case 3: Legacy file-like object
    elif hasattr(file, "read"):
        file_name = getattr(file, "name", "uploaded_file")
        try:
            # compute size
            current = file.tell() if hasattr(file, "tell") else 0
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(current)
            file_path = file_handler.save_uploaded_file(file)
        except Exception as e:
            logger.error(f"Failed to process file-like object: {e}")
            raise ValueError(f"Could not process uploaded file: {e!s}")

    else:
        raise ValueError("Unsupported file object returned by Gradio")

    return file_path, file_name, file_size


async def process_file_upload(file) -> Tuple[str, str, str, str]:
    """Process uploaded file with advanced content extraction"""
    try:
        # Extract file information using the new function
        try:
            file_path, file_name, file_size = extract_file_info(file)
        except ValueError as e:
            return "", f"❌ {e!s}", "", ""

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
        preview = content.get("preview", "Preview not available")

        # Detect language using the centralized utility function
        sample_text = extract_text_sample_for_language_detection(content)
        detected_lang = language_detector.detect_language_from_text(sample_text)
        state.source_language = detected_lang

        # Store processing info
        metadata = content.get("metadata")
        if metadata:
            state.processing_info = {
                "file_type": metadata.file_type,
                "total_pages": metadata.total_pages,
                "total_elements": metadata.total_text_elements,
                "file_size_mb": metadata.file_size_mb,
                "processing_time": metadata.processing_time,
                "dpi": getattr(metadata, "dpi", "N/A"),
            }

        # Create status message with detailed info
        status_parts = [
            "✅ File processed successfully",
            f"📄 Type: {Path(file_name).suffix.upper()}",
            f"🌐 Detected language: {detected_lang}",
        ]

        if metadata:
            status_parts.extend(
                [
                    f"📊 Pages: {metadata.total_pages}",
                    f"📝 Text elements: {metadata.total_text_elements}",
                    f"💾 Size: {metadata.file_size_mb:.2f} MB",
                    f"⏱️ Processing: {metadata.processing_time:.2f}s",
                ]
            )

            if hasattr(metadata, "dpi"):
                status_parts.append(f"🖼️ Resolution: {metadata.dpi} DPI")

        status = "\n".join(status_parts)

        # Processing details for display
        processing_details = json.dumps(state.processing_info, indent=2)

        logger.info(f"Advanced processing complete: {file_name}")
        return preview, status, detected_lang, processing_details

    except Exception as e:
        logger.error(f"File upload error: {e!s}")
        return "", f"❌ Upload failed: {e!s}", "", ""


async def start_translation(
    target_language: str, max_pages: int, philosophy_mode: bool
) -> Tuple[str, str, bool]:
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
        state.philosophy_mode = philosophy_mode
        # Gradio sliders may return float; cast safely to int
        # Gradio sliders may return float; cast safely to int
        try:
            state.max_pages = int(max_pages) if max_pages else 0
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert max_pages to int: {e}, using default 0")
            state.max_pages = 0

        # Create session for philosophy mode
        if philosophy_mode:
            session = user_choice_manager.create_session(
                session_name=(
                    f"Philosophy Translation "
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ),
                document_name=(
                    Path(state.current_file).name if state.current_file else "Unknown"
                ),
                source_language=state.source_language,
                target_language=target_language,
            )
            state.session_id = session.session_id
            logger.info(f"Created philosophy session: {state.session_id}")

        # Start translation in background
        asyncio.create_task(perform_advanced_translation())

        return (
            "🚀 Advanced translation started...",
            "Translation in progress with format preservation",
            False,
        )

    except Exception as e:
        logger.error(f"Translation start error: {e!s}")
        return f"❌ Failed to start translation: {e!s}", "", False


async def translate_content(
    content: dict,
    source_language: str,
    target_language: str,
    max_pages: int = None,
    progress_callback=None,
    philosophy_mode: bool = False,
    session_id: str = None,
) -> Tuple[dict, str]:
    """Core translation function that handles text extraction and translation.

    Args:
        content: Extracted document content dictionary
        source_language: Source language code
        target_language: Target language code
        max_pages: Optional limit on number of pages to translate
        progress_callback: Optional callback function for progress updates
        philosophy_mode: Whether to use philosophy-enhanced translation
        session_id: Optional session ID for philosophy mode

    Returns:
        Tuple of (translated_by_page dict, output_filename)
    """
    try:
        # Extract text for translation based on content type
        if progress_callback:
            progress_callback(10)

        if content["type"] == "pdf_advanced":
            text_by_page = content["text_by_page"]
        elif content["type"] in ["docx", "txt"]:
            # Convert to page-based format for consistency
            if content["type"] == "docx":
                texts = [
                    para["text"]
                    for para in content["paragraphs"]
                    if para["text"].strip()
                ]
            else:
                texts = content["lines"]
            text_by_page = {0: texts}
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

        # Translate content page by page respecting max_pages limit
        if progress_callback:
            progress_callback(30)

        translated_by_page = {}

        # Determine pages to process based on numeric order
        page_keys = sorted(text_by_page.keys(), key=lambda x: int(x))
        if max_pages is not None and max_pages > 0:
            page_keys = page_keys[:max_pages]
        pages_to_process = {k: text_by_page[k] for k in page_keys}

        total_pages = len(pages_to_process)
        for idx, (page_num, page_texts) in enumerate(pages_to_process.items()):
            if not page_texts:
                translated_by_page[page_num] = []
                continue

            logger.info(
                f"Translating page {page_num + 1}/{total_pages} "
                f"({len(page_texts)} elements)"
            )

            # Choose translation service based on mode
            if philosophy_mode and session_id:
                # Use philosophy-enhanced translation
                translated_texts = []
                for text in page_texts:
                    if text.strip():
                        result = (
                            await (
                                philosophy_translation_service.translate_with_context(
                                    text=text,
                                    session_id=session_id,
                                    source_language=source_language,
                                    target_language=target_language,
                                )
                            )
                        )
                        translated_texts.append(result["translation"])
                    else:
                        translated_texts.append(text)
            else:
                # Batch translate text elements for efficiency
                batch_size = 20
                translated_texts = [None] * len(page_texts)

                # Identify non-empty text indices to translate
                indices_to_translate = [
                    idx for idx, txt in enumerate(page_texts) if txt.strip()
                ]

                for start in range(0, len(indices_to_translate), batch_size):
                    batch_indices = indices_to_translate[start : start + batch_size]
                    batch_texts = [page_texts[idx] for idx in batch_indices]

                    try:
                        batch_translated = await translation_service.translate_batch(
                            batch_texts, source_language, target_language
                        )
                    except Exception as e:
                        logger.warning(
                            f"Batch translation failed for page {page_num} "
                            f"(indices {batch_indices}): {e}"
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
            if progress_callback:
                page_progress = 30 + ((idx + 1) / total_pages * 50)
                progress_callback(int(page_progress))

        # Generate output filename
        file_path = content.get("file_path", "document")
        output_filename = generate_output_filename(
            Path(file_path).name if file_path else "document", target_language
        )

        return translated_by_page, output_filename

    except Exception as e:
        logger.error(f"Translation content error: {e!s}")
        raise


async def perform_advanced_translation() -> None:
    """Perform the advanced translation process with format preservation"""
    try:
        state.translation_status = "processing"
        logger.info(
            f"Starting advanced translation: {state.source_language} -> "
            f"{state.target_language}"
        )

        content = state.current_content
        content["file_path"] = state.current_file

        # Define progress callback
        def update_progress(progress: int):
            state.translation_progress = progress

        # Use the centralized translate_content function
        max_pages = state.max_pages if state.max_pages > 0 else None
        translated_by_page, output_filename = await translate_content(
            content=content,
            source_language=state.source_language,
            target_language=state.target_language,
            max_pages=max_pages,
            progress_callback=update_progress,
            philosophy_mode=state.philosophy_mode,
            session_id=getattr(state, "session_id", None),
        )

        # Generate output file with preserved formatting
        state.translation_progress = 85

        # If a page limit was applied, restrict layouts for PDF reconstruction
        if max_pages is not None:
            page_keys = list(translated_by_page.keys())
            limited_layouts = [
                lay for lay in content.get("layouts", []) if lay.page_num in page_keys
            ]
            content_limited = dict(content)
            content_limited["layouts"] = limited_layouts
        else:
            content_limited = content

        logger.info(f"Creating translated document: {output_filename}")
        state.output_file = document_processor.create_translated_document(
            content_limited, translated_by_page, output_filename
        )

        state.translation_progress = 100
        state.translation_status = "completed"

        logger.info(f"Advanced translation completed: {state.output_file}")

    except Exception as e:
        logger.error(f"Advanced translation error: {e!s}")
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
        return (
            f"🔄 Advanced translation in progress... "
            f"({state.translation_progress}%)",
            state.translation_progress,
            False,
        )
    elif state.translation_status == "completed":
        return ("✅ Advanced translation completed with format preservation!", 100, True)
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
                state.output_file, output_format.lower()
            )
            return converted_file

        return state.output_file

    except Exception as e:
        logger.error(f"Download preparation error: {e!s}")
        return f"❌ Download failed: {e!s}"


async def process_advanced_translation_job(
    job_id: str, file_path: str, source_lang: str, target_lang: str
) -> None:
    """Process translation job with advanced formatting preservation"""
    try:
        job = translation_jobs[job_id]
        job["status"] = "processing"

        # Extract content with advanced processing
        job["progress"] = 20
        content = document_processor.extract_content(file_path)
        content["file_path"] = file_path

        # Define progress callback for job updates
        def update_job_progress(progress: int):
            # Scale progress to job progress range (20-80)
            job["progress"] = 20 + int(progress * 0.6)

        # Use the centralized translate_content function
        translated_by_page, output_filename = await translate_content(
            content=content,
            source_language=source_lang,
            target_language=target_lang,
            max_pages=None,  # Process all pages for job
            progress_callback=update_job_progress,
            philosophy_mode=False,  # Jobs don't use philosophy mode
            session_id=None,
        )

        # Create output with preserved formatting
        job["progress"] = 80

        output_file = document_processor.create_translated_document(
            content, translated_by_page, output_filename
        )

        job["progress"] = 100
        job["status"] = "completed"
        job["output_file"] = output_file

        logger.info(f"Advanced translation job completed: {job_id}")
    except Exception as e:
        logger.error(f"Error in advanced translation job {job_id}: {e!s}")
        job["status"] = "failed"
        job["error"] = str(e)

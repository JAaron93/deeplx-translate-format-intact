"""FastAPI route handlers for document translation API."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from core.state_manager import state, translation_jobs
from core.translation_handler import (
    document_processor,
    file_handler,
    language_detector,
    neologism_detector,
    process_advanced_translation_job,
    user_choice_manager,
)
from dolphin_ocr.errors import get_error_message
from models.neologism_models import (
    ConfidenceFactors,
    DetectedNeologism,
    MorphologicalAnalysis,
    NeologismType,
    PhilosophicalContext,
)
from models.user_choice_models import (
    ChoiceScope,
    ChoiceType,
)
from utils import pdf_validator
from utils.language_utils import extract_text_sample_for_language_detection

logger = logging.getLogger(__name__)

# Templates
templates = Jinja2Templates(directory="templates")

# Create APIRouter instances
api_router = APIRouter()
app_router = APIRouter()


@app_router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Advanced Document Translator API",
        "ui_url": "/ui",
        "philosophy_ui_url": "/philosophy",
        "version": "2.0.0",
        "features": [
            "Advanced PDF processing",
            "Image-text overlay preservation",
            "High-resolution rendering",
            "Comprehensive format support",
            "Philosophy-enhanced neologism detection",
            "User choice management for translations",
        ],
    }


@app_router.get("/philosophy", response_class=HTMLResponse)
async def philosophy_interface(request: Request) -> HTMLResponse:
    """Philosophy-enhanced translation interface."""
    return templates.TemplateResponse("philosophy_interface.html", {"request": request})


# Philosophy API Endpoints
@api_router.post("/philosophy/choice")
async def save_user_choice(choice_data: Dict[str, Any]):
    """Save a user choice for a neologism."""
    try:
        # Extract choice data
        term = choice_data.get("term")
        choice = choice_data.get("choice")
        custom_translation = choice_data.get("custom_translation", "")
        notes = choice_data.get("notes", "")
        session_id = choice_data.get("session_id")

        # Create a simple neologism representation
        neologism = DetectedNeologism(
            term=term,
            confidence=0.8,
            neologism_type=NeologismType.PHILOSOPHICAL_TERM,
            start_pos=0,
            end_pos=len(term),
            sentence_context="Context sentence",
            morphological_analysis=MorphologicalAnalysis(),
            philosophical_context=PhilosophicalContext(),
            confidence_factors=ConfidenceFactors(),
        )

        # Map choice string to ChoiceType
        choice_type_mapping = {
            "preserve": ChoiceType.PRESERVE,
            "translate": ChoiceType.TRANSLATE,
            "custom": ChoiceType.CUSTOM_TRANSLATION,
        }

        choice_type = choice_type_mapping.get(choice, ChoiceType.PRESERVE)

        # Save the choice
        user_choice = user_choice_manager.make_choice(
            neologism=neologism,
            choice_type=choice_type,
            translation_result=custom_translation,
            session_id=session_id,
            choice_scope=ChoiceScope.CONTEXTUAL,
            user_notes=notes,
        )

        return {
            "success": True,
            "choice_id": user_choice.choice_id,
            "message": "Choice saved successfully",
        }

    except Exception as e:
        logger.error(f"Error saving user choice: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.get("/philosophy/neologisms")
async def get_detected_neologisms(_session_id: Optional[str] = None):
    """Get detected neologisms for the current session.

    Args:
        _session_id: Session identifier (reserved for future use)
    """
    try:
        # Return neologisms from state
        neologisms = (
            state.neologism_analysis.get("detected_neologisms", [])
            if state.neologism_analysis
            else []
        )
        total = (
            len(state.neologism_analysis.get("detected_neologisms", []))
            if state.neologism_analysis
            else 0
        )
        return {"neologisms": neologisms, "total": total}
    except Exception as e:
        logger.error(f"Error getting neologisms: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.get("/philosophy/progress")
async def get_philosophy_progress():
    """Get current philosophy processing progress."""
    try:
        total_neologisms = 0
        if state.neologism_analysis and isinstance(state.neologism_analysis, dict):
            detected = state.neologism_analysis.get("detected_neologisms", [])
            if isinstance(detected, list):
                total_neologisms = len(detected)

        processed_neologisms = 0
        if isinstance(state.user_choices, list):
            processed_neologisms = len(
                [
                    choice
                    for choice in state.user_choices
                    if isinstance(choice, dict) and choice.get("processed", False)
                ]
            )
        return {
            "total_neologisms": total_neologisms,
            "processed_neologisms": processed_neologisms,
            "choices_made": len(state.user_choices),
            "session_id": state.session_id,
            "philosophy_mode": state.philosophy_mode,
        }
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.post("/philosophy/export-choices")
async def export_user_choices(export_data: Dict[str, Any]):
    """Export user choices to JSON."""
    try:
        session_id = export_data.get("session_id")

        if session_id:
            file_path = user_choice_manager.export_session_choices(session_id)
        else:
            file_path = user_choice_manager.export_all_choices()

        if file_path:
            return FileResponse(
                file_path,
                media_type="application/json",
                filename=(
                    "philosophy-choices-"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                ),
            )
        else:
            raise HTTPException(status_code=500, detail="Export failed")

    except Exception as e:
        logger.error(f"Error exporting choices: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.post("/philosophy/import-choices")
async def import_user_choices(import_data: Dict[str, Any]):
    """Import user choices from dictionary."""
    try:
        choices = import_data.get("choices", {})
        session_id = import_data.get("session_id")

        # Validate that choices is a dictionary
        if not isinstance(choices, dict):
            raise HTTPException(
                status_code=400, detail="'choices' must be a dictionary"
            )

        # Use the new dictionary-accepting method
        count = user_choice_manager.import_choices_from_dict(choices, session_id)

        return {
            "success": True,
            "count": count,
            "message": f"Imported {count} choices successfully",
        }

    except ValueError as e:
        logger.error(f"Validation error importing choices: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error importing choices: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.get("/philosophy/terminology")
async def get_terminology():
    """Get current terminology database."""
    try:
        # Get terminology from neologism detector
        terminology = neologism_detector.terminology_map
        return terminology

    except Exception as e:
        logger.error(f"Error getting terminology: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):  # noqa: B008
    """Enhanced upload endpoint with advanced processing."""
    try:
        # Save file first so validators can inspect header and structure
        file_path = file_handler.save_upload_file(file)

        # Basic format validation
        fmt = pdf_validator.validate_pdf_extension_and_header(file_path)
        if not fmt.ok:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "DOLPHIN_005",
                    "message": "Only PDF format supported",
                    "timestamp": datetime.now().isoformat(),
                    "context": {"path": Path(file_path).name},
                },
            )

        # Encryption check
        enc = pdf_validator.detect_pdf_encryption(file_path)
        if enc.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "DOLPHIN_014",
                    "message": get_error_message("DOLPHIN_014"),
                    "timestamp": datetime.now().isoformat(),
                    "context": {"path": Path(file_path).name},
                },
            )

        # Process with advanced extraction
        content = document_processor.extract_content(file_path)

        # Detect language using the utility function
        sample_text = extract_text_sample_for_language_detection(content)
        detected_lang = language_detector.detect_language_from_text(sample_text)

        # Clean metadata access pattern
        metadata = content.get("metadata")
        if metadata:
            if hasattr(metadata, "__dict__"):
                metadata_dict = metadata.__dict__
            elif isinstance(metadata, dict):
                metadata_dict = metadata
            else:
                metadata_dict = None
        else:
            metadata_dict = None

        return {
            "message": "File processed with advanced extraction",
            "filename": file.filename,
            "detected_language": detected_lang,
            "file_path": file_path,
            "content_type": content["type"],
            "metadata": metadata_dict,
        }

    except HTTPException:
        # Allow previously constructed HTTP errors to pass through
        raise
    except Exception as e:
        except Exception as e:
            logger.error("Enhanced upload error: %s", e)
-           return {
-               "detail": {
-                   "error_code": "DOLPHIN_002",
-                   "message": str(e),
-                   "timestamp": datetime.now().isoformat(),
-               }
-           }, 500
+           raise HTTPException(
+               status_code=500,
+               detail={
+                   "error_code": "DOLPHIN_002",
+                   "message": str(e),
+                   "timestamp": datetime.now().isoformat(),
+               }
+           ) from e


@api_router.post("/translate")
async def translate_document(
    background_tasks: BackgroundTasks,
    file_path: str,
    source_language: str,
    target_language: str,
):
    """Enhanced translation endpoint."""
    try:
        import uuid

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
            "format_preservation": True,
        }

        # Start background translation with advanced processing
        background_tasks.add_task(
            process_advanced_translation_job,
            job_id,
            file_path,
            source_language,
            target_language,
        )

        return {
            "job_id": job_id,
            "status": "started",
            "type": "advanced",
        }

    except Exception as e:
        logger.error(f"Enhanced translation start error: {e!s}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get enhanced job status."""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return translation_jobs[job_id]


@api_router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download translated file with enhanced metadata."""
    if job_id not in translation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = translation_jobs[job_id]
    if (job["status"] != "completed") or (not job["output_file"]):
        raise HTTPException(
            status_code=400,
            detail="Translation not completed",
        )

    return FileResponse(
        job["output_file"],
        media_type="application/octet-stream",
        filename=Path(job["output_file"]).name,
        headers={
            "X-Processing-Type": "advanced",
            "X-Format-Preserved": "true",
        },
    )

from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path
from typing import Optional


class UnsupportedFormatError(Exception):
    """Raised when an unsupported or invalid document format is encountered.

    Optionally carries an error_code for standardized codes
    (e.g., DOLPHIN_014 for encrypted PDFs).
    """

    def __init__(self, message: str, *, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class DocumentReconstructionError(Exception):
    """Raised when PDF reconstruction fails."""


class PDFEncryptionCheckWarning(RuntimeWarning):
    """Warning raised when PDF encryption could not be checked.

    Emitted when basic validation passes but pypdf-based encryption check
    failed due to environment or parsing issues.
    """


class PDFDocumentReconstructor:
    """Validate PDF format before any reconstruction work.

    This class focuses on basic format checks:
    - Extension must be .pdf (case-insensitive)
    - File must exist and begin with %PDF- header
    - Reject encrypted PDFs (DOLPHIN_014) when detected via pypdf
    """

    supported_extension = ".pdf"

    @classmethod
    def is_pdf_format(cls, file_path: str | PathLike[str]) -> bool:
        """Return True if the path ends with the configured PDF extension.

        Implemented as a classmethod so subclasses can override
        `supported_extension` and have this method reflect that override.
        Comparison is Unicode-safe and tolerant of missing leading dot in
        subclass overrides.
        """
        ext = Path(file_path).suffix.casefold().lstrip(".")
        configured = str(cls.supported_extension).casefold().lstrip(".")
        return ext == configured

    def validate_pdf_format_or_raise(self, file_path: str) -> None:
        """Validate that a file is a readable, non-encrypted PDF.

        Raises UnsupportedFormatError when any requirement is not met.
        - Unsupported extension
        - File not found
        - Missing %PDF- header
        - Encrypted PDF (error_code DOLPHIN_014)
        """
        p = Path(file_path)

        # Extension check first via classmethod to honor subclass overrides
        if not type(self).is_pdf_format(file_path):
            ext = p.suffix or "(none)"
            raise UnsupportedFormatError(
                f"Unsupported format '{ext}'; only PDF is supported."
            )

        # Existence and header check
        if not p.exists():
            raise UnsupportedFormatError(f"File not found: {file_path}")

        try:
            with p.open("rb") as f:
                head = f.read(5)
        except OSError as e:
            raise UnsupportedFormatError(
                f"Unable to read file: {file_path}: {e}"
            ) from e

        if head != b"%PDF-":
            raise UnsupportedFormatError(
                "File content is not a valid PDF (missing %PDF- header)."
            )

        # Encrypted PDF detection via pypdf, if available
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            if getattr(reader, "is_encrypted", False):
                # Standardized rejection per requirements (DOLPHIN_014)
                raise UnsupportedFormatError(
                    ("Encrypted PDFs not supported - " "please provide unlocked PDF"),
                    error_code="DOLPHIN_014",
                )
        except ModuleNotFoundError:
            # pypdf not installed; skip encryption detection
            pass
        except UnsupportedFormatError:
            # Re-raise our own exceptions
            raise
        except Exception as e:
            # Warn but don't fail on pypdf errors - the file passed basic checks
            warnings.warn(
                f"Warning: Could not check PDF encryption: {e}",
                category=PDFEncryptionCheckWarning,
                stacklevel=2,
            )

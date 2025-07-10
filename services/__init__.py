"""
Core services package containing document processing, translation,
and language detection functionality.
"""

import logging

logger = logging.getLogger(__name__)

# Conditional imports with external dependency handling
_available_services = []
_service_availability = {}

# Always available services (no external dependencies)
try:
    from .neologism_detector import NeologismDetector  # noqa: F401
    _available_services.append('NeologismDetector')
    _service_availability['NEOLOGISM_DETECTOR_AVAILABLE'] = True
    logger.debug("NeologismDetector imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import NeologismDetector: {e}")
    _service_availability['NEOLOGISM_DETECTOR_AVAILABLE'] = False

# Services with external dependencies
try:
    from .translation_service import TranslationService  # noqa: F401
    _available_services.append('TranslationService')
    _service_availability['TRANSLATION_SERVICE_AVAILABLE'] = True
    logger.debug("TranslationService imported successfully")
except ImportError as e:
    logger.debug(f"TranslationService not available: {e}")
    _service_availability['TRANSLATION_SERVICE_AVAILABLE'] = False

try:
    from .language_detector import LanguageDetector  # noqa: F401
    _available_services.append('LanguageDetector')
    _service_availability['LANGUAGE_DETECTOR_AVAILABLE'] = True
    logger.debug("LanguageDetector imported successfully")
except ImportError as e:
    logger.debug(f"LanguageDetector not available: {e}")
    _service_availability['LANGUAGE_DETECTOR_AVAILABLE'] = False

try:
    from .enhanced_document_processor import (  # noqa: F401
        EnhancedDocumentProcessor
    )
    _available_services.append('EnhancedDocumentProcessor')
    _service_availability['ENHANCED_DOCUMENT_PROCESSOR_AVAILABLE'] = True
    logger.debug("EnhancedDocumentProcessor imported successfully")
except ImportError as e:
    logger.debug(f"EnhancedDocumentProcessor not available: {e}")
    _service_availability['ENHANCED_DOCUMENT_PROCESSOR_AVAILABLE'] = False

# Dynamically build __all__ list based on successfully imported services
__all__ = _available_services.copy()

# Export availability flags for runtime dependency checking
NEOLOGISM_DETECTOR_AVAILABLE = _service_availability[
    'NEOLOGISM_DETECTOR_AVAILABLE'
]
TRANSLATION_SERVICE_AVAILABLE = _service_availability[
    'TRANSLATION_SERVICE_AVAILABLE'
]
LANGUAGE_DETECTOR_AVAILABLE = _service_availability[
    'LANGUAGE_DETECTOR_AVAILABLE'
]
ENHANCED_DOCUMENT_PROCESSOR_AVAILABLE = _service_availability[
    'ENHANCED_DOCUMENT_PROCESSOR_AVAILABLE'
]

# Log summary of available services
services_count = len(_available_services)
services_list = ', '.join(_available_services)
logger.info(f"Services module initialized with {services_count} "
            f"available services: {services_list}")
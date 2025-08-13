# Implementation Plan

Convert the Dolphin OCR migration design into a series of prompts for a code-generation LLM that will implement each step in a test-driven manner. Prioritize best practices, incremental progress, and early testing, ensuring no big jumps in complexity at any stage. Make sure that each prompt builds on the previous prompts, and ends with wiring things together. There should be no hanging or orphaned code that isn't integrated into a previous step. Focus ONLY on tasks that involve writing, modifying, or testing code.

- [x] 1. Set up core infrastructure and configuration management
  - Create basic configuration management system with environment variable loading
  - Implement standardized error codes and exception classes for Dolphin OCR system
  - Set up basic logging configuration
  - Create base test fixtures and utilities for Dolphin OCR testing
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 8.1, 8.2_

- [x] 2. Implement PDF-to-Image conversion service
  - [x] 2.1 Create PDFToImageConverter class with pdf2image integration
    - Write PDFToImageConverter class with configurable DPI and format settings
    - Implement convert_pdf_to_images method with memory-efficient page processing
    - Add image optimization for OCR processing with Pillow integration
    - Create unit tests for PDF conversion with various document types
    - _Requirements: 1.3, 4.1, 4.2_

  - [x] 2.2 Add error handling and validation for PDF conversion
    - Implement robust error handling for corrupted PDFs and unsupported formats
    - Add file validation and format checking before conversion
    - Create basic test suite for common error scenarios
    - Integrate with standardized error codes (DOLPHIN_005, DOLPHIN_011)
    - _Requirements: 8.1, 8.2, 1.5_

- [x] 3. Create enhanced Dolphin OCR service integration
  - [x] 3.1 Implement DolphinOCRService class with Hugging Face authentication
    - Write DolphinOCRService class with HF token authentication and Modal endpoint configuration
    - Implement process_document_images method with batch optimization
    - Add basic OCR result validation and structure checking
    - Create unit tests for OCR service with mocked API responses
    - _Requirements: 5.1, 5.2, 5.3, 2.1, 2.2_

  - [x] 3.2 Add basic performance optimization and error handling for OCR service
    - Implement basic retry mechanisms with exponential backoff for rate limits
    - Add timeout handling and graceful failure for service unavailability
    - Create basic performance logging with response times and success rates
    - Write integration tests with actual Dolphin OCR API calls
    - Gate live API tests behind an env flag (e.g., RUN_LIVE_DOLPHIN_TESTS=true) and skip by default in CI
    - _Requirements: 4.5, 4.6, 4.7, 8.3, 8.4, 8.5_

- [x] 4. Develop layout preservation engine
  - [x] 4.1 Create core layout analysis and strategy determination
    - Write LayoutPreservationEngine class with text fit analysis capabilities
    - Implement determine_layout_strategy method with font scaling, text wrapping, and hybrid approaches
    - Add quality score calculation for layout preservation assessment
    - Create unit tests for layout strategy selection with various text length ratios
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.2 Implement layout adjustment application and optimization
    - Write apply_layout_adjustments method with font scaling and text wrapping logic
    - Add basic text wrapping respecting word boundaries
    - Implement bbox expansion and hybrid strategy application
    - Create basic test suite for layout adjustments
    - _Requirements: 3.4, 3.5_

- [x] 5. Build layout-aware translation service
  - [x] 5.1 Create translation service with layout constraints
    - Write LayoutAwareTranslationService class integrating McpLingoClient and LayoutPreservationEngine
    - Implement translate_with_layout_constraints method with length optimization
    - Add batch translation processing with layout context preservation
    - Create unit tests for translation with layout constraint handling
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 5.2 Add translation optimization and quality assurance
    - Implement translation length optimization for layout fitting
    - Add confidence score tracking from both OCR and translation processes
    - Create translation quality validation with layout impact assessment
    - Write integration tests for complete translation workflow with layout preservation
    - _Requirements: 7.4, 7.5_

- [x] 6. Implement PDF document reconstructor
  - [x] 6.1 Create PDFDocumentReconstructor with basic format validation
    - Write PDFDocumentReconstructor class with basic PDF format validation using file extension and simple header checks
    - Implement basic PDF format checking methods (is_pdf_format, validate_pdf_format_or_raise)
    - Add encrypted PDF detection and rejection with error code DOLPHIN_014
    - Add custom exceptions (UnsupportedFormatError, DocumentReconstructionError) with specific error messages
    - Create unit tests for PDF format validation, encrypted PDF detection, and error handling scenarios
    - Detect encryption via pypdf (PdfReader.is_encrypted); reject with DOLPHIN_014
    - Unit tests: encrypted PDFs, basic format validation, error handling
    - _Requirements: 6.1, 6.2, 6.9, 6.10_

  - [x] 6.2 Implement PDF reconstruction with layout preservation
    - Write reconstruct_pdf method using reportlab with translated layout integration
    - Implement font handling, text positioning, and multi-line text wrapping
    - Add color preservation and style application from original layout
    - Create unit tests for PDF reconstruction with various layout scenarios
    - _Requirements: 6.1, 6.2, 6.8_

  - [x] 6.3 Implement PDF quality validation helper methods
    - Create _extract_pdf_text helper method using hybrid approach combining pdfminer.six/pypdf direct extraction and Tesseract OCR fallback (language-configurable)
    - Stream and/or page-chunk OCR to avoid OOM on large PDFs; cap pages/time and expose timeouts
    - Optimize PDFMiner chunking to use iterator slicing instead of full-file iteration per chunk, preventing O(n²) page reading for large documents
    - Optimize OCR fallback per-page convert_from_path: implement batch processing to reduce PDF reopening overhead, allow poppler_path configuration via environment variable, and handle poppler absence gracefully
    - Implement _compute_text_accuracy method with simple length-based checks: ensure translated text is reasonable length compared to original
    - Write _compare_layout_hashes method with basic similarity check: compare hash lengths and return simple score based on length similarity
    - Add basic error handling and fallback mechanisms for text extraction failures
    - Enhance summary logging with warning category breakdown: aggregate warnings into dictionary of reason->count (e.g., "OCR confidence low": 5, "image corruption": 2, "page skip": 1), include category counts in warning/info messages, and clamp verbosity for large documents by truncating logged text snippets to fixed safe length (e.g., 200 chars) with "(truncated)" indicator
    - Create unit tests for each helper method with various PDF types (text-based, image-based, hybrid)
    - _Requirements: 6.11, 6.12_

  - [x] 6.4 Add basic PDF quality validation
    - Implement validate_pdf_reconstruction_quality method with simple quality checks
    - Add basic threshold checking for text accuracy, font preservation, and layout preservation
    - Create simple quality assessment: check if text is preserved, fonts are mostly correct, and layout is roughly maintained
    - Create basic test suite for PDF quality validation with simple pass/fail criteria
    - _Requirements: 6.3, 6.4, 6.5, 6.10, 6.11, 6.12_

- [x] 7. Create basic error handling and monitoring system
  - [x] 7.1 Implement error handling strategy with standardized codes
    - Write ErrorHandlingStrategy class with basic API error handling and recovery mechanisms
    - Implement specific handlers for rate limits (DOLPHIN_001), service unavailability (DOLPHIN_002), authentication failures (DOLPHIN_003), processing timeouts (DOLPHIN_004), invalid document format (DOLPHIN_005), and encrypted PDFs (DOLPHIN_014)
    - Add basic error logging with error code, timestamp, and context
    - Create unit tests for error handling scenarios with proper error code assignment
    - Maintain a simple registry (e.g., errors/codes.py) enumerating all DOLPHIN_* codes with descriptions
    - _Requirements: 8.1, 8.2_

  - [x] 7.2 Build basic monitoring system
    - Write simple MonitoringService class with basic performance metrics tracking
    - Add basic error rate monitoring and performance logging
    - Create simple logging for service health and basic metrics
    - Write unit tests for monitoring functionality
    - _Requirements: 8.3, 8.4, 8.5_

 - [-] 8. Integrate complete document processing workflow
  - [x] 8.1 Create main document processor orchestrating all services
    - Write DocumentProcessor class integrating all services (PDF converter, OCR, translation, reconstruction)
    - Implement process_document method with complete workflow orchestration
    - Add request validation, progress tracking, and result compilation
    - Create integration tests for complete document processing workflow
    - _Requirements: 6.1, 6.2, 6.3_

  - [-] 8.2 Add basic async processing and concurrent handling
    - Implement async processing patterns for document handling
    - Add basic concurrent request management with simple limits and queue handling
    - Choose concurrency models explicitly:
      - Use asyncio + task groups for IO-bound OCR/translation calls
      - Use a process pool (or workers) for CPU-bound PDF/image processing
      - Add basic rate-limiting aligned with OCR provider
    - _Requirements: 4.5, 4.6, 4.7_

- [ ] 9. Update API endpoints and user interface integration
  - [ ] 9.1 Implement basic PDF validation utilities
    - Create utils/pdf_validator.py with basic PDF validation functions
    - Implement basic file extension and header validation for PDF format
    - Implement detect_pdf_encryption() using PyPDF2/pypdf to check encryption flags and password protection
    - Add basic validate_pdf_structure() to detect obviously corrupted PDF files
    - Create simple validation response objects with error codes (DOLPHIN_005, DOLPHIN_014) and user-friendly messages
    - Add unit tests for validation functions with basic file types (PDF, encrypted PDFs, non-PDF files)
    - _Requirements: 6.1, 6.2, 8.1, 8.2_

  - [ ] 9.2 Modify FastAPI routes with basic server-side validation
    - Update existing API routes in api/routes.py to use new DocumentProcessor and PDF validation utilities
    - Remove all PyMuPDF dependencies and imports from API layer
    - Integrate basic PDF validation utilities: call basic format validation and detect_pdf_encryption() before processing
    - Return error code DOLPHIN_005 with message "Only PDF format supported" for non-PDF uploads
    - Return error code DOLPHIN_014 with message "Encrypted PDFs not supported - please provide unlocked PDF" for encrypted PDFs
    - Add basic error handling with standardized error codes in API responses for validation failures
    - Create API integration tests validating new workflow endpoints with valid and invalid file uploads
    - _Requirements: 1.1, 1.2, 1.5, 6.1, 6.2_

  - [ ] 9.3 Update Gradio interface with basic validation and server integration
    - Modify ui/gradio_interface.py to integrate with new PDF document processing workflow
    - Add progress indicators and basic quality metrics display for OCR processing
    - Implement basic client-side file upload validation for user experience (file extension checks)
    - Handle server-side validation responses: display error code DOLPHIN_005 for non-PDF uploads and DOLPHIN_014 for encrypted PDFs
    - Add appropriate user messaging for validation failures
    - Create UI integration tests for PDF document upload, validation, and processing workflow
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.1, 6.2_

- [ ] 10. Create test suite with reproducible test documents
  - [ ] 10.1 Set up simple test documents
    - Use the three test PDFs from assets directory: simple text (11 pages), complex layout (single page with embedded image), scanned book (149 pages)
    - Create basic test functions that can process these three documents and check if they work
    - Add simple validation: can the translated PDFs be opened and do they contain text
    - _Requirements: 10.1, 10.2, 10.3_

  - [ ] 10.2 Build basic integration tests
    - Write simple integration tests that process the three test documents end-to-end
    - Check that each test document can be translated without crashing
    - Verify that translated PDFs can be opened and contain reasonable text
    - Add basic smoke tests to catch obvious failures during development
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 11. Complete PyMuPDF removal and dependency cleanup
  - [ ] 11.1 Remove PyMuPDF dependencies and legacy code
    - Remove all PyMuPDF (fitz) imports and dependencies from codebase
    - Delete services/advanced_pdf_processor.py and other PyMuPDF-based services
    - Update requirements.txt to remove PyMuPDF and add new dependencies (pdf2image, reportlab)
    - Create migration validation tests ensuring no PyMuPDF code remains
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ] 11.2 Update documentation and deployment configuration
    - Update README.md and documentation to reflect new Dolphin OCR PDF-only capabilities
    - Modify deployment scripts and environment configuration for new dependencies
    - Add environment variable documentation for all new configuration options
    - Document system dependencies and installation (e.g., poppler for pdf2image)
    - Pin minimal supported versions: pdf2image>=3.1.0, Pillow>=10.0.0, reportlab>=4.0.0
    - Include font installation guidance for consistent PDF rendering across environments
    - Create deployment validation tests ensuring proper configuration and functionality
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

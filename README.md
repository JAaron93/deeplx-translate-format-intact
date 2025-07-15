# Advanced Document Translator

Professional document translation with comprehensive formatting preservation based on the amazon-translate-pdf approach.

## 🚀 Features

### Advanced PDF Processing
- **Image-text overlay technique** for superior formatting preservation
- **High-resolution rendering** (300 DPI) for precise text positioning
- **Complete layout analysis** with text element extraction
- **Background image preservation** with text overlay reconstruction
- **Comprehensive metadata extraction** including fonts, colors, and positioning

### Document Format Support
- **PDF**: Advanced processing with image-text overlay preservation
- **DOCX**: Structure-aware processing with style preservation
- **TXT**: Simple text processing with line-based translation

### Translation Features
- **Multiple translation services** (DeepL, Google Translate, Azure)
- **Automatic language detection** with confidence scoring
- **Element-by-element translation** preserving original layout
- **Error handling and fallback** to original text if translation fails
- **Progress tracking** with detailed status updates

## 🏗️ Architecture

### Core Components

1. **AdvancedPDFProcessor** (`services/advanced_pdf_processor.py`)
   - High-resolution page rendering
   - Precise text element extraction with positioning
   - Image-text overlay reconstruction
   - Layout backup and recovery

2. **EnhancedDocumentProcessor** (`services/enhanced_document_processor.py`)
   - Multi-format document processing
   - Content type detection and routing
   - Format conversion capabilities
   - Preview generation

3. **Advanced Web Interface** (`app.py`)
   - Enhanced Gradio UI with processing details
   - Real-time progress tracking
   - Comprehensive status reporting
   - Advanced download options

### Processing Pipeline

```
Document Upload
    ↓
Format Detection & Validation
    ↓
Advanced Content Extraction
    ├── PDF: Image rendering + text positioning
    ├── DOCX: Structure analysis + style extraction
    └── TXT: Line-based processing
    ↓
Language Detection
    ↓
Element-by-Element Translation
    ↓
Format Reconstruction
    ├── PDF: Image background + text overlay
    ├── DOCX: Style-preserved document
    └── TXT: Translated text file
    ↓
Download with Format Options
```

## 🛠️ Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure translation services in `config/settings.py`
4. Run the application:
   ```bash
   python app.py
   ```

## 📁 Key Files

- `app.py` - Main application with advanced Gradio interface
- `services/advanced_pdf_processor.py` - Advanced PDF processing engine
- `services/enhanced_document_processor.py` - Multi-format document handler
- `services/translation_service.py` - Translation service integrations
- `services/language_detector.py` - Language detection utilities
- `config/settings.py` - Configuration management
- `utils/` - File handling and validation utilities

## 🔧 Configuration

### PDF Processing Settings
- `PDF_DPI`: Resolution for PDF rendering (default: 300)
- `PRESERVE_IMAGES`: Whether to preserve embedded images (default: True)
- `MEMORY_THRESHOLD`: Memory usage threshold for garbage collection

### Translation Services
Configure API keys and endpoints in `config/settings.py` for:
- DeepL API
- Google Cloud Translate
- Azure Cognitive Services Translator

## 🎯 Advantages Over PyMuPDF-Only Approach

1. **Superior Formatting Preservation**
   - Image-text overlay technique maintains exact visual layout
   - High-resolution rendering captures fine details
   - Precise text positioning with pixel-level accuracy

2. **Comprehensive Layout Analysis**
   - Complete extraction of text elements with metadata
   - Font, color, and styling information preservation
   - Advanced handling of complex page structures

3. **Robust Error Handling**
   - Graceful degradation when translation fails
   - Memory management for large documents
   - Automatic cleanup and resource management

4. **Enhanced User Experience**
   - Real-time processing status with detailed metrics
   - Advanced preview with processing information
   - Multiple output format options

## 📊 Processing Metrics

The system provides detailed processing metrics:
- File type and size analysis
- Processing time tracking
- Text element count and distribution
- Memory usage monitoring
- Translation progress and success rates

## 🔄 Migration from PyMuPDF

The old PyMuPDF-based implementation has been moved to `app_old_pymupdf.py` for reference. The new implementation:

- ✅ Replaces basic text extraction with advanced layout analysis
- ✅ Implements image-text overlay for better formatting preservation
- ✅ Adds comprehensive metadata extraction
- ✅ Provides detailed processing information
- ✅ Includes layout backup and recovery mechanisms

## 🚨 Important Notes

- The system still uses PyMuPDF but with an enhanced approach
- Image-text overlay technique requires more memory but provides superior results
- Processing time may be longer due to high-resolution rendering
- Layout backups are automatically created for complex documents

## 📈 Performance Considerations

- **Memory Usage**: Higher due to image rendering, managed with automatic garbage collection
- **Processing Time**: Longer for complex documents, with progress tracking
- **Quality**: Significantly improved formatting preservation
- **Scalability**: Designed for production use with proper resource management

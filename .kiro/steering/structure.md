# Project Structure

## Directory Organization

### Core Application
```
app.py                 # Main FastAPI application entry point
api/                   # FastAPI route handlers and API endpoints
├── routes.py          # Main API routes and web interface routes
ui/                    # Gradio interface components
├── gradio_interface.py # Web UI for document upload and processing
```

### Business Logic (Services Layer)
```
services/              # Core business logic and processing services
├── translation_service.py              # Base translation service
├── enhanced_translation_service.py     # Drop-in replacement with parallel processing
├── parallel_translation_service.py     # High-performance parallel translation engine
├── advanced_pdf_processor.py           # PDF processing with image-text overlay
├── enhanced_document_processor.py      # Multi-format document handler
├── document_processor.py               # Base document processing
├── language_detector.py                # Language detection utilities
├── neologism_detector.py               # Philosophy-focused neologism detection
├── morphological_analyzer.py           # Text analysis and morphology
├── philosophical_context_analyzer.py   # Philosophy-specific processing
├── user_choice_manager.py              # User preference management
└── confidence_scorer.py                # Translation confidence scoring
```

### Data Models
```
models/                # Pydantic models and data structures
├── user_choice_models.py  # User preference and choice models
└── neologism_models.py    # Neologism detection and analysis models
```

### Core Infrastructure
```
core/                  # Core application infrastructure
├── state_manager.py   # Application state and job management
└── translation_handler.py # Translation workflow coordination
```

### Configuration
```
config/                # Configuration files and settings
├── settings.py        # Main application settings
├── main.py           # Configuration management
├── languages.json    # Supported language definitions
├── klages_terminology.json      # Philosophy terminology mappings
├── philosophical_indicators.json # Philosophy-specific indicators
└── debug_test_words.json       # Debug and test data
```

### Data Layer
```
database/              # Database and persistence layer
├── choice_database.py # User choice persistence
└── user_choices.db   # SQLite database file
```

### Utilities
```
utils/                 # Shared utility functions
├── file_handler.py    # File I/O operations
├── language_utils.py  # Language processing utilities
└── validators.py      # Input validation functions
```

### Testing
```
tests/                 # Test suite
├── test_*.py         # Unit and integration tests
└── services/         # Service-specific tests
```

### Static Assets & Templates
```
static/               # Static web assets
├── philosophy_interface.css  # Custom CSS
└── philosophy_interface.js   # Frontend JavaScript

templates/            # Jinja2 templates
└── philosophy_interface.html # Web interface template
```

### Working Directories
```
uploads/              # Temporary file uploads
downloads/            # Generated translated documents
input/                # Input document staging
output/            esencige dependross-packa for clute importsd
- Absoerrege pref same packats withinlative impor Re
-torts lasmption ipplicacal a- Lod
orts seconrty imp
- Third-pats first imporibraryStandard lon
- t Organizati# Imporsor.py`

#esr `_procrvice.py` oh `_se* end witrvice files*n
- **Sextensio`.json` es with ptive name* use descriion files*nfigurat
- **Coest_`h `tefixed wit files** pr*Testce.py`)
- *rvilation_sehanced_trans(e.g., `enose rpng pudicatiames** inescriptive ns
- **D directorie files andor Python** fke_case
- **SnationsvenConing  File Nam

##plicationout apg throughinggDetailed lol text)
- to originallback ation (faegradraceful dervices
- Gndling in sexception haensive omprehing
- CHandlrror ### Eon data

ratiic configustatN files for - JSOpy`
s.g/settinggs in `confized settinCentralienv`
- via `.iguration nfcosed vironment-ba- Engement
on Manaigurati### Confic

iness logs and buscces an datan betweer separatio Clea
-odels/`res in `mta structus define da
- Modelayerase/` latabted in `dions isolaeratatabase optern
- Dy Pat Repositor##
#ance
formr peroughout fot thrawai- Async/s
ndler route hanjected intoe ies ar
- Serviclitybi responsiinglehas a sice - Each servogic
s lsines buin allconta` ces/n `servivices ittern
- Ser Layer Pa# Servicetterns

##tecture Pa Archi##
```

n logsApplicatio        # s/  
logssing filesroceemporary p       # T       emp/   taging
tument s Output doc   #

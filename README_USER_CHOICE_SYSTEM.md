# User Choice Management System

A comprehensive system for managing user translation choices in philosophy-focused translation work, enabling users to control how neologisms are handled throughout their translation workflow.

## Overview

The User Choice Management System provides sophisticated control over neologism translation decisions, allowing users to:
- Make translation choices for detected neologisms
- Reuse choices across similar contexts
- Resolve conflicts between different choices
- Maintain consistency across translation sessions
- Import/export terminology preferences
- Track usage statistics and analytics

## Architecture

The system consists of four main components:

### 1. Data Models (`models/user_choice_models.py`)
- **UserChoice**: Core choice representation with context matching
- **ChoiceSession**: Session management for document workflows
- **ChoiceConflict**: Conflict detection and resolution
- **TranslationPreference**: User preference management
- **TranslationContext**: Context-aware matching system

### 2. Database Layer (`database/choice_database.py`)
- SQLite-based persistent storage
- Optimized schema with proper indexing
- Foreign key constraints for data integrity
- Context-aware search capabilities
- Export/import functionality

### 3. Core Service (`services/user_choice_manager.py`)
- Main orchestration service
- Session lifecycle management
- Choice processing and conflict resolution
- Batch operations for efficiency
- Integration with neologism detection

### 4. Comprehensive Testing (`tests/`)
- **test_user_choice_models.py**: Data model testing
- **test_choice_database.py**: Database functionality testing
- **test_user_choice_manager.py**: Service integration testing
- 100+ test cases covering all functionality

## Key Features

### Choice Types
- **TRANSLATE**: Translate the neologism to target language
- **PRESERVE**: Keep the original term unchanged
- **CUSTOM_TRANSLATION**: Use a custom translation
- **SKIP**: Skip processing for this neologism

### Choice Scopes
- **GLOBAL**: Apply to all occurrences everywhere
- **CONTEXTUAL**: Apply only in similar contexts
- **DOCUMENT**: Apply only within current document
- **SESSION**: Apply only within current session

### Context Matching
- Sophisticated similarity algorithms
- Hash-based context comparison
- Configurable similarity thresholds
- Automatic context expansion

### Conflict Resolution
- **HIGHEST_CONFIDENCE**: Choose highest confidence option
- **MOST_RECENT**: Use most recently made choice
- **MOST_SPECIFIC**: Prefer more specific scope
- **USER_DECISION**: Require explicit user resolution

### Session Management
- Document-based session creation
- Automatic session cleanup
- Consistency score tracking
- Session statistics and analytics

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

2. **Initialize Database**:
```python
from services.user_choice_manager import UserChoiceManager

manager = UserChoiceManager(db_path="choices.db")
# Database will be created automatically
```

## Quick Start

### Basic Usage

```python
from services.user_choice_manager import UserChoiceManager, create_session_for_document
from models.user_choice_models import ChoiceType, ChoiceScope

# Initialize manager
manager = UserChoiceManager(db_path="choices.db")

# Create session
session = create_session_for_document(
    manager=manager,
    document_name="philosophical_text.pdf",
    user_id="philosopher_123",
    source_lang="de",
    target_lang="en"
)

# Make a choice (neologism from detection system)
choice = manager.make_choice(
    neologism=detected_neologism,
    choice_type=ChoiceType.TRANSLATE,
    translation_result="being-there",
    session_id=session.session_id,
    choice_scope=ChoiceScope.GLOBAL,
    confidence_level=0.95,
    user_notes="Heidegger's fundamental concept"
)

# Complete session
manager.complete_session(session.session_id)
```

### Integration with Neologism Detection

```python
from services.neologism_detector import NeologismDetector
from services.user_choice_manager import UserChoiceManager

# Initialize both systems
detector = NeologismDetector()
manager = UserChoiceManager()

# Analyze text
analysis = detector.analyze_text(text, "document_id")

# Process each neologism
for neologism in analysis.detected_neologisms:
    # Get recommendation
    recommendation = manager.get_recommendation_for_neologism(
        neologism, session_id
    )
    
    # Make choice based on recommendation
    if recommendation['suggested_action'] == 'apply_existing':
        # Use existing choice
        existing_choice = recommendation['existing_choice']
        print(f"Applying existing choice: {existing_choice['translation']}")
    else:
        # Make new choice
        choice = manager.make_choice(
            neologism=neologism,
            choice_type=ChoiceType.TRANSLATE,
            translation_result="custom_translation",
            session_id=session_id,
            choice_scope=ChoiceScope.CONTEXTUAL
        )
```

### Batch Processing

```python
# Process multiple neologisms efficiently
results = manager.process_neologism_batch(
    neologisms=analysis.detected_neologisms,
    session_id=session.session_id,
    auto_apply_similar=True
)

# Apply choices to analysis
application_results = manager.apply_choices_to_analysis(
    analysis=analysis,
    session_id=session.session_id
)
```

### Terminology Management

```python
# Import terminology
terminology = {
    "Dasein": "being-there",
    "Zeitlichkeit": "temporality",
    "Geworfenheit": "thrownness"
}

imported_count = manager.import_terminology_as_choices(
    terminology_dict=terminology,
    session_id=session.session_id,
    source_language="de",
    target_language="en"
)

# Export choices
export_data = manager.export_session_choices(session.session_id)
with open("terminology.json", "w") as f:
    f.write(export_data)
```

## Advanced Features

### Conflict Resolution

```python
# Check for conflicts
conflicts = manager.get_unresolved_conflicts()

# Resolve conflicts
for conflict in conflicts:
    manager.resolve_conflict(
        conflict_id=conflict.conflict_id,
        resolution_strategy=ConflictResolution.HIGHEST_CONFIDENCE,
        notes="Resolved automatically"
    )
```

### Statistics and Analytics

```python
# Get comprehensive statistics
stats = manager.get_statistics()
print(f"Total choices made: {stats['manager_stats']['total_choices_made']}")
print(f"Sessions created: {stats['manager_stats']['sessions_created']}")
print(f"Conflicts resolved: {stats['manager_stats']['conflicts_resolved']}")

# Validate data integrity
integrity_report = manager.validate_data_integrity()
print(f"Data integrity issues: {integrity_report['total_issues']}")
```

### Session Management

```python
# Get active sessions
active_sessions = manager.get_active_sessions()

# Clean up expired sessions
cleaned_up = manager.cleanup_expired_sessions()
print(f"Cleaned up {cleaned_up} expired sessions")

# Update session status
manager.update_session_status(session_id, SessionStatus.COMPLETED)
```

## Examples

### Complete Integration Example
Run the comprehensive integration example:

```bash
python examples/user_choice_integration_example.py
```

This example demonstrates:
- Basic choice workflow
- Choice reuse and context matching
- Conflict resolution
- Terminology import/export
- Statistics and analytics
- Advanced features

### Neologism Detection Integration
See [`examples/neologism_integration_example.py`](examples/neologism_integration_example.py) for integration with the neologism detection system.

## Database Schema

The system uses SQLite with the following tables:

### Tables
- **user_choices**: Main choice storage
- **choice_sessions**: Session management
- **choice_conflicts**: Conflict tracking
- **translation_preferences**: User preferences
- **translation_contexts**: Context information
- **session_statistics**: Usage analytics

### Indexes
- Optimized for term-based lookups
- Context hash indexing
- Session-based queries
- Conflict detection queries

## Configuration

### Environment Variables
- `CHOICE_DB_PATH`: Database file path (default: "choices.db")
- `SESSION_EXPIRY_HOURS`: Session expiry time (default: 24)
- `AUTO_RESOLVE_CONFLICTS`: Auto-resolve conflicts (default: True)
- `SIMILARITY_THRESHOLD`: Context similarity threshold (default: 0.7)

### Manager Configuration
```python
manager = UserChoiceManager(
    db_path="custom_choices.db",
    auto_resolve_conflicts=False,
    session_expiry_hours=48,
    similarity_threshold=0.8
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_user_choice_models.py -v
python -m pytest tests/test_choice_database.py -v
python -m pytest tests/test_user_choice_manager.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Performance Considerations

### Database Optimization
- Proper indexing for fast lookups
- Connection pooling for concurrent access
- Batch operations for efficiency
- Automatic vacuum and maintenance

### Memory Management
- Lazy loading of large datasets
- Efficient context matching algorithms
- Configurable caching strategies
- Automatic cleanup of expired data

### Scalability
- Designed for large documents (2,000+ pages)
- Handles hundreds of neologisms efficiently
- Batch processing capabilities
- Incremental processing support

## Security Considerations

### Data Protection
- Input validation and sanitization
- SQL injection prevention
- Safe deserialization of imported data
- Secure session management

### Privacy
- User data isolation
- Configurable data retention
- Export/import data control
- Audit trail capabilities

## Troubleshooting

### Common Issues

1. **Database Lock Errors**
   - Ensure proper connection handling
   - Use context managers for database operations
   - Check for long-running transactions

2. **Context Matching Issues**
   - Adjust similarity thresholds
   - Verify context data quality
   - Check hash generation consistency

3. **Performance Issues**
   - Monitor database size and indexes
   - Use batch operations for large datasets
   - Consider database maintenance operations

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

manager = UserChoiceManager(db_path="choices.db")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please:
1. Check the examples and documentation
2. Review the test cases for usage patterns
3. Submit issues with detailed reproduction steps
4. Include relevant configuration and environment details

---

## Technical Details

### Context Matching Algorithm
The system uses a sophisticated context matching algorithm that:
- Generates hash-based context signatures
- Calculates similarity scores using multiple factors
- Supports configurable similarity thresholds
- Provides context expansion capabilities

### Conflict Resolution Strategy
The conflict resolution system:
- Detects conflicts automatically during choice creation
- Categorizes conflicts by type and severity
- Provides multiple resolution strategies
- Maintains audit trails for all resolutions

### Session Management
The session system:
- Tracks document-based workflows
- Manages user sessions with automatic expiry
- Calculates consistency scores
- Provides usage analytics and statistics

This comprehensive system provides robust, scalable, and user-friendly management of translation choices for philosophical text processing.
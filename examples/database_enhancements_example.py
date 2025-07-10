#!/usr/bin/env python3
"""
Example demonstrating the three critical database enhancements in choice_database.py:
1. Alpha Parameter Configuration
2. JSON Encoding Configuration
3. Batch Import Optimization
"""

import sys
import json
import time
import tempfile
import os
import atexit
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, '.')

from database.choice_database import ChoiceDatabase
from models.user_choice_models import (
    UserChoice, ChoiceSession, ChoiceType, ChoiceScope,
    TranslationContext, SessionStatus, ConflictResolution
)

# Global list to track temporary files for cleanup
_temp_files = []

def _create_temp_db(prefix="test_db_"):
    """Create a temporary database file and track it for cleanup."""
    temp_file = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".db", delete=False)
    temp_file.close()
    _temp_files.append(temp_file.name)
    return temp_file.name

def cleanup_temp_files():
    """Clean up all temporary database files."""
    for temp_file in _temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError as e:
            print(f"Warning: Could not clean up {temp_file}: {e}")
    _temp_files.clear()

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)


def demonstrate_alpha_configuration():
    """Demonstrate configurable alpha parameter for learning rate."""
    print("🔧 Alpha Parameter Configuration Example")
    print("=" * 50)
    
    # Example 1: Default alpha (0.1)
    db1 = ChoiceDatabase(_create_temp_db("alpha_default_"))
    print(f"Default alpha: {db1.learning_rate_alpha}")
    
    # Example 2: Custom alpha for faster learning
    db2 = ChoiceDatabase(_create_temp_db("alpha_fast_"), learning_rate_alpha=0.3)
    print(f"Fast learning alpha: {db2.learning_rate_alpha}")
    
    # Example 3: Conservative alpha for stable learning
    db3 = ChoiceDatabase(_create_temp_db("alpha_conservative_"),
                         learning_rate_alpha=0.05)
    print(f"Conservative alpha: {db3.learning_rate_alpha}")
    
    # Example 4: Runtime alpha adjustment
    db3.learning_rate_alpha = 0.08
    print(f"Adjusted alpha: {db3.learning_rate_alpha}")
    
    # Example 5: Demonstrate learning rate impact
    print("\n📊 Learning Rate Impact Simulation:")
    
    # Create a test choice
    context = TranslationContext(
        sentence_context="The concept of Bewusstsein is central to phenomenology.",
        philosophical_domain="Phenomenology"
    )
    
    choice = UserChoice(
        choice_id="alpha_demo_001",
        neologism_term="Bewusstsein",
        choice_type=ChoiceType.TRANSLATE,
        translation_result="consciousness",
        context=context,
        choice_scope=ChoiceScope.CONTEXTUAL,
        success_rate=0.5  # Starting at 50% success rate
    )
    
    # Test with different alpha values
    for alpha_val in [0.01, 0.1, 0.3]:
        db = ChoiceDatabase(_create_temp_db(f"alpha_{alpha_val}_"),
                           learning_rate_alpha=alpha_val)
        db.save_user_choice(choice)
        
        # Simulate successful usage
        db.update_choice_usage(choice.choice_id, success=True)
        updated_choice = db.get_user_choice(choice.choice_id)
        
        print(f"Alpha {alpha_val}: Success rate {choice.success_rate:.3f} → "
              f"{updated_choice.success_rate:.3f}")
    
    print("✅ Alpha configuration demonstrates learning rate flexibility\n")


def demonstrate_json_encoding():
    """Demonstrate configurable JSON encoding for international characters."""
    print("🌍 JSON Encoding Configuration Example")
    print("=" * 50)
    
    # Create choices with international characters
    german_context = TranslationContext(
        sentence_context="Heideggers Konzept der Befindlichkeit ist grundlegend für die Phänomenologie.",
        philosophical_domain="Existenzphilosophie",
        author="Heidegger",
        source_language="Deutsch",
        target_language="English"
    )
    
    choices = [
        UserChoice(
            choice_id="intl_001",
            neologism_term="Befindlichkeit",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="attunement",
            context=german_context,
            choice_scope=ChoiceScope.CONTEXTUAL
        ),
        UserChoice(
            choice_id="intl_002", 
            neologism_term="Geworfenheit",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="thrownness",
            context=german_context,
            choice_scope=ChoiceScope.CONTEXTUAL
        )
    ]
    
    # Example 1: Preserve international characters (default)
    db1 = ChoiceDatabase(_create_temp_db("encoding_preserve_"),
                         ensure_ascii=False)
    for choice in choices:
        db1.save_user_choice(choice)
    
    json_preserved = db1.export_choices_to_json()
    print("🔤 Preserved International Characters:")
    print(f"Contains 'Befindlichkeit': {'Befindlichkeit' in json_preserved}")
    print(f"Contains 'Geworfenheit': {'Geworfenheit' in json_preserved}")
    print(f"Contains 'Phänomenologie': {'Phänomenologie' in json_preserved}")
    
    # Example 2: ASCII-only encoding
    db2 = ChoiceDatabase(_create_temp_db("encoding_ascii_"),
                         ensure_ascii=True)
    for choice in choices:
        db2.save_user_choice(choice)
    
    json_ascii = db2.export_choices_to_json()
    print("\n🔣 ASCII-Only Encoding:")
    print(f"Contains Unicode escapes: {'\\u' in json_ascii}")
    print(f"File size preserved: {len(json_preserved)} bytes")
    print(f"File size ASCII: {len(json_ascii)} bytes")
    
    # Example 3: Runtime encoding adjustment
    db1.ensure_ascii = True
    json_switched = db1.export_choices_to_json()
    print(f"\n🔄 Runtime Switch to ASCII: {'\\u' in json_switched}")
    
    print("✅ JSON encoding handles international characters flexibly\n")


def demonstrate_batch_import():
    """Demonstrate high-performance batch import optimization."""
    print("⚡ Batch Import Optimization Example")
    print("=" * 50)
    
    # Generate large dataset for batch import
    print("📊 Generating large test dataset...")
    
    philosophers = ["Heidegger", "Husserl", "Sartre", "Merleau-Ponty", "Gadamer"]
    domains = ["Phenomenology", "Existentialism", "Hermeneutics", "Ontology"]
    
    test_choices = []
    for i in range(2000):  # Generate 2000 choices
        philosopher = philosophers[i % len(philosophers)]
        domain = domains[i % len(domains)]
        
        context = TranslationContext(
            sentence_context=f"Philosophical context {i} discussing {domain}.",
            philosophical_domain=domain,
            author=philosopher,
            source_language="German",
            target_language="English"
        )
        
        choice = UserChoice(
            choice_id=f"batch_demo_{i:04d}",
            neologism_term=f"Konzept{i}",
            choice_type=ChoiceType.TRANSLATE,
            translation_result=f"concept_{i}",
            context=context,
            choice_scope=ChoiceScope.CONTEXTUAL,
            confidence_level=0.7 + (i % 3) * 0.1,  # Vary confidence
            usage_count=i % 20,
            success_rate=0.8 + (i % 5) * 0.04
        )
        test_choices.append(choice)
    
    print(f"✓ Generated {len(test_choices)} test choices")
    
    # Export to JSON for batch import
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'session_id': 'batch_demo_session',
        'total_choices': len(test_choices),
        'choices': [choice.to_dict() for choice in test_choices]
    }
    
    json_data = json.dumps(export_data, indent=2)
    print(f"✓ Exported to JSON: {len(json_data):,} bytes")
    
    # Example 1: Standard batch size
    print("\n🚀 Standard Batch Size (1000):")
    db1 = ChoiceDatabase(_create_temp_db("batch_standard_"), batch_size=1000)
    
    # Create session to avoid foreign key constraints
    session = ChoiceSession(
        session_id="batch_demo_session",
        session_name="Batch Demo Session",
        status=SessionStatus.ACTIVE,
        conflict_resolution_strategy=ConflictResolution.LATEST_WINS
    )
    db1.save_session(session)
    
    start_time = time.time()
    imported_count = db1.import_choices_from_json(json_data)
    duration = time.time() - start_time
    
    print(f"✓ Imported {imported_count} choices in {duration:.2f}s")
    print(f"✓ Performance: {imported_count/duration:.0f} choices/second")
    
    # Example 2: Large batch size for maximum performance
    print("\n🏎️ Large Batch Size (2000):")
    db2 = ChoiceDatabase(_create_temp_db("batch_large_"), batch_size=2000)
    db2.save_session(session)
    
    start_time = time.time()
    imported_count = db2.import_choices_from_json(json_data)
    duration = time.time() - start_time
    
    print(f"✓ Imported {imported_count} choices in {duration:.2f}s")
    print(f"✓ Performance: {imported_count/duration:.0f} choices/second")
    
    # Example 3: Small batch size for memory-constrained environments
    print("\n💾 Small Batch Size (500):")
    db3 = ChoiceDatabase(_create_temp_db("batch_small_"), batch_size=500)
    db3.save_session(session)
    
    start_time = time.time()
    imported_count = db3.import_choices_from_json(json_data)
    duration = time.time() - start_time
    
    print(f"✓ Imported {imported_count} choices in {duration:.2f}s")
    print(f"✓ Performance: {imported_count/duration:.0f} choices/second")
    
    # Verify data integrity
    print("\n🔍 Data Integrity Verification:")
    stats = db1.get_database_statistics()
    print(f"✓ Total choices in database: {stats['total_choices']}")
    
    # Test specific choice retrieval
    sample_choice = db1.get_user_choice("batch_demo_1000")
    if sample_choice:
        print(f"✓ Sample choice retrieved: {sample_choice.neologism_term}")
        print(f"✓ Confidence level: {sample_choice.confidence_level}")
    
    print("✅ Batch import optimization delivers high performance\n")


def demonstrate_combined_features():
    """Demonstrate all three enhancements working together."""
    print("🔄 Combined Features Example")
    print("=" * 50)
    
    # Create database with all custom configurations
    db = ChoiceDatabase(
        _create_temp_db("combined_"),
        learning_rate_alpha=0.15,      # Custom learning rate
        ensure_ascii=False,            # Preserve international characters
        batch_size=1500               # Optimized batch size
    )
    
    print(f"✓ Database configured with:")
    print(f"  - Learning rate alpha: {db.learning_rate_alpha}")
    print(f"  - Preserve international chars: {not db.ensure_ascii}")
    print(f"  - Batch size: {db.batch_size}")
    
    # Create session
    session = ChoiceSession(
        session_id="combined_demo",
        session_name="Combined Features Demo",
        status=SessionStatus.ACTIVE,
        conflict_resolution_strategy=ConflictResolution.LATEST_WINS
    )
    db.save_session(session)
    
    # Create choices with international characters
    choices = []
    terms = ["Sein", "Dasein", "Bewußtsein", "Möglichkeit", "Wirklichkeit"]
    
    for i, term in enumerate(terms):
        context = TranslationContext(
            sentence_context=f"Der Begriff {term} ist zentral für die Philosophie.",
            philosophical_domain="Ontologie",
            author="Heidegger",
            source_language="Deutsch",
            target_language="English"
        )
        
        choice = UserChoice(
            choice_id=f"combined_{i:03d}",
            neologism_term=term,
            choice_type=ChoiceType.TRANSLATE,
            translation_result=f"translation_{i}",
            context=context,
            choice_scope=ChoiceScope.CONTEXTUAL,
            success_rate=0.6
        )
        choices.append(choice)
    
    # Export and import using all features
    for choice in choices:
        db.save_user_choice(choice)
    
    # Test JSON export with international characters
    json_data = db.export_choices_to_json()
    print(f"✓ JSON export preserves 'Bewußtsein': {'Bewußtsein' in json_data}")
    
    # Test learning rate by updating usage
    db.update_choice_usage("combined_001", success=True)
    updated_choice = db.get_user_choice("combined_001")
    print(f"✓ Learning rate applied: {choices[1].success_rate:.3f} → {updated_choice.success_rate:.3f}")
    
    # Test batch operations
    print(f"✓ Batch size configuration: {db.batch_size} choices per batch")
    
    print("✅ All three enhancements work together seamlessly\n")


def main():
    """Run all demonstration examples."""
    print("🚀 Database Enhancements Demonstration")
    print("=" * 60)
    print()
    
    demonstrate_alpha_configuration()
    demonstrate_json_encoding()
    demonstrate_batch_import()
    demonstrate_combined_features()
    
    print("🎉 All enhancements demonstrated successfully!")
    print("✅ Alpha Parameter Configuration: Flexible learning rates")
    print("✅ JSON Encoding Configuration: International character support")
    print("✅ Batch Import Optimization: High-performance bulk operations")
    print("✅ Combined Features: Seamless integration")


if __name__ == "__main__":
    main()
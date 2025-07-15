"""
Example integration of the User Choice Management System with neologism detection.

This example demonstrates how to:
1. Set up the User Choice Management System
2. Integrate it with the neologism detection engine
3. Process philosophical texts with user-controlled translation choices
4. Handle conflicts and maintain consistency
5. Export and import terminology preferences

Usage:
    python examples/user_choice_integration_example.py
"""

import os

from models.user_choice_models import (
    ChoiceScope,
    ChoiceType,
    ConflictResolution,
)
from services.neologism_detector import NeologismDetector
from services.user_choice_manager import UserChoiceManager, create_session_for_document


def setup_example_environment():
    """Set up the example environment with sample data."""
    print("Setting up User Choice Management System example...")

    # Create a user choice manager
    manager = UserChoiceManager(
        db_path="example_choices.db",
        auto_resolve_conflicts=True,
        session_expiry_hours=24,
    )

    # Create a neologism detector
    detector = NeologismDetector(
        terminology_path="config/klages_terminology.json", philosophical_threshold=0.3
    )

    print("✓ User Choice Manager and Neologism Detector initialized")
    return manager, detector


def demonstrate_basic_choice_workflow(manager, detector):
    """Demonstrate basic choice workflow."""
    print("\n=== Basic Choice Workflow ===")

    # Sample philosophical text
    text = """
    Das Dasein ist ein zentraler Begriff in Heideggers Fundamentalontologie.
    Die Seinsvergessenheit charakterisiert die moderne Epoche der Technik.
    Durch die Destruktion der Ontologie soll das ursprüngliche Seinsverständnis
    freigelegt werden. Die Zeitlichkeit konstituiert das Sein des Daseins.
    """

    # Create a session for this document
    session = create_session_for_document(
        manager=manager,
        document_name="Heidegger_Sein_und_Zeit.pdf",
        user_id="philosophy_student_123",
        source_lang="de",
        target_lang="en",
    )

    print(f"✓ Created session: {session.session_name}")
    print(f"  Session ID: {session.session_id}")

    # Analyze the text for neologisms
    analysis = detector.analyze_text(text, "heidegger_sample")
    print(f"✓ Detected {analysis.total_detections} neologisms")

    # Process each neologism and make choices
    choices_made = []
    for neologism in analysis.detected_neologisms:
        print(f"\n--- Processing: {neologism.term} ---")
        print(f"Confidence: {neologism.confidence:.2f}")
        print(f"Context: {neologism.sentence_context[:60]}...")

        # Get recommendation
        recommendation = manager.get_recommendation_for_neologism(
            neologism, session.session_id
        )

        print(f"Recommendation: {recommendation['suggested_action']}")

        # Make choices based on the term
        if neologism.term == "Dasein":
            choice = manager.make_choice(
                neologism=neologism,
                choice_type=ChoiceType.TRANSLATE,
                translation_result="being-there",
                session_id=session.session_id,
                choice_scope=ChoiceScope.GLOBAL,
                confidence_level=0.95,
                user_notes="Heidegger's fundamental concept of human existence",
            )
            print("✓ Made choice: TRANSLATE → 'being-there'")

        elif neologism.term == "Seinsvergessenheit":
            choice = manager.make_choice(
                neologism=neologism,
                choice_type=ChoiceType.TRANSLATE,
                translation_result="forgetfulness-of-being",
                session_id=session.session_id,
                choice_scope=ChoiceScope.CONTEXTUAL,
                confidence_level=0.9,
                user_notes="Heidegger's critique of Western metaphysics",
            )
            print("✓ Made choice: TRANSLATE → 'forgetfulness-of-being'")

        elif neologism.term == "Zeitlichkeit":
            choice = manager.make_choice(
                neologism=neologism,
                choice_type=ChoiceType.TRANSLATE,
                translation_result="temporality",
                session_id=session.session_id,
                choice_scope=ChoiceScope.CONTEXTUAL,
                confidence_level=0.85,
                user_notes="The temporal structure of Dasein",
            )
            print("✓ Made choice: TRANSLATE → 'temporality'")

        else:
            choice = manager.make_choice(
                neologism=neologism,
                choice_type=ChoiceType.PRESERVE,
                translation_result="",
                session_id=session.session_id,
                choice_scope=ChoiceScope.DOCUMENT,
                confidence_level=0.7,
                user_notes="Preserve original term for philosophical precision",
            )
            print("✓ Made choice: PRESERVE original term")

        choices_made.append(choice)

    print(f"\n✓ Made {len(choices_made)} choices total")

    # Complete the session
    manager.complete_session(session.session_id)
    print(
        f"✓ Session completed with consistency score: {session.consistency_score:.2f}"
    )

    return session, choices_made


def demonstrate_choice_reuse(manager, detector):
    """Demonstrate choice reuse and context matching."""
    print("\n=== Choice Reuse and Context Matching ===")

    # New text with previously seen terms
    new_text = """
    In der Analyse des Daseins zeigt sich die fundamentale Struktur der Zeitlichkeit.
    Das Dasein ist immer schon in die Welt hinein geworfene Existenz.
    Die Seinsvergessenheit der Moderne verstellt den Zugang zum ursprünglichen Sein.
    """

    # Create new session
    new_session = create_session_for_document(
        manager=manager,
        document_name="Heidegger_Metaphysik.pdf",
        user_id="philosophy_student_123",
        source_lang="de",
        target_lang="en",
    )

    # Analyze new text
    new_analysis = detector.analyze_text(new_text, "heidegger_sample_2")

    # Process with existing choices
    results = manager.process_neologism_batch(
        neologisms=new_analysis.detected_neologisms,
        session_id=new_session.session_id,
        auto_apply_similar=True,
    )

    print(f"✓ Processed {len(results)} neologisms")

    applied_count = 0
    for neologism, suggested_choice in results:
        if suggested_choice:
            print(
                f"  → {neologism.term}: Auto-applied '{suggested_choice.translation_result}'"
            )
            applied_count += 1
        else:
            print(f"  → {neologism.term}: No existing choice found")

    print(f"✓ Auto-applied {applied_count} existing choices")

    # Apply choices to analysis
    application_results = manager.apply_choices_to_analysis(
        analysis=new_analysis, session_id=new_session.session_id
    )

    print("✓ Analysis results:")
    print(f"  Total neologisms: {application_results['total_neologisms']}")
    print(f"  Choices found: {application_results['choices_found']}")
    print(f"  Choices applied: {application_results['choices_applied']}")
    print(f"  New choices needed: {application_results['new_choices_needed']}")

    return new_session, application_results


def demonstrate_conflict_resolution(manager, detector):
    """Demonstrate conflict detection and resolution."""
    print("\n=== Conflict Detection and Resolution ===")

    # Create a scenario with conflicting choices
    text = "Das Dasein zeigt sich in verschiedenen Weisen des Seins."

    # Create session
    conflict_session = create_session_for_document(
        manager=manager,
        document_name="Heidegger_Conflicting_Interpretation.pdf",
        user_id="philosophy_student_456",  # Different user
        source_lang="de",
        target_lang="en",
    )

    # Analyze text
    analysis = detector.analyze_text(text, "conflict_test")

    # Make first choice
    dasein_neologism = None
    for neologism in analysis.detected_neologisms:
        if neologism.term == "Dasein":
            dasein_neologism = neologism
            break

    if dasein_neologism:
        # First choice: different translation
        choice1 = manager.make_choice(
            neologism=dasein_neologism,
            choice_type=ChoiceType.TRANSLATE,
            translation_result="existence",  # Different from previous "being-there"
            session_id=conflict_session.session_id,
            choice_scope=ChoiceScope.CONTEXTUAL,
            confidence_level=0.8,
            user_notes="Alternative translation emphasizing existence",
        )
        print("✓ Made conflicting choice: TRANSLATE → 'existence'")

        # Second choice: preserve instead of translate
        choice2 = manager.make_choice(
            neologism=dasein_neologism,
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            session_id=conflict_session.session_id,
            choice_scope=ChoiceScope.DOCUMENT,
            confidence_level=0.9,
            user_notes="Preserve for philosophical precision",
        )
        print("✓ Made conflicting choice: PRESERVE")

        # Check for unresolved conflicts
        conflicts = manager.get_unresolved_conflicts()
        print(f"✓ Found {len(conflicts)} unresolved conflicts")

        for conflict in conflicts:
            print(f"  Conflict: {conflict.neologism_term}")
            print(f"    Type: {conflict.conflict_type}")
            print(f"    Severity: {conflict.severity:.2f}")
            print(f"    Context similarity: {conflict.context_similarity:.2f}")

            # Manually resolve conflict
            manager.resolve_conflict(
                conflict_id=conflict.conflict_id,
                resolution_strategy=ConflictResolution.HIGHEST_CONFIDENCE,
                notes="Resolved by choosing highest confidence option",
            )
            print("    ✓ Resolved using highest confidence strategy")

    return conflict_session


def demonstrate_terminology_import_export(manager):
    """Demonstrate terminology import and export."""
    print("\n=== Terminology Import/Export ===")

    # Create custom terminology
    custom_terminology = {
        "Geworfenheit": "thrownness",
        "Zuhandenheit": "readiness-to-hand",
        "Vorhandenheit": "presence-at-hand",
        "Sein-zum-Tode": "being-toward-death",
        "Man": "the-they",
        "Gerede": "idle-talk",
        "Neugier": "curiosity",
        "Zweideutigkeit": "ambiguity",
    }

    # Create session for import
    import_session = create_session_for_document(
        manager=manager,
        document_name="Custom_Terminology.json",
        user_id="terminology_curator",
        source_lang="de",
        target_lang="en",
    )

    # Import terminology as choices
    imported_count = manager.import_terminology_as_choices(
        terminology_dict=custom_terminology,
        session_id=import_session.session_id,
        source_language="de",
        target_language="en",
    )

    print(f"✓ Imported {imported_count} terminology entries as choices")

    # Export session choices
    export_data = manager.export_session_choices(import_session.session_id)

    # Save to file
    export_file = "exported_terminology.json"
    with open(export_file, "w", encoding="utf-8") as f:
        f.write(export_data)

    print(f"✓ Exported choices to {export_file}")

    # Create new session and import
    new_import_session = create_session_for_document(
        manager=manager,
        document_name="Imported_Terminology.json",
        user_id="another_user",
        source_lang="de",
        target_lang="en",
    )

    # Import from file
    with open(export_file, encoding="utf-8") as f:
        import_data = f.read()

    reimported_count = manager.import_choices(
        import_data, new_import_session.session_id
    )
    print(f"✓ Re-imported {reimported_count} choices")

    # Cleanup
    if os.path.exists(export_file):
        os.remove(export_file)

    return import_session, new_import_session


def demonstrate_statistics_and_analytics(manager):
    """Demonstrate statistics and analytics."""
    print("\n=== Statistics and Analytics ===")

    # Get comprehensive statistics
    stats = manager.get_statistics()

    print("Manager Statistics:")
    for key, value in stats["manager_stats"].items():
        print(f"  {key}: {value}")

    print("\nDatabase Statistics:")
    for key, value in stats["database_stats"].items():
        if key == "choice_type_distribution":
            print(f"  {key}:")
            for choice_type, count in value.items():
                print(f"    {choice_type}: {count}")
        else:
            print(f"  {key}: {value}")

    print(f"\nActive Sessions: {stats['active_sessions']}")
    print(f"Session Expiry Hours: {stats['session_expiry_hours']}")
    print(f"Auto Resolve Conflicts: {stats['auto_resolve_conflicts']}")

    # Get active sessions
    active_sessions = manager.get_active_sessions()
    print("\nActive Sessions Details:")
    for session in active_sessions:
        print(f"  Session: {session.session_name}")
        print(f"    ID: {session.session_id}")
        print(f"    User: {session.user_id}")
        print(f"    Document: {session.document_name}")
        print(f"    Total Choices: {session.total_choices}")
        print(f"    Consistency Score: {session.consistency_score:.2f}")
        print(f"    Created: {session.created_at}")

    # Validate data integrity
    integrity_report = manager.validate_data_integrity()
    print("\nData Integrity Report:")
    print(f"  Total Issues: {integrity_report['total_issues']}")
    print(f"  Recommendations: {len(integrity_report['recommendations'])}")
    for rec in integrity_report["recommendations"]:
        print(f"    - {rec}")


def demonstrate_advanced_features(manager, detector):
    """Demonstrate advanced features."""
    print("\n=== Advanced Features ===")

    # Complex philosophical text
    complex_text = """
    Die ontologische Differenz zwischen Sein und Seiendem bildet den Grundzug
    der Heideggerschen Fundamentalontologie. Das Dasein als Sein-in-der-Welt
    ist charakterisiert durch Geworfenheit, Entwurf und Verfallenheit.
    Die Destruktion der traditionellen Ontologie soll das ursprüngliche
    Seinsverständnis freilegen, das durch die Seinsvergessenheit der
    abendländischen Metaphysik verschüttet wurde.
    """

    # Create session
    advanced_session = create_session_for_document(
        manager=manager,
        document_name="Advanced_Heidegger_Analysis.pdf",
        user_id="advanced_philosopher",
        source_lang="de",
        target_lang="en",
    )

    # Analyze complex text
    complex_analysis = detector.analyze_text(complex_text, "complex_analysis")

    # Process with recommendations
    recommendations = []
    for neologism in complex_analysis.detected_neologisms:
        rec = manager.get_recommendation_for_neologism(
            neologism, advanced_session.session_id
        )
        recommendations.append(rec)

        print(f"\nNeologism: {neologism.term}")
        print(f"  Confidence: {neologism.confidence:.2f}")
        print(f"  Type: {neologism.neologism_type.value}")
        print(f"  Recommendation: {rec['suggested_action']}")
        print(f"  Reasons: {', '.join(rec['reasons'])}")

        # Make sophisticated choices based on recommendations
        if rec["suggested_action"] == "apply_existing" and rec["existing_choice"]:
            print(
                f"  → Applying existing choice: {rec['existing_choice']['translation']}"
            )
        elif rec["suggested_action"] == "consider_similar" and rec["similar_choices"]:
            print(f"  → Similar choices available: {len(rec['similar_choices'])}")
        else:
            # Make new choice based on neologism characteristics
            if neologism.confidence > 0.8:
                choice_type = ChoiceType.TRANSLATE
                translation = f"[{neologism.term.lower()}]"  # Placeholder translation
            else:
                choice_type = ChoiceType.PRESERVE
                translation = ""

            choice = manager.make_choice(
                neologism=neologism,
                choice_type=choice_type,
                translation_result=translation,
                session_id=advanced_session.session_id,
                choice_scope=ChoiceScope.CONTEXTUAL,
                confidence_level=neologism.confidence,
                user_notes=f"Auto-generated based on {neologism.confidence:.2f} confidence",
            )
            print(f"  → Made new choice: {choice_type.value}")

    # Complete session
    manager.complete_session(advanced_session.session_id)

    return advanced_session, recommendations


def cleanup_example():
    """Clean up example files."""
    print("\n=== Cleanup ===")

    files_to_remove = ["example_choices.db", "exported_terminology.json"]

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Removed {file_path}")


def main():
    """Main example execution."""
    print("=" * 60)
    print("User Choice Management System Integration Example")
    print("=" * 60)

    try:
        # Setup
        manager, detector = setup_example_environment()

        # Demonstrate features
        session1, choices1 = demonstrate_basic_choice_workflow(manager, detector)
        session2, results2 = demonstrate_choice_reuse(manager, detector)
        session3 = demonstrate_conflict_resolution(manager, detector)
        session4, session5 = demonstrate_terminology_import_export(manager)
        demonstrate_statistics_and_analytics(manager)
        session6, recommendations = demonstrate_advanced_features(manager, detector)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(
            f"✓ Created {len([session1, session2, session3, session4, session5, session6])} sessions"
        )
        print("✓ Processed multiple philosophical texts")
        print("✓ Demonstrated choice reuse and context matching")
        print("✓ Handled conflicts and resolutions")
        print("✓ Imported/exported terminology")
        print("✓ Showed advanced recommendation system")

        # Final statistics
        final_stats = manager.get_statistics()
        print("\nFinal Statistics:")
        print(
            f"  Total Choices Made: {final_stats['manager_stats']['total_choices_made']}"
        )
        print(f"  Sessions Created: {final_stats['manager_stats']['sessions_created']}")
        print(
            f"  Conflicts Resolved: {final_stats['manager_stats']['conflicts_resolved']}"
        )
        print(
            f"  Database Size: {final_stats['database_stats']['db_size_bytes']} bytes"
        )

        print(
            "\n✓ User Choice Management System integration example completed successfully!"
        )

    except Exception as e:
        print(f"\n❌ Error during example execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        cleanup_example()


if __name__ == "__main__":
    main()

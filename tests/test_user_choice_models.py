"""Tests for user choice models."""

import json

import pytest

from models.user_choice_models import (
    ChoiceConflict,
    ChoiceScope,
    ChoiceSession,
    ChoiceType,
    ConflictResolution,
    SessionStatus,
    TranslationContext,
    UserChoice,
    create_choice_id,
    create_conflict_id,
    create_session_id,
    detect_choice_conflicts,
    filter_choices_by_context,
    find_best_matching_choice,
)


class TestTranslationContext:
    """Test TranslationContext functionality."""

    def test_context_creation(self):
        """Test basic context creation."""
        context = TranslationContext(
            sentence_context="This is a test sentence with Dasein.",
            semantic_field="existentialism",
            philosophical_domain="philosophy_of_mind",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        assert context.sentence_context == "This is a test sentence with Dasein."
        assert context.semantic_field == "existentialism"
        assert context.philosophical_domain == "philosophy_of_mind"
        assert context.author == "Heidegger"
        assert context.source_language == "de"
        assert context.target_language == "en"

    def test_context_hash_generation(self):
        """Test context hash generation."""
        context1 = TranslationContext(
            semantic_field="existentialism",
            philosophical_domain="philosophy_of_mind",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        context2 = TranslationContext(
            semantic_field="existentialism",
            philosophical_domain="philosophy_of_mind",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        # Same contexts should have same hash
        assert context1.generate_context_hash() == context2.generate_context_hash()

        # Different contexts should have different hashes
        context3 = TranslationContext(
            semantic_field="metaphysics",
            philosophical_domain="ontology",
            author="Kant",
            source_language="de",
            target_language="en",
        )

        assert context1.generate_context_hash() != context3.generate_context_hash()

    def test_context_similarity_calculation(self):
        """Test context similarity calculation."""
        context1 = TranslationContext(
            semantic_field="existentialism",
            philosophical_domain="philosophy_of_mind",
            author="Heidegger",
            source_language="de",
            target_language="en",
            surrounding_terms=["sein", "dasein", "existence"],
            related_concepts=["being", "time", "anxiety"],
        )

        # Very similar context
        context2 = TranslationContext(
            semantic_field="existentialism",
            philosophical_domain="philosophy_of_mind",
            author="Heidegger",
            source_language="de",
            target_language="en",
            surrounding_terms=["sein", "dasein", "temporal"],
            related_concepts=["being", "time", "finitude"],
        )

        similarity = context1.calculate_similarity(context2)
        assert similarity > 0.8  # Should be very similar

        # Different context
        context3 = TranslationContext(
            semantic_field="ethics",
            philosophical_domain="moral_philosophy",
            author="Kant",
            source_language="de",
            target_language="en",
            surrounding_terms=["duty", "categorical", "imperative"],
            related_concepts=["moral", "duty", "reason"],
        )

        similarity = context1.calculate_similarity(context3)
        assert similarity < 0.5  # Should be different

    def test_context_to_dict(self):
        """Test context serialization."""
        context = TranslationContext(
            sentence_context="Test sentence",
            semantic_field="test_field",
            philosophical_domain="test_domain",
            author="Test Author",
            source_language="de",
            target_language="en",
            surrounding_terms=["term1", "term2"],
            related_concepts=["concept1", "concept2"],
        )

        context_dict = context.to_dict()

        assert context_dict["sentence_context"] == "Test sentence"
        assert context_dict["semantic_field"] == "test_field"
        assert context_dict["philosophical_domain"] == "test_domain"
        assert context_dict["author"] == "Test Author"
        assert context_dict["source_language"] == "de"
        assert context_dict["target_language"] == "en"
        assert context_dict["surrounding_terms"] == ["term1", "term2"]
        assert context_dict["related_concepts"] == ["concept1", "concept2"]
        assert "context_hash" in context_dict


class TestUserChoice:
    """Test UserChoice functionality."""

    def test_user_choice_creation(self):
        """Test basic user choice creation."""
        context = TranslationContext(
            semantic_field="existentialism",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        choice = UserChoice(
            choice_id="test_choice_1",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
            choice_scope=ChoiceScope.CONTEXTUAL,
            confidence_level=0.9,
            user_notes="Heidegger's fundamental concept",
        )

        assert choice.choice_id == "test_choice_1"
        assert choice.neologism_term == "Dasein"
        assert choice.choice_type == ChoiceType.TRANSLATE
        assert choice.translation_result == "being-there"
        assert choice.choice_scope == ChoiceScope.CONTEXTUAL
        assert choice.confidence_level == 0.9
        assert choice.user_notes == "Heidegger's fundamental concept"

    def test_usage_stats_update(self):
        """Test usage statistics updates."""
        context = TranslationContext()
        choice = UserChoice(
            choice_id="test_choice_1",
            neologism_term="test_term",
            choice_type=ChoiceType.TRANSLATE,
            context=context,
        )

        initial_count = choice.usage_count
        initial_rate = choice.success_rate

        # Update with success
        choice.update_usage_stats(success=True)

        assert choice.usage_count == initial_count + 1
        assert choice.success_rate >= initial_rate  # Should maintain or improve
        assert choice.last_used_at is not None

        # Update with failure
        choice.update_usage_stats(success=False)

        assert choice.usage_count == initial_count + 2
        assert choice.success_rate < initial_rate  # Should decrease

    def test_choice_applicability(self):
        """Test choice applicability to contexts."""
        context1 = TranslationContext(
            semantic_field="existentialism",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        # Global scope choice
        global_choice = UserChoice(
            choice_id="global_choice",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            context=context1,
            choice_scope=ChoiceScope.GLOBAL,
        )

        # Different context
        context2 = TranslationContext(
            semantic_field="ethics",
            author="Kant",
            source_language="de",
            target_language="en",
        )

        # Global choice should apply everywhere
        assert global_choice.is_applicable_to_context(context2)

        # Contextual choice
        contextual_choice = UserChoice(
            choice_id="contextual_choice",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            context=context1,
            choice_scope=ChoiceScope.CONTEXTUAL,
        )

        # Should apply to similar context
        similar_context = TranslationContext(
            semantic_field="existentialism",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        assert contextual_choice.is_applicable_to_context(similar_context)

        # Should not apply to very different context
        assert not contextual_choice.is_applicable_to_context(context2)

    def test_choice_serialization(self):
        """Test choice serialization."""
        context = TranslationContext(semantic_field="test_field", author="Test Author")

        choice = UserChoice(
            choice_id="test_choice",
            neologism_term="test_term",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="test_translation",
            context=context,
            export_tags={"tag1", "tag2"},
        )

        # Test to_dict
        choice_dict = choice.to_dict()
        assert choice_dict["choice_id"] == "test_choice"
        assert choice_dict["neologism_term"] == "test_term"
        assert choice_dict["choice_type"] == "translate"
        assert choice_dict["translation_result"] == "test_translation"
        assert "context" in choice_dict
        assert set(choice_dict["export_tags"]) == {"tag1", "tag2"}

        # Test to_json
        json_str = choice.to_json()
        parsed = json.loads(json_str)
        assert parsed["choice_id"] == "test_choice"


class TestChoiceSession:
    """Test ChoiceSession functionality."""

    def test_session_creation(self):
        """Test basic session creation."""
        session = ChoiceSession(
            session_id="test_session_1",
            session_name="Test Session",
            document_name="test_document.pdf",
            user_id="user123",
            source_language="de",
            target_language="en",
        )

        assert session.session_id == "test_session_1"
        assert session.session_name == "Test Session"
        assert session.document_name == "test_document.pdf"
        assert session.user_id == "user123"
        assert session.source_language == "de"
        assert session.target_language == "en"
        assert session.status == SessionStatus.ACTIVE

    def test_session_stats_update(self):
        """Test session statistics updates."""
        session = ChoiceSession(session_id="test_session", session_name="Test Session")

        context = TranslationContext()

        # Add different types of choices
        choice1 = UserChoice(
            choice_id="choice1",
            neologism_term="term1",
            choice_type=ChoiceType.TRANSLATE,
            context=context,
            confidence_level=0.8,
        )

        choice2 = UserChoice(
            choice_id="choice2",
            neologism_term="term2",
            choice_type=ChoiceType.PRESERVE,
            context=context,
            confidence_level=0.9,
        )

        choice3 = UserChoice(
            choice_id="choice3",
            neologism_term="term3",
            choice_type=ChoiceType.CUSTOM_TRANSLATION,
            context=context,
            confidence_level=0.7,
        )

        # Add choices to session
        session.add_choice_stats(choice1)
        session.add_choice_stats(choice2)
        session.add_choice_stats(choice3)

        assert session.total_choices == 3
        assert session.translate_count == 1
        assert session.preserve_count == 1
        assert session.custom_count == 1
        assert session.skip_count == 0

        # Check average confidence
        expected_avg = (0.8 + 0.9 + 0.7) / 3
        assert abs(session.average_confidence - expected_avg) < 0.01

    def test_session_completion(self):
        """Test session completion."""
        session = ChoiceSession(session_id="test_session", session_name="Test Session")

        assert session.status == SessionStatus.ACTIVE
        assert session.completed_at is None

        session.complete_session()

        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None

    def test_consistency_calculation(self):
        """Test consistency score calculation."""
        session = ChoiceSession(session_id="test_session", session_name="Test Session")

        context = TranslationContext()

        # Consistent choices for same term
        choice1 = UserChoice(
            choice_id="choice1",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
        )

        choice2 = UserChoice(
            choice_id="choice2",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
        )

        # Inconsistent choice for same term
        choice3 = UserChoice(
            choice_id="choice3",
            neologism_term="Dasein",
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            context=context,
        )

        # Test consistent choices
        consistency = session.calculate_consistency_score([choice1, choice2])
        assert consistency == 1.0

        # Test inconsistent choices
        consistency = session.calculate_consistency_score([choice1, choice2, choice3])
        assert consistency < 1.0


class TestChoiceConflict:
    """Test ChoiceConflict functionality."""

    def test_conflict_creation(self):
        """Test conflict creation and analysis."""
        context = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        choice_a = UserChoice(
            choice_id="choice_a",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
        )

        choice_b = UserChoice(
            choice_id="choice_b",
            neologism_term="Dasein",
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            context=context,
        )

        conflict = ChoiceConflict(
            conflict_id="conflict_1",
            neologism_term="Dasein",
            choice_a=choice_a,
            choice_b=choice_b,
        )

        assert conflict.conflict_id == "conflict_1"
        assert conflict.neologism_term == "Dasein"
        assert conflict.choice_a == choice_a
        assert conflict.choice_b == choice_b

    def test_conflict_analysis(self):
        """Test conflict analysis."""
        # Similar contexts
        context1 = TranslationContext(
            semantic_field="existentialism",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        context2 = TranslationContext(
            semantic_field="existentialism",
            author="Heidegger",
            source_language="de",
            target_language="en",
        )

        choice_a = UserChoice(
            choice_id="choice_a",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context1,
        )

        choice_b = UserChoice(
            choice_id="choice_b",
            neologism_term="Dasein",
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            context=context2,
        )

        conflict = ChoiceConflict(
            conflict_id="conflict_1",
            neologism_term="Dasein",
            choice_a=choice_a,
            choice_b=choice_b,
        )

        conflict.analyze_conflict()

        assert conflict.context_similarity > 0.8  # Should be high for similar contexts
        assert conflict.conflict_type == "choice_type_mismatch"
        assert conflict.severity > 0.8  # High severity for similar contexts

    def test_conflict_resolution(self):
        """Test conflict resolution."""
        context = TranslationContext()

        # Create choices with different timestamps
        choice_a = UserChoice(
            choice_id="choice_a",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
            confidence_level=0.8,
            created_at="2023-01-01T10:00:00",
        )

        choice_b = UserChoice(
            choice_id="choice_b",
            neologism_term="Dasein",
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            context=context,
            confidence_level=0.9,
            created_at="2023-01-01T11:00:00",
        )

        conflict = ChoiceConflict(
            conflict_id="conflict_1",
            neologism_term="Dasein",
            choice_a=choice_a,
            choice_b=choice_b,
        )

        # Test latest wins resolution
        resolved_id = conflict.resolve_conflict(ConflictResolution.LATEST_WINS)
        assert resolved_id == "choice_b"  # Later timestamp

        # Test highest confidence resolution
        conflict.resolved_choice_id = None  # Reset
        resolved_id = conflict.resolve_conflict(ConflictResolution.HIGHEST_CONFIDENCE)
        assert resolved_id == "choice_b"  # Higher confidence

        # Test context-specific resolution
        conflict.resolved_choice_id = None  # Reset
        resolved_id = conflict.resolve_conflict(ConflictResolution.CONTEXT_SPECIFIC)
        assert resolved_id is None  # Should keep both


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_choice_id(self):
        """Test choice ID creation."""
        choice_id = create_choice_id("Dasein", "context_hash_123")
        assert isinstance(choice_id, str)
        assert len(choice_id) == 16  # Should be 16 characters

        # Same inputs should produce same ID
        choice_id2 = create_choice_id("Dasein", "context_hash_123")
        # Note: IDs include timestamp, so they will be different
        assert isinstance(choice_id2, str)
        assert len(choice_id2) == 16

    def test_create_session_id(self):
        """Test session ID creation."""
        session_id = create_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) == 16

    def test_create_conflict_id(self):
        """Test conflict ID creation."""
        conflict_id = create_conflict_id("choice_a", "choice_b")
        assert isinstance(conflict_id, str)
        assert len(conflict_id) == 16

        # Same inputs should produce same ID
        conflict_id2 = create_conflict_id("choice_a", "choice_b")
        assert conflict_id == conflict_id2

    def test_filter_choices_by_context(self):
        """Test filtering choices by context."""
        context1 = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        context2 = TranslationContext(semantic_field="ethics", author="Kant")

        target_context = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        # Create choices with different contexts
        choice1 = UserChoice(
            choice_id="choice1",
            neologism_term="term1",
            choice_type=ChoiceType.TRANSLATE,
            context=context1,
            choice_scope=ChoiceScope.CONTEXTUAL,
        )

        choice2 = UserChoice(
            choice_id="choice2",
            neologism_term="term2",
            choice_type=ChoiceType.TRANSLATE,
            context=context2,
            choice_scope=ChoiceScope.CONTEXTUAL,
        )

        choice3 = UserChoice(
            choice_id="choice3",
            neologism_term="term3",
            choice_type=ChoiceType.TRANSLATE,
            context=context1,
            choice_scope=ChoiceScope.GLOBAL,
        )

        choices = [choice1, choice2, choice3]

        # Filter by target context
        filtered = filter_choices_by_context(choices, target_context)

        # Should include choice1 (similar context) and choice3 (global scope)
        assert len(filtered) == 2
        assert choice1 in filtered
        assert choice3 in filtered
        assert choice2 not in filtered

    def test_find_best_matching_choice(self):
        """Test finding best matching choice."""
        target_context = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        # Create choices with different similarity and success rates
        context1 = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        context2 = TranslationContext(semantic_field="existentialism", author="Sartre")

        choice1 = UserChoice(
            choice_id="choice1",
            neologism_term="term1",
            choice_type=ChoiceType.TRANSLATE,
            context=context1,
            choice_scope=ChoiceScope.CONTEXTUAL,
            success_rate=0.8,
        )

        choice2 = UserChoice(
            choice_id="choice2",
            neologism_term="term2",
            choice_type=ChoiceType.TRANSLATE,
            context=context2,
            choice_scope=ChoiceScope.CONTEXTUAL,
            success_rate=0.9,
        )

        choices = [choice1, choice2]

        # Find best match
        best_match = find_best_matching_choice(choices, target_context)

        # Should prefer choice1 due to better context similarity
        assert best_match == choice1

    def test_detect_choice_conflicts(self):
        """Test conflict detection."""
        context = TranslationContext(
            semantic_field="existentialism", author="Heidegger"
        )

        # Create conflicting choices for same term
        choice1 = UserChoice(
            choice_id="choice1",
            neologism_term="Dasein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being-there",
            context=context,
        )

        choice2 = UserChoice(
            choice_id="choice2",
            neologism_term="Dasein",
            choice_type=ChoiceType.PRESERVE,
            translation_result="",
            context=context,
        )

        # Different term - should not conflict
        choice3 = UserChoice(
            choice_id="choice3",
            neologism_term="Sein",
            choice_type=ChoiceType.TRANSLATE,
            translation_result="being",
            context=context,
        )

        choices = [choice1, choice2, choice3]

        conflicts = detect_choice_conflicts(choices)

        # Should detect one conflict between choice1 and choice2
        assert len(conflicts) == 1
        assert conflicts[0].neologism_term == "Dasein"
        assert conflicts[0].choice_a in [choice1, choice2]
        assert conflicts[0].choice_b in [choice1, choice2]


if __name__ == "__main__":
    pytest.main([__file__])

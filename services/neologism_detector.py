"""Neologism Detection Engine for philosophy-focused translation."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import spaCy for German linguistic analysis
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Import our data models
from models.neologism_models import (
    ConfidenceFactors,
    DetectedNeologism,
    MorphologicalAnalysis,
    NeologismAnalysis,
    NeologismType,
    PhilosophicalContext,
)
from services.confidence_scorer import ConfidenceScorer

# Import the new component classes
from services.morphological_analyzer import MorphologicalAnalyzer
from services.philosophical_context_analyzer import PhilosophicalContextAnalyzer

logger = logging.getLogger(__name__)


class NeologismDetector:
    """Core neologism detection engine with German morphological analysis."""

    def __init__(
        self,
        terminology_path: Optional[str] = None,
        spacy_model: str = "de_core_news_sm",
        cache_size: int = 1000,
        philosophical_threshold: float = 0.3,
        morphological_analyzer: Optional[MorphologicalAnalyzer] = None,
        philosophical_context_analyzer: Optional[PhilosophicalContextAnalyzer] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
    ):
        """Initialize the neologism detector.

        Args:
            terminology_path: Path to terminology JSON file
            spacy_model: Name of spaCy German model to use
            cache_size: Size of LRU cache for morphological analysis
            philosophical_threshold: Minimum philosophical density for
                detection
            morphological_analyzer: Optional MorphologicalAnalyzer instance
                (for dependency injection)
            philosophical_context_analyzer: Optional
                PhilosophicalContextAnalyzer instance
            confidence_scorer: Optional ConfidenceScorer instance
        """
        self.terminology_path = terminology_path
        self.spacy_model = spacy_model
        self.cache_size = cache_size
        self.philosophical_threshold = philosophical_threshold

        # Lazy loading attributes - models are loaded only when first accessed
        self._nlp = None
        self._nlp_lock = threading.Lock()
        self._terminology_map = None
        self._philosophical_indicators = None
        self._german_morphological_patterns = None

        # Component analyzers with dependency injection support
        self._morphological_analyzer = morphological_analyzer
        self._philosophical_context_analyzer = philosophical_context_analyzer
        self._confidence_scorer = confidence_scorer

        # Statistics tracking
        self.total_analyses = 0

        logger.info(
            f"NeologismDetector initialized with lazy loading for model: "
            f"{spacy_model}"
        )

    @property
    def nlp(self) -> Optional[Any]:
        """Lazy loading property for spaCy model with thread safety."""
        if self._nlp is None:
            with self._nlp_lock:
                if self._nlp is None:  # Double-check locking pattern
                    self._nlp = self._initialize_spacy_model()
                    logger.info(f"Lazy-loaded spaCy model: {self.spacy_model}")
        return self._nlp

    @property
    def terminology_map(self) -> dict[str, str]:
        """Lazy loading property for terminology mapping."""
        if self._terminology_map is None:
            self._terminology_map = self._load_terminology()
        return self._terminology_map

    @property
    def philosophical_indicators(self) -> set[str]:
        """Lazy loading property for philosophical indicators."""
        if self._philosophical_indicators is None:
            self._philosophical_indicators = self._load_philosophical_indicators()
        return self._philosophical_indicators

    @property
    def german_morphological_patterns(self) -> dict[str, list[str]]:
        """Lazy loading property for German morphological patterns."""
        if self._german_morphological_patterns is None:
            self._german_morphological_patterns = self._load_german_patterns()
        return self._german_morphological_patterns

    @property
    def morphological_analyzer(self) -> MorphologicalAnalyzer:
        """Lazy loading property for morphological analyzer."""
        if self._morphological_analyzer is None:
            self._morphological_analyzer = MorphologicalAnalyzer(
                spacy_model=self.nlp, cache_size=self.cache_size
            )
        return self._morphological_analyzer

    @property
    def philosophical_context_analyzer(self) -> PhilosophicalContextAnalyzer:
        """Lazy loading property for philosophical context analyzer."""
        if self._philosophical_context_analyzer is None:
            self._philosophical_context_analyzer = PhilosophicalContextAnalyzer(
                spacy_model=self.nlp, terminology_map=self.terminology_map
            )
        return self._philosophical_context_analyzer

    @property
    def confidence_scorer(self) -> ConfidenceScorer:
        """Lazy loading property for confidence scorer."""
        if self._confidence_scorer is None:
            self._confidence_scorer = ConfidenceScorer(
                philosophical_indicators=self.philosophical_indicators,
                german_morphological_patterns=self.german_morphological_patterns,
            )
        return self._confidence_scorer

    def _initialize_spacy_model(self) -> Optional[Any]:
        """Initialize spaCy German model for linguistic analysis."""
        if not SPACY_AVAILABLE:
            logger.warning(
                "spaCy not available, morphological analysis will be limited"
            )
            return None

        try:
            # Try to load the specified model
            nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
            return nlp
        except OSError:
            logger.warning(
                f"spaCy model '{self.spacy_model}' not found, trying fallback"
            )
            try:
                # Try basic German model
                nlp = spacy.load("de_core_news_sm")
                logger.info("Loaded fallback spaCy model: de_core_news_sm")
                return nlp
            except OSError:
                logger.warning(
                    "No German spaCy model available, using linguistic " "fallback"
                )
                return None

    def _load_terminology(self) -> dict[str, str]:
        """Load terminology mapping from JSON file."""
        terminology = {}

        if self.terminology_path and Path(self.terminology_path).exists():
            try:
                with open(self.terminology_path, encoding="utf-8") as f:
                    terminology = json.load(f)
                logger.info(f"Loaded {len(terminology)} terminology entries")
            except Exception as e:
                logger.error(f"Error loading terminology: {e}")

        # Add default Klages terminology if available
        try:
            klages_path = (
                Path(__file__).parent.parent / "config" / "klages_terminology.json"
            )
            if klages_path.exists():
                with open(klages_path, encoding="utf-8") as f:
                    klages_terms = json.load(f)
                    terminology.update(klages_terms)
                logger.info(f"Added {len(klages_terms)} Klages terminology entries")
        except Exception as e:
            logger.warning(f"Could not load Klages terminology: {e}")

        return terminology

    def _load_philosophical_indicators(self) -> set[str]:
        """Load philosophical indicator words and patterns."""
        indicators = {
            # Core philosophical terms
            "philosophie",
            "metaphysik",
            "ontologie",
            "epistemologie",
            "ethik",
            "ästhetik",
            "logik",
            "dialektik",
            "hermeneutik",
            "phänomenologie",
            # Conceptual terms
            "begriff",
            "konzept",
            "idee",
            "prinzip",
            "kategorie",
            "struktur",
            "system",
            "methode",
            "theorie",
            "hypothese",
            "thesis",
            "antithesis",
            # Existence and being
            "sein",
            "dasein",
            "existenz",
            "wesen",
            "substanz",
            "realität",
            "wirklichkeit",
            "erscheinung",
            "phänomen",
            "noumen",
            # Consciousness and mind
            "bewusstsein",
            "geist",
            "seele",
            "psyche",
            "verstand",
            "vernunft",
            "intellekt",
            "intuition",
            "erkenntnis",
            "wahrnehmung",
            "anschauung",
            # Value and meaning
            "wert",
            "bedeutung",
            "sinn",
            "zweck",
            "ziel",
            "ideal",
            "norm",
            "gut",
            "böse",
            "schön",
            "hässlich",
            "wahr",
            "falsch",
            # Temporal and spatial
            "zeit",
            "raum",
            "ewigkeit",
            "unendlichkeit",
            "endlichkeit",
            "temporal",
            "spatial",
            "chronos",
            "kairos",
            # Abstract relations
            "relation",
            "verhältnis",
            "beziehung",
            "zusammenhang",
            "einheit",
            "vielheit",
            "identität",
            "differenz",
            "ähnlichkeit",
            "verschiedenheit",
            # Specific philosophical movements
            "idealismus",
            "materialismus",
            "empirismus",
            "rationalismus",
            "kritizismus",
            "positivismus",
            "existentialismus",
            "nihilismus",
        }

        # Add terminology terms as indicators (avoid circular dependency)
        # Only add if terminology is already loaded, don't trigger lazy loading
        if self._terminology_map is not None:
            indicators.update(term.lower() for term in self._terminology_map.keys())

        return indicators

    def _load_german_patterns(self) -> dict[str, list[str]]:
        """Load German morphological patterns for compound analysis."""
        return {
            "compound_linking": ["s", "n", "es", "en", "er", "e", "ns", "ts"],
            "philosophical_prefixes": [
                "vor",
                "nach",
                "über",
                "unter",
                "zwischen",
                "gegen",
                "mit",
                "ur",
                "proto",
                "meta",
                "anti",
                "pseudo",
                "neo",
                "para",
            ],
            "abstract_suffixes": [
                "heit",
                "keit",
                "ung",
                "schaft",
                "tum",
                "nis",
                "sal",
                "ismus",
                "ität",
                "ation",
                "ismus",
                "logie",
                "sophie",
            ],
            "compound_patterns": [
                # Standard compound linking
                r"\w+(?:s|n|es|en|er|e|ns|ts)\w+",
                # Philosophical compounds
                r"\w+(?:bewusstsein|wirklichkeit|erkenntnis|wahrnehmung)",
                # Philosophical prefix compounds
                r"(?:welt|lebens|seins|geist|seele)\w+",
            ],
        }

    def analyze_text(
        self, text: str, text_id: str = "unknown", chunk_size: int = 2000
    ) -> NeologismAnalysis:
        """Analyze text for neologisms with comprehensive detection.

        Args:
            text: Input text to analyze
            text_id: Identifier for the text
            chunk_size: Size of text chunks for processing

        Returns:
            NeologismAnalysis with detected neologisms and statistics
        """
        start_time = time.time()

        # Initialize analysis
        analysis = NeologismAnalysis(
            text_id=text_id,
            analysis_timestamp=datetime.now().isoformat(),
            total_tokens=len(text.split()),
            analyzed_chunks=0,
        )

        try:
            # Process text in chunks for memory efficiency
            chunks = self._chunk_text(text, chunk_size)
            analysis.analyzed_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_neologisms = self._analyze_chunk(chunk, chunk_idx)

                for neologism in chunk_neologisms:
                    neologism.source_text_id = text_id
                    analysis.add_detection(neologism)

            # Calculate analysis quality metrics
            analysis.analysis_quality = self._calculate_analysis_quality(analysis)
            analysis.coverage_ratio = self._calculate_coverage_ratio(analysis, text)

            # Update semantic fields and concepts
            analysis.semantic_fields = self._extract_semantic_fields(analysis)
            analysis.dominant_concepts = self._extract_dominant_concepts(analysis)

        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            analysis.analysis_quality = 0.0

        finally:
            analysis.processing_time = time.time() - start_time
            # Get cache info from morphological analyzer
            cache_info = self.morphological_analyzer.get_cache_info()
            analysis.cache_hits = cache_info["hits"]
            self.total_analyses += 1

        logger.info(
            f"Analyzed text '{text_id}': {analysis.total_detections} "
            f"neologisms detected"
        )
        return analysis

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks while preserving sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        sentences = self._split_sentences(text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using spaCy or regex fallback."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            # Fallback sentence splitting
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

    def _analyze_chunk(self, chunk: str, chunk_idx: int) -> list[DetectedNeologism]:
        """Analyze a text chunk for neologisms."""
        neologisms = []

        # Extract candidate terms
        candidates = self._extract_candidates(chunk)

        for candidate in candidates:
            # Check if already known terminology
            if candidate.term.lower() in self.terminology_map:
                continue

            # Perform morphological analysis
            morphological_analysis = self._analyze_morphology(candidate.term)

            # Analyze philosophical context
            philosophical_context = self._analyze_philosophical_context(
                candidate.term, chunk, candidate.start_pos, candidate.end_pos
            )

            # Calculate confidence factors
            confidence_factors = self._calculate_confidence_factors(
                candidate.term, morphological_analysis, philosophical_context
            )

            # Determine if it's a neologism based on confidence
            confidence_score = confidence_factors.calculate_weighted_score()

            if confidence_score >= self.philosophical_threshold:
                neologism = DetectedNeologism(
                    term=candidate.term,
                    confidence=confidence_score,
                    neologism_type=self._classify_neologism_type(
                        morphological_analysis
                    ),
                    start_pos=candidate.start_pos,
                    end_pos=candidate.end_pos,
                    sentence_context=candidate.sentence_context,
                    morphological_analysis=morphological_analysis,
                    philosophical_context=philosophical_context,
                    confidence_factors=confidence_factors,
                    detection_timestamp=datetime.now().isoformat(),
                )

                neologisms.append(neologism)

        return neologisms

    @dataclass
    class CandidateTerm:
        """Temporary structure for candidate terms."""

        term: str
        start_pos: int
        end_pos: int
        sentence_context: str

    def _extract_candidates(self, text: str) -> list[CandidateTerm]:
        """Extract candidate terms from text for neologism analysis."""
        candidates = []

        if self.nlp:
            doc = self.nlp(text)

            for token in doc:
                # Skip short words, punctuation, and common words
                if (
                    len(token.text) < 6
                    or token.is_punct
                    or token.is_stop
                    or token.like_num
                ):
                    continue

                # Focus on nouns and complex words
                if token.pos_ in ["NOUN", "PROPN"] or self._is_compound_word(
                    token.text
                ):
                    sentence_context = token.sent.text if token.sent else text

                    candidates.append(
                        self.CandidateTerm(
                            term=token.text,
                            start_pos=token.idx,
                            end_pos=token.idx + len(token.text),
                            sentence_context=sentence_context,
                        )
                    )
        else:
            # Fallback extraction using regex
            candidates = self._extract_candidates_regex(text)

        return candidates

    def _extract_candidates_regex(self, text: str) -> list[CandidateTerm]:
        """Fallback candidate extraction using regex patterns."""
        candidates = []

        # German compound word patterns
        compound_patterns = [
            # CapitalizedCompounds
            r"\b[A-ZÄÖÜ][a-zäöüß]{5,}(?:[A-ZÄÖÜ][a-zäöüß]+)+\b",
            # linked compounds
            r"\b[a-zäöüß]+(?:s|n|es|en|er|e|ns|ts)[a-zäöüß]{4,}\b",
            # abstract suffixes
            r"\b[a-zäöüß]+(?:heit|keit|ung|schaft|tum|nis|sal|ismus|ität|logie|sophie)\b",
        ]

        sentences = self._split_sentences(text)

        for sentence in sentences:
            for pattern in compound_patterns:
                for match in re.finditer(pattern, sentence):
                    term = match.group()
                    if len(term) >= 6:  # Minimum length for consideration
                        candidates.append(
                            self.CandidateTerm(
                                term=term,
                                start_pos=match.start(),
                                end_pos=match.end(),
                                sentence_context=sentence,
                            )
                        )

        return candidates

    def _is_compound_word(self, word: str) -> bool:
        """Check if word appears to be a German compound."""
        if len(word) < 8:
            return False

        # Check for compound patterns
        patterns = self.german_morphological_patterns["compound_patterns"]
        for pattern in patterns:
            if re.search(pattern, word.lower()):
                return True

        # Check for multiple capital letters (German noun compounds)
        capital_count = sum(1 for c in word if c.isupper())
        if capital_count >= 2:
            return True

        return False

    def _analyze_morphology(self, term: str) -> MorphologicalAnalysis:
        """Analyze morphological structure of a term."""
        return self.morphological_analyzer.analyze(term)

    def _analyze_philosophical_context(
        self, term: str, text: str, start_pos: int, end_pos: int
    ) -> PhilosophicalContext:
        """Analyze philosophical context around a term."""
        return self.philosophical_context_analyzer.analyze_context(
            term, text, start_pos, end_pos
        )

    def _calculate_confidence_factors(
        self,
        term: str,
        morphological: MorphologicalAnalysis,
        philosophical: PhilosophicalContext,
    ) -> ConfidenceFactors:
        """Calculate individual confidence factors."""
        return self.confidence_scorer.calculate_confidence_factors(
            term, morphological, philosophical
        )

    def _classify_neologism_type(
        self, morphological: MorphologicalAnalysis
    ) -> NeologismType:
        """Classify the type of neologism."""
        if morphological.is_compound:
            return NeologismType.COMPOUND

        if morphological.prefixes or morphological.suffixes:
            return NeologismType.DERIVED

        # Check for philosophical indicators
        philosophical_suffixes = ["ismus", "logie", "sophie"]
        if any(suffix in morphological.suffixes for suffix in philosophical_suffixes):
            return NeologismType.PHILOSOPHICAL_TERM

        # Check for technical indicators
        technical_suffixes = ["ation", "ität", "ismus"]
        if any(suffix in morphological.suffixes for suffix in technical_suffixes):
            return NeologismType.TECHNICAL_TERM

        return NeologismType.UNKNOWN

    def _calculate_analysis_quality(self, analysis: NeologismAnalysis) -> float:
        """Calculate overall analysis quality score."""
        if not analysis.detected_neologisms:
            return 0.5  # Neutral score for no detections

        # Quality factors
        neologisms = analysis.detected_neologisms
        confidence_avg = sum(n.confidence for n in neologisms) / len(neologisms)
        type_diversity = len(set(n.neologism_type for n in neologisms))

        quality = (confidence_avg * 0.6) + (min(1.0, type_diversity / 5.0) * 0.4)

        return quality

    def _calculate_coverage_ratio(
        self, analysis: NeologismAnalysis, text: str
    ) -> float:
        """Calculate coverage ratio of analysis."""
        if not analysis.detected_neologisms:
            return 0.0

        # Calculate ratio of analyzed terms to total words
        total_words = len(text.split())
        analyzed_terms = len(analysis.detected_neologisms)

        # Use configurable expected neologism ratio
        expected_ratio = getattr(self, "expected_neologism_ratio", 0.1)
        return min(1.0, analyzed_terms / max(1, total_words * expected_ratio))

    def _extract_semantic_fields(self, analysis: NeologismAnalysis) -> list[str]:
        """Extract semantic fields from analysis."""
        return self.philosophical_context_analyzer.extract_semantic_fields(
            analysis.detected_neologisms
        )

    def _extract_dominant_concepts(self, analysis: NeologismAnalysis) -> list[str]:
        """Extract dominant concepts from analysis."""
        return self.philosophical_context_analyzer.extract_dominant_concepts(
            analysis.detected_neologisms
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        cache_info = self.morphological_analyzer.get_cache_info()

        return {
            "total_analyses": self.total_analyses,
            "cache_hits": cache_info["hits"],
            "cache_misses": cache_info["misses"],
            "cache_size": cache_info["currsize"],
            "cache_hit_rate": cache_info["hit_rate"],
            "spacy_available": SPACY_AVAILABLE,
            "spacy_model": self.spacy_model if self.nlp else None,
            "terminology_entries": len(self.terminology_map),
            "philosophical_indicators": len(self.philosophical_indicators),
        }

    def clear_cache(self) -> None:
        """Clear component caches."""
        self.morphological_analyzer.clear_cache()
        logger.info("Component caches cleared")


# Convenience functions for batch processing


def analyze_document_batch(
    detector: NeologismDetector, texts: list[str], text_ids: Optional[list[str]] = None
) -> list[NeologismAnalysis]:
    """Analyze multiple documents for neologisms."""
    if text_ids is None:
        text_ids = [f"doc_{i}" for i in range(len(texts))]

    analyses = []
    for text, text_id in zip(texts, text_ids):
        analysis = detector.analyze_text(text, text_id)
        analyses.append(analysis)

    return analyses


def merge_neologism_analyses(
    analyses: list[NeologismAnalysis],
) -> NeologismAnalysis:
    """Merge multiple analyses into a single comprehensive analysis."""
    if not analyses:
        raise ValueError("Cannot merge empty list of analyses")

    # Use the merge function from the models module
    from models.neologism_models import merge_analyses

    return merge_analyses(analyses)

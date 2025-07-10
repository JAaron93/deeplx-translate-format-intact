"""Morphological Analysis Engine for German philosophical texts."""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Optional, Any
from functools import lru_cache
from spacy.language import Language

from models.neologism_models import MorphologicalAnalysis

logger = logging.getLogger(__name__)


class MorphologicalAnalyzer:
    """Handles morphological analysis of German terms including compound analysis."""
    
    def __init__(self, 
                 spacy_model: Optional[Language] = None,
                 cache_size: int = 1000):
        """
        Initialize the morphological analyzer.
        
        Args:
            spacy_model: spaCy model instance for linguistic analysis
            cache_size: Size of LRU cache for morphological analysis
        """
        self.nlp = spacy_model
        self.cache_size = cache_size
        self.german_morphological_patterns = self._load_german_patterns()
        
        # Configure the cache for the analyze method
        self.analyze = lru_cache(maxsize=cache_size)(self._analyze_uncached)
        
        logger.info("MorphologicalAnalyzer initialized")
    
    def _load_german_patterns(self) -> Dict[str, List[str]]:
        """Load German morphological patterns for compound analysis."""
        return {
            "compound_linking": [
                "s", "n", "es", "en", "er", "e", "ns", "ts"
            ],
            "philosophical_prefixes": [
                "vor", "nach", "über", "unter", "zwischen", "gegen", "mit",
                "ur", "proto", "meta", "anti", "pseudo", "neo", "para"
            ],
            "abstract_suffixes": [
                "heit", "keit", "ung", "schaft", "tum", "nis", "sal",
                "ismus", "ität", "ation", "logie", "sophie"
            ],
            "compound_patterns": [
                r"\w+(?:s|n|es|en|er|e|ns|ts)\w+",  # Standard compound linking
                r"\w+(?:bewusstsein|wirklichkeit|erkenntnis|wahrnehmung)",  # Philosophical compounds
                r"(?:welt|lebens|seins|geist|seele)\w+",  # Philosophical prefix compounds
            ]
        }
    
    def _analyze_uncached(self, term: str) -> MorphologicalAnalysis:
        """Analyze morphological structure of a term (uncached version)."""
        analysis = MorphologicalAnalysis()
        
        # Basic metrics
        analysis.word_length = len(term)
        analysis.syllable_count = self._count_syllables(term)
        
        # Compound analysis
        if self._is_compound_word(term):
            analysis.is_compound = True
            analysis.compound_parts = self._split_compound(term)
            analysis.compound_pattern = self._identify_compound_pattern(term)
        
        # Morpheme analysis
        analysis.prefixes = self._extract_prefixes(term)
        analysis.suffixes = self._extract_suffixes(term)
        analysis.root_words = self._extract_root_words(term)
        analysis.morpheme_count = len(analysis.prefixes) + len(analysis.suffixes) + len(analysis.root_words)
        
        # Complexity metrics
        analysis.structural_complexity = self._calculate_structural_complexity(analysis)
        analysis.morphological_productivity = self._calculate_morphological_productivity(analysis)
        
        return analysis
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for German words."""
        vowels = "aeiouäöü"
        count = 0
        prev_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_was_vowel:
                    count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        return max(1, count)
    
    def _is_compound_word(self, word: str) -> bool:
        """Check if word appears to be a German compound."""
        if len(word) < 8:
            return False
        
        # Check for compound patterns
        for pattern in self.german_morphological_patterns["compound_patterns"]:
            if re.search(pattern, word.lower()):
                return True
        
        # Check for multiple capital letters (German noun compounds)
        capital_count = sum(1 for c in word if c.isupper())
        if capital_count >= 2:
            return True
        
        return False
    
    def _split_compound(self, term: str) -> List[str]:
        """Split German compound word into components."""
        parts = []
        
        # Try spaCy-based splitting if available
        if self.nlp:
            doc = self.nlp(term)
            if doc and len(doc) > 0:
                # Use morphological analysis if available
                token = doc[0]
                if hasattr(token, 'morph') and token.morph.get('Compound'):
                    # spaCy detected compound structure
                    parts = token.morph.get('Compound')
        
        # Fallback to pattern-based splitting
        if not parts:
            parts = self._split_compound_pattern(term)
        
        return parts if parts else [term]
    
    def _split_compound_pattern(self, term: str) -> List[str]:
        """Pattern-based compound splitting for German."""
        parts = []
        
        # Simple heuristic: split on capital letters after lowercase
        current_part = ""
        for i, char in enumerate(term):
            if char.isupper() and i > 0 and term[i-1].islower():
                if current_part:
                    parts.append(current_part)
                current_part = char
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Filter out very short parts
        parts = [p for p in parts if len(p) >= 3]
        
        return parts if len(parts) > 1 else [term]
    
    def _identify_compound_pattern(self, term: str) -> Optional[str]:
        """Identify the compound formation pattern."""
        linking_elements = self.german_morphological_patterns["compound_linking"]
        
        for element in linking_elements:
            pattern = f"\\w+{element}\\w+"
            if re.search(pattern, term.lower()):
                return f"linked_with_{element}"
        
        if any(char.isupper() for char in term[1:]):
            return "capital_concatenation"
        
        return "simple_concatenation"
    
    def _extract_prefixes(self, term: str) -> List[str]:
        """Extract prefixes from term."""
        prefixes = []
        term_lower = term.lower()
        
        for prefix in self.german_morphological_patterns["philosophical_prefixes"]:
            if term_lower.startswith(prefix):
                prefixes.append(prefix)
        
        return prefixes
    
    def _extract_suffixes(self, term: str) -> List[str]:
        """Extract suffixes from term."""
        suffixes = []
        term_lower = term.lower()
        
        for suffix in self.german_morphological_patterns["abstract_suffixes"]:
            if term_lower.endswith(suffix):
                suffixes.append(suffix)
        
        return suffixes
    
    def _extract_root_words(self, term: str) -> List[str]:
        """Extract root words from compound or derived term."""
        roots = []
        
        # For compounds, the parts are potential roots
        if self._is_compound_word(term):
            compound_parts = self._split_compound(term)
            for part in compound_parts:
                # Remove linking elements and check if it's a meaningful root
                cleaned_part = self._clean_morpheme(part)
                if len(cleaned_part) >= 3:
                    roots.append(cleaned_part)
        
        # For derived words, try to extract root
        term_clean = term.lower()
        for prefix in self.german_morphological_patterns["philosophical_prefixes"]:
            if term_clean.startswith(prefix):
                term_clean = term_clean[len(prefix):]
                break
        
        for suffix in self.german_morphological_patterns["abstract_suffixes"]:
            if term_clean.endswith(suffix):
                term_clean = term_clean[:-len(suffix)]
                break
        
        if term_clean and len(term_clean) >= 3:
            roots.append(term_clean)
        
        return list(set(roots)) if roots else [term]
    
    def _clean_morpheme(self, morpheme: str) -> str:
        """Clean morpheme by removing linking elements."""
        cleaned = morpheme.lower()
        
        # Remove common linking elements
        for element in self.german_morphological_patterns["compound_linking"]:
            if cleaned.endswith(element):
                cleaned = cleaned[:-len(element)]
                break
        
        return cleaned
    
    def _calculate_structural_complexity(self, analysis: MorphologicalAnalysis) -> float:
        """Calculate structural complexity score."""
        complexity = 0.0
        
        # Length factor
        complexity += min(1.0, analysis.word_length / 20.0)
        
        # Morpheme count factor
        complexity += min(1.0, analysis.morpheme_count / 5.0)
        
        # Compound factor
        if analysis.is_compound:
            complexity += 0.3
            complexity += min(0.3, len(analysis.compound_parts) / 10.0)
        
        # Syllable factor
        complexity += min(0.4, analysis.syllable_count / 8.0)
        
        return min(1.0, complexity)
    
    def _calculate_morphological_productivity(self, analysis: MorphologicalAnalysis) -> float:
        """Calculate morphological productivity score."""
        productivity = 0.0
        
        # Prefix productivity
        if analysis.prefixes:
            productivity += 0.3
        
        # Suffix productivity
        if analysis.suffixes:
            productivity += 0.3
        
        # Compound productivity
        if analysis.is_compound and len(analysis.compound_parts) > 2:
            productivity += 0.4
        
        return min(1.0, productivity)
    
    def clear_cache(self) -> None:
        """Clear the morphological analysis cache."""
        if hasattr(self.analyze, 'cache_clear'):
            self.analyze.cache_clear()
        logger.info("Morphological analysis cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        if hasattr(self.analyze, 'cache_info'):
            cache_info = self.analyze.cache_info()
            return {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "currsize": cache_info.currsize,
                "maxsize": cache_info.maxsize,
                "hit_rate": cache_info.hits / max(1, cache_info.hits + cache_info.misses)
            }
        return {"hits": 0, "misses": 0, "currsize": 0, "maxsize": 0, "hit_rate": 0.0}
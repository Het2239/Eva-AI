#!/usr/bin/env python3
"""
EVA RAG - Extraction Quality Assessment
=======================================
Assess document extraction quality to enable confidence-based parser routing.
"""

import re
from dataclasses import dataclass
from typing import Optional


# Common English words for meaningful content detection
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
}


@dataclass
class QualityMetrics:
    """Quality metrics for extracted text."""
    text_length: int
    alpha_ratio: float
    avg_word_length: float
    garbage_ratio: float
    meaningful_word_ratio: float
    confidence_score: float
    
    def to_dict(self) -> dict:
        return {
            "text_length": self.text_length,
            "alpha_ratio": round(self.alpha_ratio, 3),
            "avg_word_length": round(self.avg_word_length, 2),
            "garbage_ratio": round(self.garbage_ratio, 3),
            "meaningful_word_ratio": round(self.meaningful_word_ratio, 3),
            "confidence_score": round(self.confidence_score, 3),
        }


class ExtractionQuality:
    """
    Assess extraction quality using multiple heuristics.
    
    Quality Signals:
        1. Text length - Very short extractions indicate failure
        2. Alpha ratio - Low alphabetic character ratio = garbage
        3. Average word length - OCR gibberish has very short "words"
        4. Garbage ratio - High non-alphabetic tokens = bad extraction
        5. Meaningful words - Check for common English words
    """
    
    def __init__(
        self,
        min_length: int = 50,
        min_alpha_ratio: float = 0.6,
        min_avg_word_length: float = 2.5,
        max_garbage_ratio: float = 0.3,
        min_meaningful_ratio: float = 0.1,
    ):
        """
        Initialize quality assessor with configurable thresholds.
        
        Args:
            min_length: Minimum text length for acceptable extraction
            min_alpha_ratio: Minimum ratio of alphabetic characters
            min_avg_word_length: Minimum average word length
            max_garbage_ratio: Maximum ratio of garbage tokens
            min_meaningful_ratio: Minimum ratio of meaningful words
        """
        self.min_length = min_length
        self.min_alpha_ratio = min_alpha_ratio
        self.min_avg_word_length = min_avg_word_length
        self.max_garbage_ratio = max_garbage_ratio
        self.min_meaningful_ratio = min_meaningful_ratio
    
    def _calculate_alpha_ratio(self, text: str) -> float:
        """Calculate ratio of alphabetic characters."""
        if not text:
            return 0.0
        alpha_count = sum(1 for c in text if c.isalpha())
        # Only consider non-whitespace characters
        non_whitespace = sum(1 for c in text if not c.isspace())
        return alpha_count / non_whitespace if non_whitespace > 0 else 0.0
    
    def _calculate_avg_word_length(self, words: list) -> float:
        """Calculate average word length."""
        if not words:
            return 0.0
        total_length = sum(len(w) for w in words)
        return total_length / len(words)
    
    def _calculate_garbage_ratio(self, words: list) -> float:
        """Calculate ratio of garbage tokens (non-alphabetic)."""
        if not words:
            return 1.0
        garbage_count = sum(1 for w in words if not any(c.isalpha() for c in w))
        return garbage_count / len(words)
    
    def _calculate_meaningful_ratio(self, words: list) -> float:
        """Calculate ratio of common English words."""
        if not words:
            return 0.0
        meaningful_count = sum(1 for w in words if w.lower() in COMMON_WORDS)
        return meaningful_count / len(words)
    
    def assess(self, text: str) -> QualityMetrics:
        """
        Assess extraction quality and return metrics.
        
        Args:
            text: Extracted text to assess
            
        Returns:
            QualityMetrics with individual metrics and overall confidence score
        """
        if not text or not text.strip():
            return QualityMetrics(
                text_length=0,
                alpha_ratio=0.0,
                avg_word_length=0.0,
                garbage_ratio=1.0,
                meaningful_word_ratio=0.0,
                confidence_score=0.0,
            )
        
        # Tokenize into words
        words = re.findall(r'\S+', text)
        
        # Calculate individual metrics
        text_length = len(text)
        alpha_ratio = self._calculate_alpha_ratio(text)
        avg_word_length = self._calculate_avg_word_length(words)
        garbage_ratio = self._calculate_garbage_ratio(words)
        meaningful_ratio = self._calculate_meaningful_ratio(words)
        
        # Calculate confidence score (weighted average of normalized metrics)
        scores = []
        
        # Length score (0-1, caps at 500 chars)
        length_score = min(text_length / 500, 1.0)
        scores.append(length_score * 0.15)
        
        # Alpha ratio score
        alpha_score = min(alpha_ratio / self.min_alpha_ratio, 1.0)
        scores.append(alpha_score * 0.25)
        
        # Word length score
        word_length_score = min(avg_word_length / self.min_avg_word_length, 1.0)
        scores.append(word_length_score * 0.2)
        
        # Garbage ratio score (inverted - lower is better)
        garbage_score = max(0, 1 - garbage_ratio / self.max_garbage_ratio)
        scores.append(garbage_score * 0.2)
        
        # Meaningful words score
        meaningful_score = min(meaningful_ratio / self.min_meaningful_ratio, 1.0)
        scores.append(meaningful_score * 0.2)
        
        confidence_score = sum(scores)
        
        return QualityMetrics(
            text_length=text_length,
            alpha_ratio=alpha_ratio,
            avg_word_length=avg_word_length,
            garbage_ratio=garbage_ratio,
            meaningful_word_ratio=meaningful_ratio,
            confidence_score=confidence_score,
        )
    
    def is_acceptable(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if extraction quality is acceptable.
        
        Args:
            text: Extracted text to assess
            threshold: Minimum confidence score (0.0-1.0)
            
        Returns:
            True if quality meets threshold
        """
        metrics = self.assess(text)
        return metrics.confidence_score >= threshold


# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extraction_quality.py <text_file>")
        print("       python extraction_quality.py --test")
        sys.exit(1)
    
    if sys.argv[1] == "--test":
        assessor = ExtractionQuality()
        
        # Test cases
        test_cases = [
            ("Good extraction", "The quick brown fox jumps over the lazy dog. This is a sample document with meaningful content."),
            ("Short text", "Hello"),
            ("Garbage text", "x1 x2 x3 x4 x5 ### @@@ !!!"),
            ("OCR gibberish", "I I l 1 | ! l I i l 1 I l"),
            ("Empty", ""),
        ]
        
        print("Extraction Quality Test Results")
        print("=" * 60)
        
        for name, text in test_cases:
            metrics = assessor.assess(text)
            acceptable = assessor.is_acceptable(text)
            print(f"\n{name}:")
            print(f"  Confidence: {metrics.confidence_score:.3f} {'✓' if acceptable else '✗'}")
            print(f"  Length: {metrics.text_length}, Alpha: {metrics.alpha_ratio:.2f}")
            print(f"  Avg Word: {metrics.avg_word_length:.1f}, Garbage: {metrics.garbage_ratio:.2f}")
    else:
        # Read file and assess
        with open(sys.argv[1], 'r') as f:
            text = f.read()
        
        assessor = ExtractionQuality()
        metrics = assessor.assess(text)
        
        print(f"Quality Assessment: {sys.argv[1]}")
        print("-" * 40)
        for key, value in metrics.to_dict().items():
            print(f"  {key}: {value}")
        print(f"\nAcceptable: {'✓ Yes' if assessor.is_acceptable(text) else '✗ No'}")

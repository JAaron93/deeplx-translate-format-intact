#!/usr/bin/env python3
"""Debug script to test compound word detection."""

import re
from services.neologism_detector import NeologismDetector

def debug_compound_detection():
    detector = NeologismDetector()

    test_words = [
        "Bewusstsein",
        "Wirklichkeitsbewusstsein",
        "Bewusstseinsphilosophie",
        "Lebensweltthematik"
    ]

    for word in test_words:
        result = detector.debug_compound_detection(word)
        print(f"{word}: {result}")

        # Debug the logic
        word_lower = word.lower()
        print(f"  Length: {len(word)} (>= 10: {len(word) >= 10})")

        # Check capital letters
        capital_count = sum(1 for c in word if c.isupper())
        print(f"  Capital count: {capital_count} (>= 2: {capital_count >= 2})")

        # Check philosophical endings
        philosophical_endings = detector.german_morphological_patterns["philosophical_endings"]
        for ending in philosophical_endings:
            if word_lower.endswith(ending):
                prefix = word_lower[:-len(ending)]
                print(f"  Ends with '{ending}', prefix: '{prefix}' (len: {len(prefix)}) -> compound: {len(prefix) >= 4}")

        # Check standard compound linking
        if re.search(r"^\w{4,}(?:s|en|er)\w{4,}$", word_lower):
            print(f"  Matches standard compound linking pattern")

        # Check philosophical prefix compounds
        if re.search(r"^(?:welt|lebens|seins|geist|seele)\w{4,}$", word_lower):
            print(f"  Matches philosophical prefix pattern")

        print()

if __name__ == "__main__":
    debug_compound_detection()

#!/usr/bin/env python3
"""
grep_normalizer.py
==================
Curriculum text normalization logic for Project Chimera.
Cleans raw text inputs for vectorization and TTN fusion.
"""

import re
import string
from typing import List

class GrepNormalizer:
    def __init__(self):
        # Common Norwegian curriculum stop words or noise patterns
        self.noise_patterns = [
            r"Side \d+ av \d+",
            r"Læreplan i .* - .* \(.*\)",
            r"Sist endret \d{2}\.\d{2}\.\d{4}"
        ]

    def normalize(self, text: str) -> str:
        """
        Main normalization pipeline.
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, " ", text)
            
        # 3. Handle whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        # 4. Remove punctuation but keep competence aim markers like '7.0' or 'G5'
        # Simple regex for curriculum markers vs punctuation
        text = "".join([c for c in text if c.isalnum() or c in (" ", ".", "-")])
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes for 'grep' / search ingestion.
        """
        norm_text = self.normalize(text)
        return norm_text.split(" ")

if __name__ == "__main__":
    normalizer = GrepNormalizer()
    sample = "Læreplan i matematikk (MAT01-05). Side 1 av 12. Kompetansemål etter 10. trinn."
    print(f"Normalized: {normalizer.normalize(sample)}")

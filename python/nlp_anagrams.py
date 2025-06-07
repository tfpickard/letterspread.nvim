#!/usr/bin/env python3
"""
Enhanced anagram detection using NLP
Provides semantic similarity, word frequency analysis, and definitions
"""

import sys
import json
import argparse
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import spacy
    import nltk
    from nltk.corpus import wordnet, brown
    from nltk.probability import FreqDist
    import numpy as np
except ImportError as e:
    print(f"Error importing required libraries: {e}", file=sys.stderr)
    print("Install with: pip install spacy nltk numpy", file=sys.stderr)
    sys.exit(1)


class EnhancedAnagramFinder:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        self.word_freq = None

    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        try:
            nltk.data.find("corpora/brown")
        except LookupError:
            nltk.download("brown", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)

    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Install with:", file=sys.stderr)
            print("python -m spacy download en_core_web_sm", file=sys.stderr)
            # Fallback to basic tokenization
            self.nlp = None

    def get_word_frequency(self, words: List[str]) -> FreqDist:
        """Get word frequency distribution from Brown corpus"""
        if self.word_freq is None:
            brown_words = [word.lower() for word in brown.words()]
            self.word_freq = FreqDist(brown_words)
        return self.word_freq

    def extract_words(self, text: str, min_length: int = 3) -> List[str]:
        """Extract words from text using NLP tokenization"""
        if self.nlp:
            doc = self.nlp(text)
            words = [
                token.text.lower()
                for token in doc
                if token.is_alpha and len(token.text) >= min_length
            ]
        else:
            # Fallback to regex
            words = re.findall(r"\b[a-zA-Z]{" + str(min_length) + ",}\b", text.lower())

        return list(set(words))  # Remove duplicates

    def sort_letters(self, word: str) -> str:
        """Sort letters in a word"""
        return "".join(sorted(word.lower()))

    def find_anagram_groups(self, words: List[str]) -> Dict[str, List[str]]:
        """Group words by their sorted letter patterns"""
        groups = defaultdict(list)

        for word in words:
            sorted_word = self.sort_letters(word)
            groups[sorted_word].append(word)

        # Filter to groups with multiple words
        return {k: v for k, v in groups.items() if len(v) > 1}

    def get_word_definition(self, word: str) -> Optional[str]:
        """Get word definition using WordNet"""
        synsets = wordnet.synsets(word)
        if synsets:
            return synsets[0].definition()
        return None

    def calculate_semantic_similarity(self, words: List[str]) -> float:
        """Calculate semantic similarity between anagram words"""
        if not self.nlp or len(words) < 2:
            return 0.0

        docs = [self.nlp(word) for word in words]
        similarities = []

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                similarity = docs[i].similarity(docs[j])
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def get_word_frequency_score(self, words: List[str]) -> float:
        """Get average frequency score for words"""
        freq_dist = self.get_word_frequency(words)
        scores = [freq_dist.freq(word) for word in words]
        return np.mean(scores) if scores else 0.0

    def analyze_anagrams(
        self,
        text: str,
        min_length: int = 3,
        include_semantic: bool = True,
        filter_frequency: bool = True,
    ) -> Dict:
        """Main anagram analysis function"""
        words = self.extract_words(text, min_length)

        if len(words) < 2:
            return {"anagram_groups": [], "metadata": {"total_words": len(words)}}

        anagram_groups = self.find_anagram_groups(words)

        results = []
        for sorted_letters, group_words in anagram_groups.items():
            group_data = {"words": group_words, "letter_pattern": sorted_letters}

            # Add definitions
            definitions = []
            for word in group_words:
                definition = self.get_word_definition(word)
                if definition:
                    definitions.append(f"{word}: {definition}")
            group_data["definitions"] = definitions

            # Add semantic similarity if requested
            if include_semantic:
                semantic_score = self.calculate_semantic_similarity(group_words)
                group_data["semantic_score"] = semantic_score

            # Add frequency information
            if filter_frequency:
                freq_score = self.get_word_frequency_score(group_words)
                group_data["frequency_score"] = freq_score

            results.append(group_data)

        # Sort by semantic similarity or frequency
        if include_semantic:
            results.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)
        elif filter_frequency:
            results.sort(key=lambda x: x.get("frequency_score", 0), reverse=True)

        # Filter out very low frequency words if requested
        if filter_frequency:
            freq_threshold = 0.00001  # Adjust as needed
            results = [
                r for r in results if r.get("frequency_score", 0) > freq_threshold
            ]

        return {
            "anagram_groups": results,
            "metadata": {
                "total_words": len(words),
                "total_groups": len(results),
                "semantic_analysis": include_semantic,
                "frequency_filtering": filter_frequency,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced anagram detection")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--min-length", type=int, default=3, help="Minimum word length")
    parser.add_argument(
        "--semantic-similarity",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Include semantic similarity",
    )
    parser.add_argument(
        "--filter-frequency",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Filter by word frequency",
    )

    args = parser.parse_args()

    # Read input
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    # Process
    try:
        finder = EnhancedAnagramFinder()
        result = finder.analyze_anagrams(
            text,
            min_length=args.min_length,
            include_semantic=(args.semantic_similarity == "true"),
            filter_frequency=(args.filter_frequency == "true"),
        )
    except Exception as e:
        print(f"Error processing anagrams: {e}", file=sys.stderr)
        sys.exit(1)

    # Write output
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enhanced poetry generation using NLP
Provides proper syllable counting, rhyme detection, sentiment analysis,
and part-of-speech tagging for better word selection
"""

import sys
import json
import argparse
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import spacy
    import nltk
    from nltk.corpus import cmudict, wordnet
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import pyphen
    import pronouncing
    import numpy as np
except ImportError as e:
    print(f"Error importing required libraries: {e}", file=sys.stderr)
    print(
        "Install with: pip install spacy nltk pyphen pronouncing numpy", file=sys.stderr
    )
    sys.exit(1)


class EnhancedPoetryGenerator:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        self.setup_syllable_counter()

    def setup_nltk(self):
        """Download required NLTK data"""
        required_data = [
            ("corpora/cmudict", "cmudict"),
            ("vader_lexicon", "vader_lexicon"),
            ("tokenizers/punkt", "punkt"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/wordnet", "wordnet"),
        ]

        for data_path, data_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                nltk.download(data_name, quiet=True)

        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Load pronunciation dictionary
        try:
            self.pronouncing_dict = cmudict.dict()
        except:
            self.pronouncing_dict = {}

    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found", file=sys.stderr)
            self.nlp = None

    def setup_syllable_counter(self):
        """Setup pyphen for syllable counting"""
        self.pyphen_dic = pyphen.Pyphen(lang="en")

    def count_syllables(self, word: str) -> int:
        """Accurate syllable counting using multiple methods"""
        word = word.lower().strip()

        # Method 1: pyphen
        try:
            syllables = len(self.pyphen_dic.positions(word)) + 1
            if syllables > 0:
                return syllables
        except:
            pass

        # Method 2: CMU Pronouncing Dictionary
        if word in self.pronouncing_dict:
            phones = self.pronouncing_dict[word][0]
            syllables = sum(1 for phone in phones if phone[-1].isdigit())
            if syllables > 0:
                return syllables

        # Method 3: vowel counting fallback
        vowels = "aeiouAEIOU"
        syllables = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel

        # Handle silent e
        if word.endswith("e") and syllables > 1:
            syllables -= 1

        return max(1, syllables)

    def extract_words_with_pos(self, text: str) -> Dict[str, List[str]]:
        """Extract words grouped by part of speech"""
        if self.nlp:
            doc = self.nlp(text)
            words_by_pos = defaultdict(list)

            for token in doc:
                if token.is_alpha and len(token.text) > 1:
                    pos_tag = token.pos_
                    words_by_pos[pos_tag].append(token.text.lower())
        else:
            # Fallback using NLTK
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            words_by_pos = defaultdict(list)

            for word, pos in pos_tags:
                if word.isalpha() and len(word) > 1:
                    words_by_pos[pos].append(word.lower())

        return dict(words_by_pos)

    def get_words_by_syllables(self, words: List[str]) -> Dict[int, List[str]]:
        """Group words by syllable count"""
        words_by_syllables = defaultdict(list)

        for word in words:
            syllables = self.count_syllables(word)
            words_by_syllables[syllables].append(word)

        return dict(words_by_syllables)

    def find_rhymes(self, word: str) -> List[str]:
        """Find rhyming words using pronouncing library"""
        try:
            rhymes = pronouncing.rhymes(word)
            return rhymes[:10]  # Limit to 10 rhymes
        except:
            return []

    def get_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        scores = self.sentiment_analyzer.polarity_scores(text)

        # Determine overall sentiment
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def select_words_by_sentiment(
        self, words: List[str], target_sentiment: str
    ) -> List[str]:
        """Filter words that match target sentiment"""
        if target_sentiment == "neutral":
            return words

        filtered_words = []
        for word in words:
            word_sentiment = self.get_sentiment(word)
            if word_sentiment["label"] == target_sentiment:
                filtered_words.append(word)

        return filtered_words if filtered_words else words  # Fallback to all words

    def generate_line(
        self,
        target_syllables: int,
        words_by_syllables: Dict[int, List[str]],
        rhyme_with: Optional[str] = None,
        sentiment: Optional[str] = None,
    ) -> str:
        """Generate a line with specified syllable count and optional rhyme/sentiment"""
        line_words = []
        remaining = target_syllables

        # If we need to rhyme, start with a rhyming word
        if rhyme_with:
            rhymes = self.find_rhymes(rhyme_with)
            if rhymes:
                # Find rhymes that fit syllable count
                suitable_rhymes = []
                for rhyme in rhymes:
                    rhyme_syllables = self.count_syllables(rhyme)
                    if rhyme_syllables <= remaining:
                        suitable_rhymes.append((rhyme, rhyme_syllables))

                if suitable_rhymes:
                    rhyme_word, rhyme_syllables = random.choice(suitable_rhymes)
                    line_words.append(rhyme_word)
                    remaining -= rhyme_syllables

        # Fill remaining syllables
        attempts = 0
        while remaining > 0 and attempts < 50:
            possible_syllables = [
                s for s in words_by_syllables.keys() if s <= remaining
            ]
            if not possible_syllables:
                break

            chosen_syllables = random.choice(possible_syllables)
            available_words = words_by_syllables[chosen_syllables]

            # Filter by sentiment if specified
            if sentiment:
                available_words = self.select_words_by_sentiment(
                    available_words, sentiment
                )

            if available_words:
                word = random.choice(available_words)
                line_words.append(word)
                remaining -= chosen_syllables

            attempts += 1

        return " ".join(line_words)

    def generate_haiku(
        self,
        words_by_syllables: Dict[int, List[str]],
        preserve_sentiment: bool = True,
        source_sentiment: Optional[str] = None,
    ) -> List[str]:
        """Generate a haiku (5-7-5 syllable pattern)"""
        pattern = [5, 7, 5]
        lines = []

        sentiment = source_sentiment if preserve_sentiment else None

        for target_syllables in pattern:
            line = self.generate_line(
                target_syllables, words_by_syllables, sentiment=sentiment
            )
            lines.append(line)

        return lines

    def generate_limerick(
        self,
        words_by_syllables: Dict[int, List[str]],
        use_rhymes: bool = True,
        preserve_sentiment: bool = True,
        source_sentiment: Optional[str] = None,
    ) -> List[str]:
        """Generate a limerick (AABBA rhyme scheme)"""
        pattern = [8, 8, 5, 5, 8]  # Approximate syllable counts
        lines = []

        sentiment = source_sentiment if preserve_sentiment else None

        # Generate first line
        line1 = self.generate_line(pattern[0], words_by_syllables, sentiment=sentiment)
        lines.append(line1)

        # Get last word for rhyming
        last_word1 = line1.split()[-1] if line1.split() else ""

        # Generate second line (rhymes with first)
        if use_rhymes and last_word1:
            line2 = self.generate_line(
                pattern[1],
                words_by_syllables,
                rhyme_with=last_word1,
                sentiment=sentiment,
            )
        else:
            line2 = self.generate_line(
                pattern[1], words_by_syllables, sentiment=sentiment
            )
        lines.append(line2)

        # Generate third line
        line3 = self.generate_line(pattern[2], words_by_syllables, sentiment=sentiment)
        lines.append(line3)

        # Generate fourth line (rhymes with third)
        last_word3 = line3.split()[-1] if line3.split() else ""
        if use_rhymes and last_word3:
            line4 = self.generate_line(
                pattern[3],
                words_by_syllables,
                rhyme_with=last_word3,
                sentiment=sentiment,
            )
        else:
            line4 = self.generate_line(
                pattern[3], words_by_syllables, sentiment=sentiment
            )
        lines.append(line4)

        # Generate fifth line (rhymes with first)
        if use_rhymes and last_word1:
            line5 = self.generate_line(
                pattern[4],
                words_by_syllables,
                rhyme_with=last_word1,
                sentiment=sentiment,
            )
        else:
            line5 = self.generate_line(
                pattern[4], words_by_syllables, sentiment=sentiment
            )
        lines.append(line5)

        return lines

    def generate_free_verse(
        self,
        words_by_syllables: Dict[int, List[str]],
        preserve_sentiment: bool = True,
        source_sentiment: Optional[str] = None,
    ) -> List[str]:
        """Generate free verse poetry"""
        num_lines = random.randint(4, 8)
        lines = []

        sentiment = source_sentiment if preserve_sentiment else None

        for _ in range(num_lines):
            syllables = random.randint(4, 12)
            line = self.generate_line(
                syllables, words_by_syllables, sentiment=sentiment
            )
            lines.append(line)

        return lines

    def generate_poetry(
        self,
        text: str,
        poetry_type: str = "haiku",
        use_rhymes: bool = True,
        preserve_sentiment: bool = True,
    ) -> Dict:
        """Main poetry generation function"""
        # Extract words and analyze sentiment
        all_words = []
        if self.nlp:
            doc = self.nlp(text)
            all_words = [
                token.text.lower()
                for token in doc
                if token.is_alpha and len(token.text) > 1
            ]
        else:
            all_words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

        if len(all_words) < 3:
            return {"error": "Not enough words in text to generate poetry"}

        # Remove duplicates while preserving some variety
        unique_words = list(set(all_words))
        words_by_syllables = self.get_words_by_syllables(unique_words)

        # Analyze source sentiment
        source_sentiment_data = self.get_sentiment(text)
        source_sentiment = (
            source_sentiment_data["label"] if preserve_sentiment else None
        )

        # Generate poetry based on type
        if poetry_type == "haiku":
            poem = self.generate_haiku(
                words_by_syllables, preserve_sentiment, source_sentiment
            )
            rhyme_scheme = "none"
        elif poetry_type == "limerick":
            poem = self.generate_limerick(
                words_by_syllables, use_rhymes, preserve_sentiment, source_sentiment
            )
            rhyme_scheme = "AABBA" if use_rhymes else "none"
        elif poetry_type == "free_verse":
            poem = self.generate_free_verse(
                words_by_syllables, preserve_sentiment, source_sentiment
            )
            rhyme_scheme = "none"
        else:
            return {"error": f"Unknown poetry type: {poetry_type}"}

        # Calculate syllable counts for each line
        syllable_counts = [
            sum(self.count_syllables(word) for word in line.split()) for line in poem
        ]

        # Analyze generated poem sentiment
        poem_text = "\n".join(poem)
        poem_sentiment = self.get_sentiment(poem_text)

        return {
            "poem": poem,
            "metadata": {
                "type": poetry_type,
                "rhyme_scheme": rhyme_scheme,
                "syllable_counts": syllable_counts,
                "source_sentiment": source_sentiment_data,
                "poem_sentiment": poem_sentiment,
                "total_words_used": len(unique_words),
                "preserve_sentiment": preserve_sentiment,
                "use_rhymes": use_rhymes,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced poetry generation")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument(
        "--type",
        default="haiku",
        choices=["haiku", "limerick", "free_verse"],
        help="Type of poetry to generate",
    )
    parser.add_argument(
        "--use-rhymes",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use rhyming",
    )
    parser.add_argument(
        "--preserve-sentiment",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Preserve source sentiment",
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
        generator = EnhancedPoetryGenerator()
        result = generator.generate_poetry(
            text,
            poetry_type=args.type,
            use_rhymes=(args.use_rhymes == "true"),
            preserve_sentiment=(args.preserve_sentiment == "true"),
        )
    except Exception as e:
        print(f"Error generating poetry: {e}", file=sys.stderr)
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

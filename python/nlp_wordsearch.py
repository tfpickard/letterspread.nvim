#!/usr/bin/env python3
"""
Enhanced word search generation using NLP
Provides semantic grouping, named entity recognition, and difficulty analysis
"""

import sys
import json
import argparse
import re
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

try:
    import spacy
    import nltk
    from nltk.corpus import wordnet
    from nltk.chunk import ne_chunk
    import numpy as np
except ImportError as e:
    print(f"Error importing required libraries: {e}", file=sys.stderr)
    print("Install with: pip install spacy nltk numpy", file=sys.stderr)
    sys.exit(1)


class EnhancedWordSearchGenerator:
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()

    def setup_nltk(self):
        """Download required NLTK data"""
        required_data = [
            ("tokenizers/punkt", "punkt"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
            ("corpora/words", "words"),
            ("corpora/wordnet", "wordnet"),
        ]

        for data_path, data_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                nltk.download(data_name, quiet=True)

    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found", file=sys.stderr)
            self.nlp = None

    def extract_words_and_entities(
        self, text: str, min_length: int = 3, max_length: int = 12
    ) -> Dict:
        """Extract words and categorize them using NLP"""
        result = {
            "words": set(),
            "entities": defaultdict(list),
            "semantic_groups": defaultdict(list),
        }

        if self.nlp:
            doc = self.nlp(text)

            # Extract regular words
            for token in doc:
                if (
                    token.is_alpha
                    and min_length <= len(token.text) <= max_length
                    and not token.is_stop
                ):
                    result["words"].add(token.text.upper())

            # Extract named entities
            for ent in doc.ents:
                if min_length <= len(ent.text) <= max_length and ent.text.isalpha():
                    entity_type = ent.label_
                    result["entities"][entity_type].append(ent.text.upper())
                    result["words"].add(ent.text.upper())

            # Group words by semantic categories
            for token in doc:
                if (
                    token.is_alpha
                    and min_length <= len(token.text) <= max_length
                    and not token.is_stop
                ):

                    word = token.text.upper()
                    pos = token.pos_

                    # Categorize by part of speech and semantic meaning
                    if pos == "NOUN":
                        result["semantic_groups"]["NOUNS"].append(word)
                    elif pos == "VERB":
                        result["semantic_groups"]["VERBS"].append(word)
                    elif pos == "ADJ":
                        result["semantic_groups"]["ADJECTIVES"].append(word)
                    elif pos == "ADV":
                        result["semantic_groups"]["ADVERBS"].append(word)
        else:
            # Fallback using NLTK
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)

            for word, pos in pos_tags:
                if word.isalpha() and min_length <= len(word) <= max_length:
                    word_upper = word.upper()
                    result["words"].add(word_upper)

                    # Basic POS categorization
                    if pos.startswith("NN"):
                        result["semantic_groups"]["NOUNS"].append(word_upper)
                    elif pos.startswith("VB"):
                        result["semantic_groups"]["VERBS"].append(word_upper)
                    elif pos.startswith("JJ"):
                        result["semantic_groups"]["ADJECTIVES"].append(word_upper)
                    elif pos.startswith("RB"):
                        result["semantic_groups"]["ADVERBS"].append(word_upper)

            # Named entity recognition with NLTK
            try:
                tree = ne_chunk(pos_tags)
                for subtree in tree:
                    if hasattr(subtree, "label"):
                        entity_name = " ".join(
                            [token for token, pos in subtree.leaves()]
                        )
                        if (
                            entity_name.isalpha()
                            and min_length <= len(entity_name) <= max_length
                        ):
                            entity_type = subtree.label()
                            result["entities"][entity_type].append(entity_name.upper())
                            result["words"].add(entity_name.upper())
            except:
                pass

        return result

    def get_semantic_categories(self, words: List[str]) -> Dict[str, List[str]]:
        """Categorize words using WordNet semantic categories"""
        categories = defaultdict(list)

        for word in words:
            synsets = wordnet.synsets(word.lower())
            if synsets:
                # Get the most common synset
                synset = synsets[0]
                lexname = synset.lexname()

                # Map WordNet lexical categories to readable names
                category_map = {
                    "noun.animal": "ANIMALS",
                    "noun.plant": "PLANTS",
                    "noun.food": "FOOD",
                    "noun.person": "PEOPLE",
                    "noun.location": "PLACES",
                    "noun.artifact": "OBJECTS",
                    "noun.body": "BODY PARTS",
                    "noun.substance": "MATERIALS",
                    "verb.motion": "MOVEMENT",
                    "verb.communication": "COMMUNICATION",
                    "verb.creation": "CREATION",
                    "adj.all": "DESCRIPTIONS",
                    "adv.all": "MANNER",
                }

                category = category_map.get(lexname, "OTHER")
                categories[category].append(word)
            else:
                categories["OTHER"].append(word)

        return dict(categories)

    def create_empty_grid(self, size: int) -> List[List[str]]:
        """Create empty grid"""
        return [[" " for _ in range(size)] for _ in range(size)]

    def can_place_word(
        self,
        grid: List[List[str]],
        word: str,
        row: int,
        col: int,
        direction: Tuple[int, int],
    ) -> bool:
        """Check if word can be placed at position and direction"""
        dr, dc = direction
        size = len(grid)

        for i, char in enumerate(word):
            new_row = row + i * dr
            new_col = col + i * dc

            if new_row < 0 or new_row >= size or new_col < 0 or new_col >= size:
                return False

            existing = grid[new_row][new_col]
            if existing != " " and existing != char:
                return False

        return True

    def place_word(
        self,
        grid: List[List[str]],
        word: str,
        row: int,
        col: int,
        direction: Tuple[int, int],
    ) -> bool:
        """Place word in grid"""
        dr, dc = direction

        for i, char in enumerate(word):
            new_row = row + i * dr
            new_col = col + i * dc
            grid[new_row][new_col] = char

        return True

    def fill_random_letters(self, grid: List[List[str]]) -> None:
        """Fill empty spaces with random letters"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        size = len(grid)

        for i in range(size):
            for j in range(size):
                if grid[i][j] == " ":
                    grid[i][j] = random.choice(letters)

    def calculate_difficulty(self, words: List[str], grid_size: int) -> str:
        """Calculate puzzle difficulty based on various factors"""
        total_letters = sum(len(word) for word in words)
        grid_area = grid_size * grid_size
        density = total_letters / grid_area

        avg_word_length = sum(len(word) for word in words) / len(words)

        # Difficulty factors
        if density > 0.3 and avg_word_length > 6:
            return "HARD"
        elif density > 0.2 and avg_word_length > 4:
            return "MEDIUM"
        else:
            return "EASY"

    def select_diverse_words(
        self, all_words: Set[str], max_words: int, semantic_groups: Dict[str, List[str]]
    ) -> List[str]:
        """Select diverse set of words from different semantic categories"""
        if len(all_words) <= max_words:
            return list(all_words)

        selected = []
        used_words = set()

        # First, try to get words from each semantic category
        for category, words in semantic_groups.items():
            if len(selected) >= max_words:
                break

            available_words = [w for w in words if w not in used_words]
            if available_words:
                # Select 1-2 words from each category
                num_from_category = min(
                    2, len(available_words), max_words - len(selected)
                )
                category_selection = random.sample(available_words, num_from_category)
                selected.extend(category_selection)
                used_words.update(category_selection)

        # Fill remaining slots with random words
        remaining_words = [w for w in all_words if w not in used_words]
        remaining_slots = max_words - len(selected)

        if remaining_slots > 0 and remaining_words:
            additional = random.sample(
                remaining_words, min(remaining_slots, len(remaining_words))
            )
            selected.extend(additional)

        return selected

    def create_wordsearch(
        self,
        text: str,
        grid_min: int = 15,
        grid_max: int = 25,
        max_words: int = 20,
        use_semantic_groups: bool = True,
    ) -> Dict:
        """Main word search creation function"""
        # Extract words and analyze them
        word_data = self.extract_words_and_entities(text)
        all_words = word_data["words"]

        if len(all_words) < 3:
            return {"error": "Not enough suitable words found in text"}

        # Select diverse words
        if use_semantic_groups:
            # Combine semantic groups and entity groups
            all_groups = {**word_data["semantic_groups"], **word_data["entities"]}
            selected_words = self.select_diverse_words(all_words, max_words, all_groups)
        else:
            selected_words = random.sample(
                list(all_words), min(max_words, len(all_words))
            )

        # Determine grid size
        max_word_length = max(len(word) for word in selected_words)
        grid_size = max(
            grid_min, min(grid_max, max_word_length + 5, len(selected_words) + 10)
        )

        # Create grid and place words
        grid = self.create_empty_grid(grid_size)
        placed_words = []
        directions = [
            (0, 1),  # right
            (1, 0),  # down
            (1, 1),  # diagonal down-right
            (1, -1),  # diagonal down-left
            (0, -1),  # left
            (-1, 0),  # up
            (-1, -1),  # diagonal up-left
            (-1, 1),  # diagonal up-right
        ]

        # Shuffle words for random placement
        words_to_place = selected_words.copy()
        random.shuffle(words_to_place)

        for word in words_to_place:
            placed = False
            attempts = 0

            while not placed and attempts < 100:
                row = random.randint(0, grid_size - 1)
                col = random.randint(0, grid_size - 1)
                direction = random.choice(directions)

                if self.can_place_word(grid, word, row, col, direction):
                    self.place_word(grid, word, row, col, direction)
                    placed_words.append(word)
                    placed = True

                attempts += 1

        # Fill empty spaces
        self.fill_random_letters(grid)

        # Organize final semantic groups (only for placed words)
        final_semantic_groups = {}
        if use_semantic_groups:
            # Get semantic categories for placed words
            placed_word_categories = self.get_semantic_categories(placed_words)

            # Combine with original groups
            for category, words in word_data["semantic_groups"].items():
                placed_in_category = [w for w in words if w in placed_words]
                if placed_in_category:
                    final_semantic_groups[category] = placed_in_category

            # Add WordNet categories
            for category, words in placed_word_categories.items():
                if words and category != "OTHER":
                    if category in final_semantic_groups:
                        final_semantic_groups[category].extend(words)
                        final_semantic_groups[category] = list(
                            set(final_semantic_groups[category])
                        )
                    else:
                        final_semantic_groups[category] = words

        # Calculate difficulty
        difficulty = self.calculate_difficulty(placed_words, grid_size)

        result = {
            "grid": grid,
            "words": placed_words,
            "metadata": {
                "grid_size": grid_size,
                "total_words_placed": len(placed_words),
                "total_words_found": len(all_words),
                "difficulty": difficulty,
                "placement_success_rate": (
                    len(placed_words) / len(selected_words) if selected_words else 0
                ),
            },
        }

        if use_semantic_groups and final_semantic_groups:
            result["semantic_groups"] = final_semantic_groups

        return result


def main():
    parser = argparse.ArgumentParser(description="Enhanced word search generation")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--grid-min", type=int, default=15, help="Minimum grid size")
    parser.add_argument("--grid-max", type=int, default=25, help="Maximum grid size")
    parser.add_argument(
        "--max-words", type=int, default=20, help="Maximum words to include"
    )
    parser.add_argument(
        "--semantic-groups",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use semantic grouping",
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
        generator = EnhancedWordSearchGenerator()
        result = generator.create_wordsearch(
            text,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            max_words=args.max_words,
            use_semantic_groups=(args.semantic_groups == "true"),
        )
    except Exception as e:
        print(f"Error creating word search: {e}", file=sys.stderr)
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

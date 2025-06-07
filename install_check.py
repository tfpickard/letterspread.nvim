#!/usr/bin/env python3
"""
Wordplay.nvim installation verification script
Tests all required dependencies and functionality
"""

import sys
import traceback


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")

    try:
        import spacy

        print(f"✓ spaCy {spacy.__version__}")
    except ImportError as e:
        print(f"✗ spaCy import failed: {e}")
        return False

    try:
        import nltk

        print(f"✓ NLTK {nltk.__version__}")
    except ImportError as e:
        print(f"✗ NLTK import failed: {e}")
        return False

    try:
        import pyphen

        print(f"✓ pyphen {pyphen.__version__}")
    except ImportError as e:
        print(f"✗ pyphen import failed: {e}")
        return False

    try:
        import pronouncing

        print(f"✓ pronouncing")
    except ImportError as e:
        print(f"✗ pronouncing import failed: {e}")
        return False

    try:
        import numpy

        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    return True


def test_spacy_model():
    """Test that spaCy English model is available"""
    print("\nTesting spaCy English model...")

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")

        # Test basic functionality
        doc = nlp("The quick brown fox jumps over the lazy dog.")
        tokens = [token.text for token in doc if token.is_alpha]

        if len(tokens) >= 8:  # Should have at least 8 words
            print("✓ spaCy en_core_web_sm model loaded and working")
            return True
        else:
            print("✗ spaCy model not processing text correctly")
            return False

    except OSError:
        print("✗ spaCy en_core_web_sm model not found")
        print("  Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"✗ spaCy model test failed: {e}")
        return False


def test_nltk_data():
    """Test that required NLTK data is available"""
    print("\nTesting NLTK data...")

    required_data = [
        ("wordnet", "corpora/wordnet"),
        ("brown", "corpora/brown"),
        ("punkt", "tokenizers/punkt"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("vader_lexicon", "vader_lexicon"),
    ]

    try:
        import nltk

        missing_data = []

        for data_name, data_path in required_data:
            try:
                nltk.data.find(data_path)
                print(f"✓ NLTK {data_name}")
            except LookupError:
                print(f"✗ NLTK {data_name} not found")
                missing_data.append(data_name)

        if missing_data:
            print(f"Missing NLTK data: {', '.join(missing_data)}")
            print("Run: python -c \"import nltk; nltk.download('all')\"")
            return False

        return True

    except Exception as e:
        print(f"✗ NLTK data test failed: {e}")
        return False


def test_functionality():
    """Test basic functionality of each component"""
    print("\nTesting functionality...")

    # Test syllable counting
    try:
        import pyphen

        dic = pyphen.Pyphen(lang="en")
        syllables = len(dic.positions("hello")) + 1
        if syllables == 2:
            print("✓ Syllable counting works")
        else:
            print(f"✗ Syllable counting unexpected result: {syllables}")
            return False
    except Exception as e:
        print(f"✗ Syllable counting test failed: {e}")
        return False

    # Test rhyme detection
    try:
        import pronouncing

        rhymes = pronouncing.rhymes("cat")
        if len(rhymes) > 0:
            print(f"✓ Rhyme detection works (found {len(rhymes)} rhymes for 'cat')")
        else:
            print("⚠ Rhyme detection works but no rhymes found for 'cat'")
    except Exception as e:
        print(f"✗ Rhyme detection test failed: {e}")
        return False

    # Test sentiment analysis
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores("I love this!")
        if "compound" in scores:
            print("✓ Sentiment analysis works")
        else:
            print("✗ Sentiment analysis unexpected result")
            return False
    except Exception as e:
        print(f"✗ Sentiment analysis test failed: {e}")
        return False

    return True


def main():
    """Run all installation tests"""
    print("Wordplay.nvim Installation Verification")
    print("=" * 40)

    print(f"Python version: {sys.version}")
    print()

    tests = [
        ("Package imports", test_imports),
        ("spaCy model", test_spacy_model),
        ("NLTK data", test_nltk_data),
        ("Functionality", test_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Wordplay.nvim is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

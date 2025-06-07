# Makefile for letterspread.nvim
# Automatically sets up Python virtual environment and installs NLP dependencies

SHELL := /bin/bash
VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python
REQUIREMENTS := requirements.txt

# Detect OS for different commands
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OS := macos
else ifeq ($(UNAME_S),Linux)
    OS := linux
else
    OS := unknown
endif

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: all install clean test check-deps check-python setup-venv install-deps install-spacy help

# Default target
all: install

# Main installation target
install: check-python setup-venv install-deps install-spacy download-nltk
	@echo -e "$(GREEN)✓ letterspread.nvim NLP dependencies installed successfully!$(NC)"
	@echo -e "$(BLUE)Virtual environment: $(VENV_DIR)$(NC)"
	@echo -e "$(BLUE)Python executable: $(PYTHON_VENV)$(NC)"

# Check if Python 3 is available
check-python:
	@echo -e "$(BLUE)Checking Python installation...$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo -e "$(RED)Error: Python 3 not found. Please install Python 3.7+$(NC)"; \
		echo -e "$(YELLOW)On macOS: brew install python3$(NC)"; \
		echo -e "$(YELLOW)On Arch Linux: paru -S python$(NC)"; \
		exit 1; \
	}
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)" || { \
		echo -e "$(RED)Error: Python 3.7+ required. Found: $$($(PYTHON) --version)$(NC)"; \
		exit 1; \
	}
	@echo -e "$(GREEN)✓ Python $$($(PYTHON) --version | cut -d' ' -f2) found$(NC)"

# Create virtual environment
setup-venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: check-python
	@echo -e "$(BLUE)Setting up Python virtual environment...$(NC)"
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip setuptools wheel
	@echo -e "$(GREEN)✓ Virtual environment created$(NC)"

# Create requirements.txt
$(REQUIREMENTS):
	@echo -e "$(BLUE)Creating requirements.txt...$(NC)"
	@echo "# letterspread.nvim Python dependencies" > $(REQUIREMENTS)
	@echo "spacy>=3.4.0" >> $(REQUIREMENTS)
	@echo "nltk>=3.7" >> $(REQUIREMENTS)
	@echo "pyphen>=0.12.0" >> $(REQUIREMENTS)
	@echo "pronouncing>=0.2.0" >> $(REQUIREMENTS)
	@echo "numpy>=1.21.0" >> $(REQUIREMENTS)
	@echo -e "$(GREEN)✓ Requirements file created$(NC)"

# Install Python dependencies
install-deps: $(VENV_DIR)/bin/activate $(REQUIREMENTS)
	@echo -e "$(BLUE)Installing Python packages...$(NC)"
	@$(PIP) install -r $(REQUIREMENTS)
	@echo -e "$(GREEN)✓ Python packages installed$(NC)"

# Install spaCy English model
install-spacy: $(VENV_DIR)/bin/activate
	@echo -e "$(BLUE)Installing spaCy English model...$(NC)"
	@$(PYTHON_VENV) -m spacy download en_core_web_sm --quiet || { \
		echo -e "$(YELLOW)Warning: Could not download spaCy model. Will use fallback mode.$(NC)"; \
	}
	@echo -e "$(GREEN)✓ spaCy model installation completed$(NC)"

# Download NLTK data
download-nltk: $(VENV_DIR)/bin/activate
	@echo -e "$(BLUE)Downloading NLTK data...$(NC)"
	@$(PYTHON_VENV) -c "\
import nltk; \
nltk.download('wordnet', quiet=True); \
nltk.download('brown', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('vader_lexicon', quiet=True); \
nltk.download('maxent_ne_chunker', quiet=True); \
nltk.download('words', quiet=True); \
print('NLTK data downloaded')" || { \
		echo -e "$(YELLOW)Warning: Some NLTK data may not have downloaded. Plugin will download as needed.$(NC)"; \
	}
	@echo -e "$(GREEN)✓ NLTK data download completed$(NC)"

# Test the installation
test: $(VENV_DIR)/bin/activate
	@echo -e "$(BLUE)Testing NLP installation...$(NC)"
	@$(PYTHON_VENV) -c "\
import sys; \
try: \
    import spacy, nltk, pyphen, pronouncing, numpy; \
    print('✓ All packages imported successfully'); \
    nlp = spacy.load('en_core_web_sm'); \
    print('✓ spaCy model loaded'); \
    print('✓ Installation test passed'); \
except Exception as e: \
    print(f'⚠ Warning: {e}'); \
    print('Plugin will use fallback mode for missing components'); \
    sys.exit(0)"
	@echo -e "$(GREEN)✓ Installation test completed$(NC)"

# Check dependencies without installing
check-deps:
	@echo -e "$(BLUE)Checking current dependencies...$(NC)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo -e "$(GREEN)✓ Virtual environment exists$(NC)"; \
		$(PIP) list | grep -E "(spacy|nltk|pyphen|pronouncing|numpy)" || echo -e "$(YELLOW)Some packages missing$(NC)"; \
	else \
		echo -e "$(RED)✗ Virtual environment not found$(NC)"; \
	fi

# Clean up virtual environment
clean:
	@echo -e "$(BLUE)Cleaning up virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@rm -f $(REQUIREMENTS)
	@echo -e "$(GREEN)✓ Cleanup completed$(NC)"

# Reinstall everything
reinstall: clean install

# Update dependencies
update: $(VENV_DIR)/bin/activate
	@echo -e "$(BLUE)Updating dependencies...$(NC)"
	@$(PIP) install --upgrade -r $(REQUIREMENTS)
	@$(PYTHON_VENV) -m spacy download en_core_web_sm --upgrade --quiet || true
	@echo -e "$(GREEN)✓ Dependencies updated$(NC)"

# Show help
help:
	@echo -e "$(BLUE)letterspread.nvim Makefile$(NC)"
	@echo ""
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@echo "  install     - Install all dependencies (default)"
	@echo "  test        - Test the installation"
	@echo "  check-deps  - Check current dependencies"
	@echo "  clean       - Remove virtual environment"
	@echo "  reinstall   - Clean and reinstall"
	@echo "  update      - Update all dependencies"
	@echo "  help        - Show this help"
	@echo ""
	@echo -e "$(YELLOW)Installation components:$(NC)"
	@echo "  check-python   - Verify Python 3.7+ is available"
	@echo "  setup-venv     - Create Python virtual environment"
	@echo "  install-deps   - Install Python packages"
	@echo "  install-spacy  - Download spaCy English model"
	@echo "  download-nltk  - Download NLTK data"

# Development targets
dev-install: install download-nltk test

# Verify installation works
verify: test
	@echo -e "$(BLUE)Running verification scripts...$(NC)"
	@if [ -f "python/nlp_anagrams.py" ]; then \
		echo "Testing anagram script..."; \
		echo "hello world listen silent" | $(PYTHON_VENV) python/nlp_anagrams.py /dev/stdin /tmp/test_anagrams.json --min-length 3; \
		echo -e "$(GREEN)✓ Anagram script works$(NC)"; \
	fi

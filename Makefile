.PHONY: clean data features models train predict test lint format help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Install development dependencies
dev-requirements:
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev]"

## Install the package in development mode
install:
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete

## Process raw data into processed data
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Generate features from processed data
features: data
	$(PYTHON_INTERPRETER) src/features/build_features.py

## Train models
train: features
	$(PYTHON_INTERPRETER) src/models/train_model.py

## Make predictions with trained models
predict: train
	$(PYTHON_INTERPRETER) src/models/predict_model.py

## Run tests
test:
	$(PYTHON_INTERPRETER) -m pytest tests/

## Lint using flake8
lint:
	flake8 src tests

## Format code using black
format:
	black src tests
	isort src tests

## Set up project
setup: requirements
	mkdir -p models
	mkdir -p data/processed
	mkdir -p data/external

## Run all steps
all: setup data features train predict

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Train readmission model
train-readmission: features
	$(PYTHON_INTERPRETER) src/models/train_model.py --model readmission

## Train mortality model
train-mortality: features
	$(PYTHON_INTERPRETER) src/models/train_model.py --model mortality

## Train length of stay model
train-los: features
	$(PYTHON_INTERPRETER) src/models/train_model.py --model los

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

## Show help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for(i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}'

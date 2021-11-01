.PHONY: help install docs format check test check-and-test


LIB_NAME = t-SVGP
SRC_NAME = src
TESTS_NAME = tests
LINT_NAMES = $(SRC_NAME) $(TESTS_NAME) notebooks
TYPE_NAMES = $(SRC_NAME)
SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)

LINT_FILE_IGNORES = "$(LIB_NAME)/$(SRC_NAME)/__init__.py:F401,F403"


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
	pip install --upgrade --upgrade-strategy eager \
		-r tests_requirements.txt \
		-r notebook_requirements.txt \
		tensorflow${VERSION_TF} \
		-e .


format: ## Formats code with `black` and `isort`
	@echo "\n=== isort =============================================="
	isort .
	@echo "\n=== black =============================================="
	black --line-length=100 $(LINT_NAMES)



test: ## Run unit and integration tests with pytest
	pytest --cov=$(SRC_NAME) \
	       --cov-report html:cover_html \
	       --cov-config .coveragerc \
	       --cov-report term \
	       --cov-report xml \
	       --cov-fail-under=80 \
	       --junitxml=reports/junit.xml \
	       -v --tb=short --durations=10 \
	       $(TESTS_NAME)

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

docs:  ## Build the documentation
	@echo "\n=== pip install doc requirements =============="
	pip install -r docs/docs_requirements.txt
	@echo "\n=== install pandoc =============="
ifeq ("$(UNAME_S)", "Linux")
	$(eval TEMP_DEB=$(shell mktemp))
	@echo "Checking for pandoc installation..."
	@(which pandoc) || ( echo "\nPandoc not found." \
	  && echo "Trying to install automatically...\n" \
	  && wget -O "$(TEMP_DEB)" $(PANDOC_DEB) \
	  && echo "\nInstalling pandoc using dpkg -i from $(PANDOC_DEB)" \
	  && echo "(If this step does not work, manually install pandoc, see http://pandoc.org/)\n" \
	  && sudo dpkg -i "$(TEMP_DEB)" \
	)
	@rm -f "$(TEMP_DEB)"
endif
ifeq ($(UNAME_S),Darwin)
	brew install pandoc
endif
	@echo "\n=== build docs =============="
	(cd docs ; make html)
	@echo "\n${SUCCESS}=== Docs are available at docs/_build/html/index.html ============== ${SUCCESS}"


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

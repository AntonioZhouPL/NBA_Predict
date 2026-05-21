.PHONY: install-dev lint test check download-data prepare-data evaluate baseline reproduce clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
NBA_PREDICT ?= $(VENV)/bin/nba-predict
SEASON ?= 2022-23
SEASON_START ?= 2013
SEASON_END ?= 2023
CV_FOLDS ?= 3
RAW ?= data/raw/2012_2023_Data.csv
DESIGN_MATRIX ?= data/processed/design_matrix.csv
MODEL ?= logistic
REPRO_DESIGN_MATRIX ?= data/reproducibility/design_matrix_snapshot.csv
REPRO_EXPECTED ?= data/reproducibility/expected_metrics.json
REPRO_REQUIREMENTS ?= requirements-lock.txt
REPRO_TOLERANCE ?= 1e-9

install-dev:
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PYTHON) -m pip install -r $(REPRO_REQUIREMENTS)
	$(PYTHON) -m pip install -e . --no-deps

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) -m compileall src
	$(MAKE) test
	$(MAKE) lint

download-data:
	$(NBA_PREDICT) download-data \
		--season-start $(SEASON_START) \
		--season-end $(SEASON_END) \
		--output $(RAW)

prepare-data:
	$(NBA_PREDICT) prepare-data \
		--raw $(RAW) \
		--output $(DESIGN_MATRIX)

evaluate:
	$(NBA_PREDICT) evaluate \
		--model $(MODEL) \
		--season $(SEASON) \
		--design-matrix $(DESIGN_MATRIX) \
		--cv-folds $(CV_FOLDS)

baseline:
	$(NBA_PREDICT) run-baseline \
		--season $(SEASON) \
		--design-matrix $(DESIGN_MATRIX) \
		--cv-folds $(CV_FOLDS)

reproduce:
	$(MAKE) baseline \
		SEASON=$(SEASON) \
		CV_FOLDS=$(CV_FOLDS) \
		DESIGN_MATRIX=$(REPRO_DESIGN_MATRIX)
	$(PYTHON) -m nba_predict.reproducibility \
		--expected $(REPRO_EXPECTED) \
		--metrics-dir outputs/metrics \
		--season $(SEASON) \
		--tolerance $(REPRO_TOLERANCE)

clean:
	rm -rf .pytest_cache .ruff_cache build dist
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +

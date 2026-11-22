PY := .venv/bin/python
PYTHON ?= python3.12

.PHONY: install test test-fast acceptance e2e fetch serve demo lint

install:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

# Unit tests (tests/unit; acceptance and e2e are excluded by addopts)
test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow and not acceptance and not e2e"

# Real data + trained models; only meaningful after T24
acceptance:
	OUTRUN_ACCEPTANCE=1 $(PY) -m pytest tests/acceptance -m acceptance

# Playwright offline-demo test (needs `$(PY) -m playwright install chromium` once)
e2e:
	$(PY) -m pytest tests/e2e -m e2e

# Resumable downloads from data/manifest.yaml (long: run in the background with a log)
fetch:
	mkdir -p logs
	$(PY) -m outrun fetch 2>&1 | tee logs/fetch.log

serve:
	$(PY) -m outrun serve

# Offline-safe demo: precomputed scenarios/, local PMTiles, no network
demo:
	OUTRUN_DEMO=1 $(PY) -m outrun serve

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

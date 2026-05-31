.PHONY: install train serve test lint typecheck docker clean

install:          ## Install package + dev tooling (editable)
	pip install -e ".[dev]"

train:            ## Train the model and write models/ artifacts
	python -m maternal_risk.train

serve:            ## Run the API + frontend (http://127.0.0.1:8000)
	python -m uvicorn maternal_risk.api:app --reload

test:             ## Run the test suite
	pytest

lint:             ## Lint with ruff
	ruff check src tests

typecheck:        ## Static type check with mypy
	mypy src

docker:           ## Build the container image
	docker build -t maternal-risk:latest .

clean:            ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info

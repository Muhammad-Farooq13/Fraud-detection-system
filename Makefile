.PHONY: help install install-dev data eda features train test lint format serve docker-build docker-run clean

PYTHON := python3

help:
	@echo "Available targets:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  data          Generate the synthetic transaction dataset"
	@echo "  eda           Run exploratory data analysis"
	@echo "  features      Build the model-ready feature set"
	@echo "  train         Train and evaluate models, save best artifact"
	@echo "  pipeline      Run data -> eda -> features -> train end-to-end"
	@echo "  test          Run the test suite"
	@echo "  lint          Run ruff + black --check"
	@echo "  format        Auto-format code with black + ruff --fix"
	@echo "  serve         Run the FastAPI server locally"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-run    Run the API via docker-compose"
	@echo "  clean         Remove generated artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

data:
	$(PYTHON) -m src.data.generate_data

eda:
	$(PYTHON) -m src.data.eda

features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train

pipeline: data eda features train

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t fraud-detection-api:latest .

docker-run:
	docker compose up --build

clean:
	rm -rf data/raw/*.csv data/processed/*.parquet models_store mlruns.db .pytest_cache **/__pycache__

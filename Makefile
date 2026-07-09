.PHONY: install dev test lint typecheck docker clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check --select E9,F63,F7,F82 codigo tests

docker:
	docker build -t llm-gnn-aml-explainability .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +

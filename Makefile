.PHONY: lint fmt test

lint:
	ruff .

fmt:
	isort .
	black .
	ruff --fix .

test:
	pytest -q
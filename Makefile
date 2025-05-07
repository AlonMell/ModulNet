.PHONY: lint fmt test

lint:
	ruff .

fmt:
	ruff --fix .

test:
	pytest -q
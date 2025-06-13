.PHONY: format

format:
	uvx ruff check --fix . && uvx ruff format .

# list recipes
default:
    @just --list

# format and apply safe lint fixes
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# check formatting, lint and types
lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check

# run the test suite
test:
    uv run pytest

# everything CI runs
ci: lint test

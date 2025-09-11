This project uses `uv` with a virtual environment located in `.venv`.
Use zsh when running commands.

This project is a jupyter notebook, with support for papermill.


To run tests use the Makefile commands:

- `make test` : run pytest with code coverage
- `make lint` : check mypy linting errors
- `make fix` : fix linting errors if possible

To perform a simple smoke test run `./run_papermill.sh sample_one_job.yaml`.

# jupyter-notebook-retirement-portfolio-modeler

A jupyter notebook that models a mixed set of assets over time using monte carlo simulations

# Usage

## Jupyter Notebook

You can run jupyter lab and interact with this notebook directly:

```sh
make setup
source .venv/bin/activate
jupyter lab
```

## Papermill

You can also create your own configuration scenarios and run them in batch using papermill. There is a script to automate this process:

```sh
./run_papermill.sh sample_papermill_settings.yaml

open notebooks/asset_modeler_sample_papermill_settings.html
```

## Development


You can run tests and linting using the instructions below:

```sh
make test

source .venv/bin/activate

# check ruff and fix issues:
make lint
make fix

# check mypy types:
make type
```

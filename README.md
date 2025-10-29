# jupyter-notebook-retirement-portfolio-modeler

A jupyter notebook that models retirement scenarios monte carlo simulations

[Example notebook run](./notebooks/asset_modeler.ipynb)

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

## Simulation Configuration

This notebook has been configured to allow one to run simulations on different life stages. A configuration consists of:

- A list of [Life Phases](https://github.com/dsummersl/jupyter-notebook-retirement-portfolio-modeler/blob/62bccc041a92d5dde8eb150b8e32d4f879ea769a/notebooks/plugins/constants.py#L59-L67): Each phase allows you to specify the age of the future retiree, their income, expenses, and savings rate.
- For each Life Phase, [Asset Actions](https://github.com/dsummersl/jupyter-notebook-retirement-portfolio-modeler/blob/62bccc041a92d5dde8eb150b8e32d4f879ea769a/notebooks/plugins/constants.py#L50-L56) can be configured - specifying new assets, updates to portfolio mixes, buying/selling of property.

See [./sample_papermill_settings.yaml](https://www.url.com) for an example configuration file.

Notes:
- Asset dollar amounts in the configuration should all be specified in "today's dollars" - the model will adjust for inflation over time.

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

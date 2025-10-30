# Retirement Portfolio Modeler

A jupyter notebook that models retirement scenarios monte carlo simulations

[Example Simulation Run](./notebooks/asset_modeler.ipynb)

Use this tool to explore diverse "what if's":

* Model stopping work at 55 vs. 65 and see the probability of your portfolio's success.
* What's the impact of my real estate? Simulate buying a new home, selling your current one, or purchasing a rental property.
* Gig economy vs. full-time work? Compare a scenario with steady income against one with variable part-time income later in life.
* What if I get an inheritance? Instantly add a new asset mid-simulation and see how it accelerates your financial independence.
* Is my portfolio too risky (or too conservative)? Model changes to your investment mix over time, automatically de-risking as you approach retirement.

## Simulation Configuration

Define your life in **Phases**. Each phase has a start age, income, and expenses. Each phase also allows you to perform **Actions**, allowing you to change your assets over time. A configuration consists of:

- A list of [Life Phases](https://github.com/dsummersl/jupyter-notebook-retirement-portfolio-modeler/blob/62bccc041a92d5dde8eb150b8e32d4f879ea769a/notebooks/plugins/constants.py#L59-L67): Each phase allows you to specify the age of the future retiree, their income, expenses, and savings rate.
- For each Life Phase, a list of [Actions](https://github.com/dsummersl/jupyter-notebook-retirement-portfolio-modeler/blob/62bccc041a92d5dde8eb150b8e32d4f879ea769a/notebooks/plugins/constants.py#L50-L56) can be configured - specifying new assets, updates to portfolio mixes, buying/selling of property.

See [./sample_papermill_settings.yaml](./sample_papermill_settings.yaml) for an example configuration file.

Notes:
- Asset dollar amounts in the configuration should all be specified in "today's dollars" - the simulation adjusts for inflation over time.

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

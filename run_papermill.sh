#!/bin/bash

# nounset: undefined variable outputs error message, and forces an exit
set -u
# errexit: abort script at first error
set -e
# print command to stdout before executing it:
# set -x


if [ "$#" -ne 1 ]; then
    echo "Run the asset_modeler notebook with papermill, and export as HTML."
    echo ""
    echo "Usage: $0 <papermill yaml file>"
    exit 1
fi

YAML=$1

YAML_BASENAME=$(basename "$YAML" .yaml)
OUTPUT_NOTEBOOK="notebooks/asset_modeler_${YAML_BASENAME}.ipynb"
OUTPUT_HTML="notebooks/asset_modeler_${YAML_BASENAME}.html"

jupytext --to notebook notebooks/asset_modeler.py
papermill notebooks/asset_modeler.ipynb $OUTPUT_NOTEBOOK -p parameter_source "$YAML" -f "$YAML"
jupyter nbconvert $OUTPUT_NOTEBOOK --no-input --to html

echo "Output notebook: $OUTPUT_NOTEBOOK"
echo "Output report: $OUTPUT_HTML"

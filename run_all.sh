#!/usr/bin/env bash
# Run every statistical analysis in order.
#   ./run_all.sh <data folder> <results folder>
# Demo without patient data:
#   python examples/make_synthetic_data.py --out example_data && ./run_all.sh example_data results
set -euo pipefail
DATA="${1:-data}"
OUT="${2:-results}"
for script in analysis/0*.py; do
    echo "### ${script}"
    python "${script}" --data "${DATA}" --out "${OUT}"
done

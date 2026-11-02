#!/usr/bin/env bash
# Full pipeline: (optional) re-download data, backtest, figures, tests.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--fetch" ]]; then python scripts/fetch_data.py; fi
python scripts/run_backtest.py
python scripts/make_figures.py
python -m pytest -q

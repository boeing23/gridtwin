#!/usr/bin/env bash
# Full pipeline, from the repo root:  bash scripts/run_all.sh /path/to/dataset/01_aggregated_datasets
# Needs: GNU Octave (>= 7) or MATLAB, Python 3.10+ with requirements.txt. About 15 min on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?path to NLR dataset/01_aggregated_datasets}"
mkdir -p data/runs results

echo "1/5  Converting NLR traces to CSV"
python3 scripts/convert_traces.py "$DATASET"

oct() { octave --no-gui -q --eval "addpath('matlab'); addpath('octave'); $1"; }

echo "2/5  Headline twin day (seed 7, faults on) -> data/runs/dctwin_data.csv"
oct "SEED=7; NOFAULT=0; OUT='data/runs/dctwin_data'; run_day"

echo "3/5  Fault-free baseline + six training days (seeds 11-16)"
oct "SEED=7; NOFAULT=1; OUT='data/runs/day7nf'; run_day" &
for s in 11 12 13 14 15 16; do oct "SEED=$s; NOFAULT=1; OUT='data/runs/day$s'; run_day" > /dev/null & done
wait

echo "4/5  Predictor + flattening scenarios (re-runs the twin per scenario)"
(cd analysis && python3 analyze.py)

echo "5/5  Figures and report"
(cd analysis && python3 figures.py && python3 report.py)
echo "Done: results/report.html"

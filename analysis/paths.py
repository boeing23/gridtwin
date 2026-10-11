"""All paths are relative to the repo root, so the pipeline runs from any checkout."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
TRACES = DATA / "traces_csv"          # written by scripts/convert_traces.py
RUNS = DATA / "runs"                  # twin outputs (Octave/MATLAB)
RESULTS = REPO / "results"
FIG = RESULTS / "figures"
TWIN_CSV = RUNS / "dctwin_data.csv"           # seed 7, faults on: the headline twin day
TWIN_JOBS = RUNS / "dctwin_data_jobs.csv"

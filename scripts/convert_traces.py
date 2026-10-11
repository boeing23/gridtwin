"""Convert the NLR training parquets to plain CSV so Octave can read them.

Usage: python scripts/convert_traces.py /path/to/dataset/01_aggregated_datasets
Writes data/traces_csv/<row>.csv (time_s, power_W) and data/traces_csv/index.txt
(model,nodes,repeat,relative_path). MATLAB users can skip this and set
p.it.data_root to the 01_aggregated_datasets folder instead.
"""
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "data" / "traces_csv"
out.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(root / "training" / "metadata.csv", index_col=0)
rows = []
for i, r in meta.iterrows():
    d = pd.read_parquet(root / r["path_save"]).reset_index()
    d.iloc[:, [0, 1]].to_csv(out / f"{i:06d}.csv", header=False, index=False)
    rows.append(f"{r['model']},{r['nodes']},{r['repeat']},traces_csv/{i:06d}.csv")
(out / "index.txt").write_text("\n".join(rows) + "\n")
print(f"Wrote {len(rows)} traces to {out}")

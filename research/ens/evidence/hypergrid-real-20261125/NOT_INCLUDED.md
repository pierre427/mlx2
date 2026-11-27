# Not included in git

This evidence was copied from `output/ens-hypergrid-real-20261125/` on 2026-11-27. These files stayed out:

- `*.npy`, `*.npz`: activation and hidden-state dumps (about 4.7 GB). They can be regenerated with the `donor_*`, `reason_probe*` and `hypergrid_real.py` scripts here.
- `fluids.jsonl` (19 MB) and `paper_cb*.txt`: third-party arXiv paper text, the inputs for the KV-mixing run.
- `__pycache__/` and the `swapguard.tripped` marker.

The text files that are included were checked for credentials and private-host references before commit.

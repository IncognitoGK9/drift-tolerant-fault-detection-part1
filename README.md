# How Slow Is Too Slow? I. The Price of Tolerating Drift in Sequential Fault Detection

Code, seeds and outputs for the paper by **Zablon Ombiro and Bernard Shibwabo** (Strathmore University, Nairobi), submitted to *IEEE Transactions on Signal Processing*. Companion paper: [drift-tolerant-fault-detection-part2](https://github.com/IncognitoGK9/drift-tolerant-fault-detection-part2).

A monitor that tolerates normal drift of width `w` cannot detect a fault slower than `w`; just above that blind spot the minimax detection delay follows a cube-root law in the false-alarm level with the effective rate `r - sqrt(r w)`, attained by a scan over least-favourable-pair templates that needs neither `r` nor `w`.

## Contents
| Folder / file | Paper item |
|---|---|
| `sim/row1_final.py` → `row1.json`, `row1_final.out` | Table I, row 1 (exact nearest-normal-path quadratic program vs. Lemma 1) |
| `sim/check_T3_templates.py` → `.json`, `.out` | Table I, row 1 (templates identity, block length, efficiency) |
| `sim/run_rows23.py`, `run_row3_resume.py` → `rows23.json`; `row2_post.py` → `row2_final.json`; `row3_summary.json` | Table I, rows 2 and 4 (the law; rate-free vs. oracle scan) |
| `sim/sim_directed_scan.py` → `directed_scan.csv` | Table I, row 3 (the price of tolerance) |
| `sim/efficiency.py`, `sim/null_check.py` (+ outputs) | family efficiency; no-alarm check under normal paths |
| `sim/vtwin_sim.py` | Figure 1 (V-shaped twin) |
| `realdata/real_test.py` → `real_runs.csv`, `real_streams.csv` | Table II (faults injected into real streams; protocol in the script docstring) |
| `realdata/analyse_real.py` → `analysis_output.txt`; `analyse_q123.py` → `q123_output.txt` | Table II summaries |
| `realdata/care_daily.py` | builds the daily wind-turbine power-curve indices from CARE |

## Data
The real-data scripts use public data that are **not redistributed here**:
- DKA Solar Centre (DKASC), Alice Springs — https://dkasolarcentre.com.au
- NREL PVDAQ (system 2107) — https://openei.org/wiki/PVDAQ
- CARE wind-turbine data set (Gück et al., 2024)

Set the data paths at the top of `real_test.py` (`U = ...`) and `care_daily.py` (`OUT = ...`) to your local copies. The per-run results used in the paper are included (`real_runs.csv`), so the tables can be reproduced without the raw data.

## Requirements
Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn`. See `requirements.txt`. Scripts that use random numbers fix their seeds; the others are deterministic.

## Licence
MIT (code). The data remain under their providers' terms.

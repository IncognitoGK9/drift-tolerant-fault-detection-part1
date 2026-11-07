# Drift-Tolerant Fault Detection — Part I: the price of tolerating drift

Code, fixed seeds and saved outputs for the paper

> **Z. Ombiro and B. Shibwabo, "How Slow Is Too Slow? I. The Price of Tolerating Drift in Sequential Fault Detection,"** submitted to *IEEE Transactions on Signal Processing*, 2026.
> Strathmore University, Nairobi, Kenya. Corresponding author: Zablon Ombiro (zablon.ombiro@strathmore.edu).

Companion paper and code: **Part II** — level jumps, slow faults and adaptive references — https://github.com/IncognitoGK9/drift-tolerant-fault-detection-part2

---

## 1. The problem in one paragraph

A monitor watches a noisy signal (for example the daily performance index of a photovoltaic system) whose *normal* behaviour drifts: seasons, soiling, ageing. To avoid false alarms it must tolerate normal drift up to a width `w` per step (falls up to `ρ−`, rises up to `ρ+`, `w = ρ− + ρ+`). A fault lowers the level at rate `r`. The paper asks: **how slow can a fault be before no tolerant detector can see it, and how long must it wait above that limit?**

## 2. What the paper shows (and what this code checks)

| Result | Statement (informal) | Checked by |
|---|---|---|
| **Blind spot** (Thm 1) | A fault with `r ≤ w` is indistinguishable from normal drift: every admissible detector waits the whole false-alarm horizon. | proof; real-data alarm rates (`realdata/`) |
| **Evidence geometry** (Lemma 1) | The normal path hardest to tell from the fault is V-shaped and starts drifting *before* the fault; the evidence after `k` steps is `E(k) ≥ ⅓ r_eff² k³` with the **effective rate** `r_eff = r − √(r w)` (not the naive margin `r − w`). | `sim/row1_final.py`, `sim/check_T3_templates.py` |
| **Minimax delay law** (Thm 2) | Every admissible detector has worst-case delay at least `k* = (6 log γ / (I r_eff²))^{1/3}` (γ = false-alarm horizon, `I` = information per step). | `sim/run_rows23.py`, `sim/row2_post.py` |
| **Price of tolerance** (Cor. 1) | Relative to a fixed-baseline detector the delay grows by `(1 − √(w/r))^{−2/3}` — 3.10 at `r = 1.5w`, 1.59 at `4w`. | `sim/sim_directed_scan.py` |
| **Attaining the law without knowing `r` or `w`** (Thm 3) | A scan over least-favourable-pair templates (isotonic projections of kinked ramps) is admissible at every γ and attains `k*` up to the family's efficiency `η_F`. | `sim/run_rows23.py`, `sim/run_row3_resume.py`, `sim/null_check.py`, `sim/efficiency.py` |
| **Real data** (Table II) | Faults injected into 30 photovoltaic and 50 wind-turbine streams: the effective-rate formula predicts the measured delay to within 2% on PV (ratio 0.98–1.01) where the margin formula is 12–49% off; on wind the measurements sit between the two. Failures are reported in the paper. | `realdata/real_test.py`, `realdata/analyse_real.py`, `realdata/analyse_q123.py` |

## 3. Repository layout

```
drift-tolerant-fault-detection-part1/
├── README.md            this file
├── LICENSE              MIT (code only)
├── requirements.txt     Python dependencies
├── sim/                 simulation checks (Table I, Figure 1)
│   ├── vtwin_sim.py               core module: V-twin geometry, exact QP for E(k), oracle and rate-free scans
│   ├── make_fig_vtwin.py          Figure 1: the V twin for r = 4w from the exact QP → fig_vtwin.png (deterministic)
│   ├── row1_final.py              Table I row 1: exact nearest-normal-path QP vs. Lemma 1 (27 cases)
│   │   → row1.json, row1_final.out
│   ├── check_T3_templates.py      Table I row 1: templates identity w²‖v_{j,q}‖² = E(j), block length, efficiency η_F
│   │   → check_T3_templates.json, .out
│   ├── run_rows23.py              Table I rows 2 and 4: the law near the boundary (r/w = 1.05…4) and oracle vs. rate-free scan
│   ├── run_row3_resume.py         resumable continuation of run_rows23.py (long runs)
│   │   → rows23.json
│   ├── row2_post.py               post-processing of rows23.json → row2_final.json (delay / k*, finite lower bound)
│   ├── row3_summary.json          summary of the rate-free vs. oracle comparison (Table I row 4)
│   ├── sim_directed_scan.py       Table I row 3: price of tolerance, drift arm vs. flat arm, γ ∈ {10⁴, 10⁶, 10¹²}
│   │   → directed_scan.csv
│   ├── efficiency.py              efficiency η_F of the 12-shape grid → efficiency_output.txt
│   └── null_check.py              no-fault alarm check under three normal paths (bound ¼) → null_check.out
└── realdata/            faults injected into real streams (Table II)
    ├── real_test.py               the test (protocol fixed before results; see docstring) → real_runs.csv, real_streams.csv
    ├── sim_directed_scan.py       template function used by real_test.py
    ├── analyse_real.py            delay law and blind-spot summaries → analysis_output.txt
    ├── analyse_q123.py            per-theorem summaries with bootstrap intervals → q123_output.txt
    └── care_daily.py              builds daily wind-turbine power-curve indices from the CARE data set
```

`real_runs.csv` (≈7.7 MB) holds one row per stream × base arm × threshold × fault speed × onset: alarm, delay, horizon, envelope width `w`, the run's own drift `a`, and the two predictions. **All tables can be rebuilt from it without the raw data.**

## 4. Reproducing the paper

```bash
git clone https://github.com/IncognitoGK9/drift-tolerant-fault-detection-part1
cd drift-tolerant-fault-detection-part1
python3 -m pip install -r requirements.txt
```

**Simulation (Table I, Figure 1)** — run from `sim/`:

| Command | What it produces | Run length |
|---|---|---|
| `python3 row1_final.py` | `row1.json`, row 1 errors of the closed form (≤ 0.93% with the correction term) | short |
| `python3 check_T3_templates.py` | templates identity to 10⁻⁹, block length, `η_F` across lags | < 1 min |
| `python3 run_rows23.py` (then `run_row3_resume.py` if interrupted), `python3 row2_post.py` | rows 2 and 4: delay / k* = 1.11–1.12 at γ = 10⁶ for every speed; rate-free cost +3.4–3.5% | long: 2,000 runs per point (resumable) |
| `python3 sim_directed_scan.py 2000` | `directed_scan.csv`: price within −1.2% … +0.1% of `(1 − √(w/r))^{−2/3}` | scales with the number of trials (2,000 per arm in the paper) |
| `python3 efficiency.py`, `python3 null_check.py` | efficiency of the shape grid; no false alarm in 900 normal runs at γ = 10³ | short |

**Real data (Table II)** — run from `realdata/`:

```bash
python3 analyse_real.py      # rebuilds analysis_output.txt from real_runs.csv
python3 analyse_q123.py      # rebuilds q123_output.txt
python3 real_test.py         # full re-run from raw data (set the data paths first, see §5)
```

The saved outputs in this repository are the ones reported in the paper; re-running reproduces them (simulations use fixed seeds; the real-data test is deterministic).

## 5. Data

The real-data scripts use public data that are **not redistributed here** (they remain under their providers' terms):

| Source | What is used | Where |
|---|---|---|
| DKA Solar Centre (DKASC), Alice Springs and Yulara | 7 systems: AS11 BP Solar, AS13 Trina, AS14 Kyocera, AS33 REC, YU1 Desert Gardens, YU2 Service Station, YU4 Laundry | https://dkasolarcentre.com.au |
| NREL PVDAQ | system 2107, 23 inverters | https://openei.org/wiki/PVDAQ |
| CARE wind-turbine data set (Gück et al., 2024) | 50 normal-labelled turbines, farms A–C | see the paper's references |

`real_test.py` expects one daily CSV per PV stream with columns `perf_index` (daily performance index) and `flag_bad_irradiance`; set `U = ...` at the bottom of the script to the folder holding them. `care_daily.py` builds the wind indices; set `OUT = ...` at its top. Per stream, the first 60% is the training block (noise scale, autocorrelation and drift envelope are estimated there and frozen) and the last 40% is the deployment block where faults of rate `r = k·w` are injected.

## 6. Requirements

Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn` (see `requirements.txt`). No GPU. Scripts that use random numbers fix their seeds; the others are deterministic.

## 7. Citation

```bibtex
@unpublished{ombiro2026howslowI,
  author = {Ombiro, Zablon and Shibwabo, Bernard},
  title  = {How Slow Is Too Slow? {I}. The Price of Tolerating Drift in Sequential Fault Detection},
  note   = {Submitted to IEEE Transactions on Signal Processing},
  year   = {2026}
}
```

## 8. Licence and contact

Code: MIT (see `LICENSE`). Data: their providers' terms. Questions: Zablon Ombiro, zablon.ombiro@strathmore.edu.

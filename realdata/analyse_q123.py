"""Q1-Q3 of HANDOVER_3A_SHORT_2026-11-07.md, computed from the existing real_runs.csv (no new runs).
Q1 (Theorem 1): alarm rates without a fault, at r=0.3w and at r=0.5w (the 'ramp' arm), 95% bootstrap CIs over streams.
Q2 (Theorem 2 / Corollary 1): measured/predicted delay for the effective-rate law and the naive margin law, per r/w.
Q3 (Theorem 3): measured delay / k*(a) with k* at level log(gamma) (the law's constant), and the false-alarm rate vs the budget."""
import numpy as np, pandas as pd
d = pd.read_csv("real_runs.csv"); rng = np.random.default_rng(0); B = 5000; GAMMA = 1e4
def boot(v):
    v = np.asarray(v, float); idx = rng.integers(0, len(v), (B, len(v))); m = v[idx].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
out = []
def p(s=""): print(s); out.append(s)
p("Q1 (Theorem 1, blind spot): alarm rate within the 900-day (PV) / 180-day (wind) horizon, natural base; mean over streams [95% bootstrap CI]")
for kind, g in d[d.arm == "natural"].groupby("kind"):
    for thr in ["theory", "calibrated"]:
        h = g[g.thr == thr]; rate = lambda k: h[h.k == k].groupby("stream").alarm.mean()
        none, k03, k05 = rate("none"), rate("0.3"), rate("ramp")
        a, b, c = boot(none), boot(k03), boot(k05); d3, d5 = boot((k03 - none).values), boot((k05 - none).values)
        p(f"  {kind:4s} {thr:10s} no fault {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]  r=0.3w {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}]  r=0.5w {c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]"
          f"  diff 0.3w {d3[0]:+.2f} [{d3[1]:+.2f},{d3[2]:+.2f}]  diff 0.5w {d5[0]:+.2f} [{d5[1]:+.2f},{d5[2]:+.2f}]")
    fl = g[g.k == "flat_noFA"].groupby("stream").alarm.mean(); p(f"  {kind:4s} fixed-baseline scan (no allowance), no fault: {fl.mean():.2f}")
p("  Note: r = w was not run in the 6 Nov 2026 test; only 0.3w and 0.5w (both inside the blind spot) exist.")
p("\nQ2 (Theorem 2 / Corollary 1): calibrated threshold, detected runs, medians; ratio = measured / predicted at the threshold's level u = x^2/2")
f = d[(d.thr == "calibrated") & d.k.isin(["1.5", "2.0", "3.0", "5.0", "10.0"])].copy(); f["k"] = f.k.astype(float); f = f[f.alarm == 1].copy()
f["lerr"] = np.log(f.delay.clip(lower=1) / f.pred_eff); f["lmar"] = np.log(f.delay.clip(lower=1) / f.pred_margin)
# k* at level log(gamma) (the constant of the law), from pred_eff at level u: k*_law = pred_eff * (log gamma / u)^(1/3)
f["u"] = f.x ** 2 / 2; f["kstar_law"] = f.pred_eff * (np.log(GAMMA) / f.u) ** (1 / 3); f["fac"] = f.delay.clip(lower=1) / f.kstar_law
for (kind, arm), g in f.groupby(["kind", "arm"]):
    p(f"  {kind} / {arm}: streams {g.stream.nunique()}, runs {len(g)}, median a/w {np.median(g.a/g.w):.2f}, median threshold x {g.x.median():.2f} (theory 6.22 PV)")
    for k, h in g.groupby("k"):
        p(f"    r/w={k:<4} delay {h.delay.median():5.1f}  pred_eff {h.pred_eff.median():5.1f}  pred_margin {h.pred_margin.median():5.1f}  ratio_eff {np.exp(h.lerr.median()):.2f}  ratio_margin {np.exp(h.lmar.median()):.2f}  | Q3: delay/k*(a;log g) median {h.fac.median():.2f} [IQR {h.fac.quantile(.25):.2f}-{h.fac.quantile(.75):.2f}]")
    s = g.groupby("stream").apply(lambda x: pd.Series({"e": np.abs(x.lerr).mean(), "m": np.abs(x.lmar).mean()}), include_groups=False)
    df = boot((s.m - s.e).values)
    p(f"    per-stream mean |log err|: eff {s.e.mean():.3f}, margin {s.m.mean():.3f}; margin - eff = {df[0]:+.3f} [{df[1]:+.3f}, {df[2]:+.3f}]  (>0 favours the effective rate)")
    p(f"    Q3 overall: delay/k*(a;log gamma) median {g.fac.median():.2f}, 5th-95th pct {g.fac.quantile(.05):.2f}-{g.fac.quantile(.95):.2f}; ratio of medians across r/w: {g.groupby('k').fac.median().max()/g.groupby('k').fac.median().min():.2f}")
p("\nQ3 (Theorem 3, admissibility): alarm rate without a fault per horizon, natural base, and the implied P(T <= 2 gamma) if alarms were geometric per horizon")
for kind, g in d[(d.arm == "natural") & (d.k == "none")].groupby("kind"):
    hz = g.horizon.median()
    for thr in ["theory", "calibrated"]:
        h = g[g.thr == thr]; r = h.groupby("stream").alarm.mean().mean(); mt = hz / max(r, 1e-9); pt = 1 - (1 - r) ** (2 * GAMMA / hz)
        p(f"  {kind:4s} {thr:10s} rate per {hz:.0f}-day horizon {r:.2f}; implied mean time to false alarm ~{mt:.0f} days (budget gamma = {GAMMA:.0f}); implied P(T<=2gamma) ~{pt:.2f} (budget <= 0.5)")
open("q123_output.txt", "w").write("\n".join(out) + "\n")

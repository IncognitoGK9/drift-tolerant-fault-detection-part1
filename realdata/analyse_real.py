"""Analysis of real_runs.csv (fixed before results). Table A: delay law. Table B: blind spot and false alarms."""
import numpy as np, pandas as pd
d = pd.read_csv("real_runs.csv"); rng = np.random.default_rng(0); B = 5000
def boot(stream_vals):
    v = np.array(stream_vals); idx = rng.integers(0, len(v), (B, len(v))); m = v[idx].mean(1); return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
print("TABLE A: delay law (calibrated threshold; detected runs; medians over runs; error = mean |log(measured/predicted)| per stream)")
f = d[(d.thr == "calibrated") & d.k.isin(["1.5", "2.0", "3.0", "5.0", "10.0"])].copy(); f["k"] = f.k.astype(float)
f = f[f.alarm == 1].copy(); f["lerr"] = np.log(f.delay.clip(lower=1) / f.pred_eff); f["lmar"] = np.log(f.delay.clip(lower=1) / f.pred_margin)
for (kind, arm), g in f.groupby(["kind", "arm"]):
    print(f"\n{kind} / {arm}: streams {g.stream.nunique()}, runs {len(g)}, a/w median {np.median(g.a / g.w):.2f}")
    for k, h in g.groupby("k"):
        tot = d[(d.thr == "calibrated") & (d.kind == kind) & (d.arm == arm) & (d.k == f"{k}")]
        print(f"  k={k:<4} det {len(h)/len(tot):.2f}  delay {h.delay.median():6.1f}  pred_eff {h.pred_eff.median():6.1f}  pred_margin {h.pred_margin.median():6.1f}  "
              f"ratio_eff {np.exp(h.lerr.median()):.2f}  ratio_margin {np.exp(h.lmar.median()):.2f}")
    s = g.groupby("stream").apply(lambda x: pd.Series({"e": np.abs(x.lerr).mean(), "m": np.abs(x.lmar).mean()}))
    diff = boot((s.m - s.e).values); hi = g[g.a / g.w >= 0.75]
    print(f"  mean |log err|: eff {s.e.mean():.3f}, margin {s.m.mean():.3f}; margin - eff = {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}] (>0 favours r_eff)")
    if len(hi): print(f"  runs with a/w >= 0.75: n={len(hi)}, median ratio_eff {np.exp(hi.lerr.median()):.2f}, ratio_margin {np.exp(hi.lmar.median()):.2f}")
print("\nTABLE B: blind spot and false alarms (natural base; alarm within the horizon)")
for kind, g in d[d.arm == "natural"].groupby("kind"):
    for thr in ["theory", "calibrated"]:
        h = g[g.thr == thr]; rate = lambda k: h[h.k == k].groupby("stream").alarm.mean()
        none, k03, ramp = rate("none"), rate("0.3"), rate("ramp"); eq = boot((k03 - none).values)
        print(f"{kind:4s} {thr:10s} no fault {none.mean():.2f}  k=0.3 {k03.mean():.2f}  ramp 0.5w {ramp.mean():.2f}  (k0.3 - none) {eq[0]:+.2f} [{eq[1]:+.2f}, {eq[2]:+.2f}] equivalent(+-0.10): {eq[1] > -0.10 and eq[2] < 0.10}")
    fl = g[g.k == "flat_noFA"].groupby("stream").alarm.mean(); print(f"{kind:4s} fixed-baseline scan, no fault: alarm rate {fl.mean():.2f}")

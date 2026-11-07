"""Real-data test of the effective-rate law (Paper 3a). Protocol fixed before any result was seen.
Streams: 30 daily PV performance-index streams (DKASC 7, PVDAQ 23) and 50 daily wind-turbine power-curve indices (CARE, normal-labelled).
Per stream: first 60% = training block, last 40% = deployment block. On the training block only (then frozen):
  winsorise at the 0.5/99.5% quantiles; residuals from a 31-day rolling median; sigma = 1.4826 MAD; lag-1 autocorrelation phi;
  long-run sd s = sigma sqrt((1+phi)/(1-phi)); envelope rho_-, rho_+ = -q01, q99 of OLS slopes over sliding windows of H days
  (PV 90, wind 30); w = rho_- + rho_+. Data are standardised by s (information I = 1 per day).
Deployment: linear faults r = k w (k in 0.3, 1.5, 2, 3, 5, 10), a benign ramp 0.5 w and no fault, injected at onsets every 30 (PV) / 10 (wind)
days. Detector: the rate-free least-favourable-pair scan (12 shapes, lags up to J), threshold sqrt(2 log(4 gamma |F|)), gamma = 1e4,
monitoring from the onset. Per run, the base drift a in cone coordinates is the OLS slope of the clean data after the onset plus rho_-,
clipped to [0, w]. Predictions: effective-rate law k*(a) with r_eff = r - sqrt(r a); naive-margin law with m = r - a; both at level
u = log(4 gamma |F|). A fixed-baseline scan (no drift allowance) is run on the no-fault deployment data to measure its false alarms."""
import numpy as np, pandas as pd, glob, os, sys, json, time
sys.argv = sys.argv[:1]; from sim_directed_scan import template
KAP = [1.25 * 1.26 ** i for i in range(12)]; GAMMA = 1e4; KS = [0.3, 1.5, 2.0, 3.0, 5.0, 10.0]
def load(f, kind):
    d = pd.read_csv(f)
    if kind == "pv": ok = d.perf_index.notna() & (~d.flag_bad_irradiance.fillna(False).astype(bool)); y = d.perf_index[ok].values
    else: y = d.perf_index.values
    return y[np.isfinite(y)].astype(float)
def train_est(y, H):
    n = int(0.6 * len(y)); tr = y[:n].copy(); lo, hi = np.quantile(tr, [0.005, 0.995]); y = np.clip(y, lo, hi); tr = y[:n]
    med = pd.Series(tr).rolling(31, center=True, min_periods=11).median().values; e = tr - med; e = e[np.isfinite(e)]
    sig = 1.4826 * np.median(np.abs(e - np.median(e))); phi = float(np.clip(np.corrcoef(e[:-1], e[1:])[0, 1], 0, 0.95))
    s = sig * np.sqrt((1 + phi) / (1 - phi)); t = np.arange(H) - (H - 1) / 2; den = (t ** 2).sum()
    sl = np.array([(tr[i:i + H] * t).sum() / den for i in range(0, n - H)])
    rm, rp = max(0.0, -np.quantile(sl, 0.01)), max(0.0, np.quantile(sl, 0.99))
    return y, n, s, phi, rm / s, rp / s
def families(J):
    F = []
    for kap in KAP:
        for j in range(1, J + 1):
            a, c, nv = template(j, kap); F.append((j, kap, a, c, nv))
    return np.array(F)
def stat_matrix(x, F):
    """Z[e, f] for all times e (template ending at e), -inf where the template does not fit"""
    T = len(x); idx = np.arange(T, dtype=float); S0 = np.r_[0, np.cumsum(x)]; S1 = np.r_[0, np.cumsum(x * idx)]
    e = np.arange(T); Z = np.full((T, len(F)), -np.inf)
    for f, (j, kap, a, c, nv) in enumerate(F):
        j = int(j); a = int(a); ok = (e + a >= 0) & (e - j >= 0); ee = e[ok]
        lo1, hi1, lo2 = ee + a, ee - j, ee - j + 1
        d1 = (S1[hi1 + 1] - S1[lo1]) + (-ee + j - c) * (S0[hi1 + 1] - S0[lo1])
        d2 = (1 - kap) * (S1[ee + 1] - S1[lo2]) + ((1 - kap) * (j - ee) - c) * (S0[ee + 1] - S0[lo2])
        Z[ok, f] = (d1 + d2) / nv
    return Z
def ramp_response(F, D):
    """R[d, f] = <-(t-onset)^+, v_f>/||v_f|| for a template ending d steps after the onset (unit rate), via suffix sums"""
    R = np.zeros((D + 1, len(F)))
    for f, (j, kap, a, c, nv) in enumerate(F):
        j = int(j); a = int(a); p = np.arange(a, 1); q = p + j; m = np.where(q <= 0, q, (1 - kap) * q).astype(float); v = m - c
        L = len(v); i = np.arange(L, dtype=float); Sv = np.cumsum(v[::-1])[::-1]; Siv = np.cumsum((i * v)[::-1])[::-1]
        d = np.arange(D + 1); qq = L - 1 - d; ok = qq >= 0; qv = qq[ok]
        R[ok, f] = -(Siv[qv] - qv * Sv[qv]) / nv
        R[~ok, f] = np.nan
    # for d beyond the template length the onset is before the window: ramp is linear on the window, <linear, v> = slope*<i, v>
        if (~ok).any():
            lin = -(Siv[0] - 0 * Sv[0]); R[~ok, f] = (-(Siv[0] - (L - 1 - d[~ok]) * Sv[0])) / nv
    return R
def flat_matrix(x, lags=range(1, 257), wins=(64, 128, 256, 512)):
    T = len(x); idx = np.arange(T, dtype=float); S0 = np.r_[0, np.cumsum(x)]; S1 = np.r_[0, np.cumsum(x * idx)]; e = np.arange(T); best = np.full(T, -np.inf)
    for W in wins:
        for j in lags:
            if j >= W: continue
            ok = e - W + 1 >= 0; ee = e[ok]; o = ee - j; Sm = j * (j + 1) / 2; nv = np.sqrt(j * (j + 1) * (2 * j + 1) / 6 - Sm ** 2 / W)
            dot = -((S1[ee + 1] - S1[o + 1]) - o * (S0[ee + 1] - S0[o + 1])) + (Sm / W) * (S0[ee + 1] - S0[ee - W + 1])
            best[ok] = np.maximum(best[ok], dot / nv)
    return best, len(wins) * 256
def run_stream(name, y, kind, F, R, x_thr, nF):
    H, step = (90, 30) if kind == "pv" else (30, 10)
    y, n, s, phi, rm, rp = train_est(y, H); w = rm + rp; z = y / s; T = len(z); t = np.arange(T); rows = []
    fb, nflat = flat_matrix(z); xf = np.sqrt(2 * np.log(4 * GAMMA * nflat))
    for arm, extra in [("natural", 0.0), ("worstbase", rp)]:
        zz = z + extra * t; Z0 = stat_matrix(zz + rm * t, F); Zm0 = Z0.max(1)
        x_cal = float(np.max(Zm0[:n][np.isfinite(Zm0[:n])]))          # training-calibrated threshold (no fault)
        for thr_name, x in [("theory", x_thr), ("calibrated", max(x_cal, x_thr * 0 + 1e-9))]:
            u = x ** 2 / 2; kst = lambda re: (6 * u / re ** 2) ** (1 / 3) if re > 0 else np.inf
            for nu in range(n + 1, T - 20, step):
                hz = min(T - 1, nu + R.shape[0] - 1); dd = np.arange(nu, hz + 1) - nu
                seg = zz[nu:min(T, nu + 120)]; sl = np.polyfit(np.arange(len(seg)), seg, 1)[0] if len(seg) > 10 else 0.0
                a = float(np.clip(sl + rm, 0, w))
                if arm == "natural" and thr_name == "theory":
                    rows.append(dict(stream=name, kind=kind, arm=arm, thr=thr_name, k="flat_noFA", onset=nu, alarm=int((fb[nu:hz + 1] > xf).any()), horizon=hz - nu))
                for k in ["none", "ramp"] + KS:
                    r = 0.0 if k == "none" else (0.5 * w if k == "ramp" else k * w)
                    hit = (Z0[nu:hz + 1] + r * R[dd]).max(1) > x
                    dl = int(dd[hit.argmax()]) if hit.any() else None
                    rows.append(dict(stream=name, kind=kind, arm=arm, thr=thr_name, x=x, k=k, onset=nu, alarm=int(hit.any()), delay=dl, horizon=hz - nu,
                                     w=w, a=a, r=r, pred_eff=kst(r - np.sqrt(r * a)) if r > 0 else np.nan, pred_margin=kst(r - a) if r > 0 else np.nan))
    return rows, dict(stream=name, kind=kind, n=T, ntrain=n, s=s, phi=phi, rho_m=rm, rho_p=rp, w=w)
if __name__ == "__main__":
    U = "/mnt/user-data/uploads/TWA_data"; files = [(f, "pv") for f in sorted(glob.glob(U + "/prepared/daily/*.csv"))] + [(f, "wind") for f in sorted(glob.glob(U + "/Claude outputs/p3a_care_daily/*.csv"))]
    lim = int(os.environ.get("LIMIT", "0")) or len(files); out = []; meta = []; t0 = time.time()
    cache = {}
    for f, kind in files[:lim]:
        J, D = (512, 900) if kind == "pv" else (150, 180)
        if kind not in cache:
            F = families(J); cache[kind] = (F, ramp_response(F, D), np.sqrt(2 * np.log(4 * GAMMA * len(F))), len(F))
        F, R, x_thr, nF = cache[kind]
        rows, m = run_stream(os.path.basename(f)[:-4], load(f, kind), kind, F, R, x_thr, nF); out += rows; meta.append(m)
        print(f"{m['stream'][:30]:30s} n={m['n']} w/s={m['w']:.4f} phi={m['phi']:.2f} rows={len(rows)} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(out).to_csv("real_runs.csv", index=False); pd.DataFrame(meta).to_csv("real_streams.csv", index=False)

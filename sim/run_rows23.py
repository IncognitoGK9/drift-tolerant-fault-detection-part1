import numpy as np, json, time, sys
from vtwin_sim import *
w = 0.02
gammas = [1e4, 1e6, 1e8]
res = {"w": w, "row2": [], "row3": [], "eta": []}
# ---------------- Row 2: near-boundary dynamics (oracle-shape scan) ----------------
for q in [1.05, 1.10, 1.25, 2.0, 4.0]:
    t0 = time.time()
    ks = {g: kstar(q * w, w, g) for g in gammas}
    lags = geometric_lags(int(1.8 * ks[1e8]), ratio=1.03)
    tm = build_templates(lags, [q])
    Pmax = max(t[2] for t in tm); H = Pmax + 200; Tmax = int(4 * ks[1e8])
    th = {g: gauss_threshold(g, len(tm)) for g in gammas}
    out, pre, cens = run_scan(tm, q, w, H, Tmax, 2000, 11, th)
    for g in gammas:
        d = np.array(out[g], float)
        lb, nlb, llb = finite_lower_bound(g, q * w, w)
        res["row2"].append(dict(q=q, gamma=g, kstar=ks[g], lb=lb, n_lb=nlb, mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))),
                                nruns=len(d), pre=pre[g], cens=cens[g], ntempl=len(tm)))
    print("row2 q", q, round(time.time() - t0, 1), "s", flush=True)
    json.dump(res, open("rows23.json", "w"), indent=1)
# ---------------- Row 3: grid density, oracle vs rate-free ----------------
def grid(g, b0=1.25, bmax=17.0):
    out, b = [], b0
    while b <= bmax: out.append(b); b *= g
    return out
iso = IsotonicRegression(increasing=True)
def eta_exact(q, j, betas):
    P = int(4 * j * np.sqrt(max(max(betas), q))) + 40
    muq = np.cumsum(np.where(np.arange(P) < P - j, 1.0, 1 - q))
    vq = muq - iso.fit_transform(np.arange(P), muq)
    best = 0.0
    for b in betas:
        mub = np.cumsum(np.where(np.arange(P) < P - j, 1.0, 1 - b))
        vb = mub - iso.fit_transform(np.arange(P), mub)
        best = max(best, float(muq @ vb) / (np.linalg.norm(vb) * np.linalg.norm(vq)))
    return best
GR = [1.10, 1.26, 1.6, 2.5]
for g in GR:
    bs = grid(g)
    row = {"g": g, "nshapes": len(bs)}
    for qq in [1.5, 2, 3, 5, 10]:
        jj = int(round(kstar(qq * w, w, 1e6)))
        row[f"eta_{qq}"] = eta_exact(qq, max(jj, 10), bs)
    # worst case eta over a fine q grid in [1.25, 16]
    qs = np.geomspace(1.25, 16, 25)
    row["eta_min"] = min(eta_exact(qq, 40, bs) for qq in qs)
    res["eta"].append(row)
    print("eta", g, round(row["eta_min"], 4), flush=True)
json.dump(res, open("rows23.json", "w"), indent=1)
for q in [2.0, 4.0]:
    ks = kstar(q * w, w, 1e6)
    lags = geometric_lags(int(1.8 * kstar(q * w, w, 1e8)), ratio=1.05)
    tm_or = build_templates(lags, [q])
    Pm = max(t[2] for t in tm_or)
    th_or = {1e6: gauss_threshold(1e6, len(tm_or))}
    Tmax = int(4 * kstar(q * w, w, 1e8))
    for g in GR:
        t0 = time.time()
        bs = grid(g)
        tm_rf = build_templates(lags, bs)
        Pmax = max(max(t[2] for t in tm_rf), Pm); H = Pmax + 200
        th_rf = {1e6: gauss_threshold(1e6, len(tm_rf))}
        o1, p1, c1 = run_scan(tm_or, q, w, H, Tmax, 600, 21, th_or)
        o2, p2, c2 = run_scan(tm_rf, q, w, H, Tmax, 600, 21, th_rf)
        d1 = np.array(o1[1e6], float); d2 = np.array(o2[1e6], float)
        n = min(len(d1), len(d2))
        # paired (same seed -> same noise); drop runs with pre-onset alarms in either by index alignment is not tracked, so use means
        res["row3"].append(dict(q=q, g=g, nshapes=len(bs), kstar=ks, oracle=float(d1.mean()), rate_free=float(d2.mean()),
                                se_o=float(d1.std(ddof=1) / np.sqrt(len(d1))), se_r=float(d2.std(ddof=1) / np.sqrt(len(d2))),
                                x_or=th_or[1e6], x_rf=th_rf[1e6], ntempl_rf=len(tm_rf), ntempl_or=len(tm_or), pre=(p1[1e6], p2[1e6])))
        print("row3 q", q, "g", g, round(time.time() - t0, 1), "s", flush=True)
        json.dump(res, open("rows23.json", "w"), indent=1)
print("DONE", flush=True)

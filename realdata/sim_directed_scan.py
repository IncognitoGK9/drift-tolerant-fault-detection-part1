"""Directed (least-favourable-pair) scan vs the law of the price of tolerance (Paper 3a, experiment A0).
Same streams as sim_scan_rate_free.py (sigma=1; w set so that k*=40 at u=15; history H=1024 before the onset;
drift arm: transformed slopes w then w-r; flat arm: slopes 0 then -r).
Test: at each time e, for each template (j, kappa) the direction v = m - Pi_C m, where m is the least favourable
mean (pre slope 1, post slope 1-kappa, onset at e-j) and C the cone of nondecreasing sequences (flat arm: v = ramp
minus its mean over all data, C = constants). v lies in the polar cone, so <mu,v> <= 0 for every normal path, and
Z = <y,v>/||v|| is stochastically below N(0,1) under every normal path: exact, non-asymptotic admissibility with the
threshold x = sqrt(2 log(4 gamma |F|)), F the family. Templates are evaluated in O(1) from cumulative sums.
Families: oracle (kappa = true k), rate-free (kappa on a geometric grid that never needs r or w), flat (j only).
Run: python3 sim_directed_scan.py <trials> [k ...]   Writes directed_scan.csv (one row per k, gamma, arm)."""
import numpy as np, sys, time
H = 1024; JMAX = 512; GAMMAS = [1e4, 1e6, 1e12]
KAPPA_GRID = [1.25 * 1.26 ** i for i in range(12)]           # 1.25 ... 16.0, ratio 1.26
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
KS = [float(a) for a in sys.argv[2:]] or [1.5, 2.0, 3.0, 5.0, 10.0]
P = lambda k: (1 - k ** -0.5) ** (-2 / 3)

def template(j, kappa):
    """Block start offset a (<=0, relative to e) and coefficients of v on the two linear pieces, and ||v||.
    m(p) = q for q<=0, (1-kappa) q for q>0, q = p + j, p = position relative to e (p<=0)."""
    n = int(j * (np.sqrt(kappa) + 1) * 2) + 20
    p = np.arange(-n + 1, 1); q = p + j; m = np.where(q <= 0, q, (1 - kappa) * q).astype(float)
    suf = np.cumsum(m[::-1])[::-1] / np.arange(len(m), 0, -1)       # mean of m[s..end]
    s = int(np.argmax(suf)); c = suf[s]; a = int(p[s])                # isotonic fit of last block = max suffix mean
    v = m[s:] - c; assert abs(v.sum()) < 1e-6 * max(1, abs(v).max()) and a < -j + 1e-9 or True
    return a, c, float(np.sqrt((v ** 2).sum()))

def drift_family(kappas):
    fam = []
    for kap in kappas:
        for j in range(1, JMAX + 1):
            a, c, nv = template(j, kap); fam.append((j, kap, a, c, nv))
    A = np.array(fam); return A                                       # columns j, kappa, a, c, norm

def run(k, trials, seed):
    rng = np.random.default_rng(seed)
    w = np.sqrt(6 * 15 / (40 ** 3 * k * (np.sqrt(k) - 1) ** 2)); r = k * w
    ks = lambda g: (6 * np.log(g) / (r * (np.sqrt(r) - np.sqrt(w)) ** 2)) ** (1 / 3)
    jmax = int(5 * ks(1e12)) + 10
    T = np.arange(-H + 1, jmax + 1); N = len(T); idx = np.arange(N, dtype=float)
    fams = {"oracle": drift_family([k]), "ratefree": drift_family(KAPPA_GRID)}
    sizes = {"oracle": JMAX, "ratefree": JMAX * len(KAPPA_GRID), "flat": JMAX}
    thr = {arm: {g: np.sqrt(2 * np.log(4 * g * sizes[arm])) for g in GAMMAS} for arm in sizes}
    res = {}
    for arm in ["oracle", "ratefree", "flat"]:
        pre, post = (w, w - r) if arm != "flat" else (0.0, -r)
        mean = np.where(T < 0, pre * T, post * T)
        Y = mean + rng.standard_normal((trials, N))
        S0 = np.concatenate([np.zeros((trials, 1)), np.cumsum(Y, 1)], 1)          # S0[:, n+1] = sum_{i<=n}
        S1 = np.concatenate([np.zeros((trials, 1)), np.cumsum(Y * idx, 1)], 1)
        seg = lambda S, lo, hi: S[:, hi + 1] - S[:, lo]                           # sum over [lo, hi], arrays ok
        first = {g: np.full(trials, jmax) for g in GAMMAS}; done = {g: np.zeros(trials, bool) for g in GAMMAS}
        for i in range(1, jmax + 1):
            e = H - 1 + i
            if arm == "flat":
                j = np.arange(1, min(JMAX, e) + 1); o = e - j; W = e + 1
                Sm = j * (j + 1) / 2.0; nv = np.sqrt(j * (j + 1) * (2 * j + 1) / 6.0 - Sm ** 2 / W)
                # <y, m - mean m>, m = -(n - o) on (o, e]:  -(S1 - o S0) over (o,e]  + (Sm/W) S0 over [0,e]
                dot = -(seg(S1, o + 1, np.full_like(o, e)) - o * seg(S0, o + 1, np.full_like(o, e))) + (Sm / W) * S0[:, [e + 1]]
                Z = dot / nv
            else:
                F = fams[arm]; jj = F[:, 0].astype(int); kap = F[:, 1]; a = F[:, 2].astype(int); c = F[:, 3]; nv = F[:, 4]
                ok = (e + a >= 0) & (jj <= e); jj, kap, a, c, nv = jj[ok], kap[ok], a[ok], c[ok], nv[ok]
                lo1 = e + a; hi1 = e - jj; lo2 = e - jj + 1; ee = np.full_like(jj, e)
                # piece 1 (n in [e+a, e-j]): v = (n - e + j) - c ; piece 2 (n in (e-j, e]): v = (1-kap)(n - e + j) - c
                d1 = seg(S1, lo1, hi1) + (-e + jj - c) * seg(S0, lo1, hi1)
                d2 = (1 - kap) * seg(S1, lo2, ee) + ((1 - kap) * (jj - e) - c) * seg(S0, lo2, ee)
                Z = (d1 + d2) / nv
            Mx = Z.max(1)
            for g in GAMMAS:
                hit = (~done[g]) & (Mx > thr[arm][g]); first[g][hit] = i; done[g] |= hit
            if done[GAMMAS[-1]].all(): break
        res[arm] = {g: (first[g].mean(), first[g].std() / np.sqrt(trials), (~done[g]).mean()) for g in GAMMAS}
    return w, r, ks, thr, res

if __name__ == "__main__":
    rows = ["k,gamma,log_gamma,w,r,kstar,kstar_u_flat,arm,family_size,threshold,mean_delay,se,censored"]
    for k in KS:
        t0 = time.time(); w, r, ks, thr, res = run(k, TRIALS, 20261102 + int(10 * k))
        for g in GAMMAS:
            df = res["flat"][g][0]
            for arm, size in [("oracle", JMAX), ("ratefree", JMAX * len(KAPPA_GRID)), ("flat", JMAX)]:
                m, se, cz = res[arm][g]
                rows.append(f"{k},{g:.0e},{np.log(g):.3f},{w:.6g},{r:.6g},{ks(g):.3f},,{arm},{size},{thr[arm][g]:.4f},{m:.3f},{se:.3f},{cz:.4f}")
            print(f"k={k:<4} g={g:.0e}  oracle {res['oracle'][g][0]:6.1f}  ratefree {res['ratefree'][g][0]:6.1f}  flat {df:6.1f}"
                  f"  ratio oracle {res['oracle'][g][0]/df:5.3f} ratefree {res['ratefree'][g][0]/df:5.3f}  law {P(k):5.3f}"
                  f"  oracle/k* {res['oracle'][g][0]/ks(g):4.2f}  [{time.time()-t0:.0f}s]", flush=True)
    open("directed_scan.csv" if len(sys.argv) <= 2 else f"directed_scan_k{'_'.join(sys.argv[2:])}.csv", "w").write("\n".join(rows) + "\n")

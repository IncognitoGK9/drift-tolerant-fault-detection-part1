"""
V-Twin Geometry: verification code for matrix rows 1-3 (pure cone, Lambda = 0).
Row 1  independent QP (original x,b formulation) vs closed form
Row 2  finite lower bound (explicit inequality of Theorem 2) vs empirical mean delay of the scan
Row 3  oracle (true shape) vs rate-free (shape grid) scan: measured delay delta vs eta^(-2/3)
All streams are generated in 'transformed' coordinates z_t = y_t + rho_minus*t, in which normal paths are
non-decreasing (the cone K) and the least favourable drifting stream has slope w before the onset and
w(1-q) after it (q = r/w).  The scan never uses r or w: only the cone K, sigma, and its shape grid.
"""
import numpy as np, json, sys, time
from scipy.optimize import lsq_linear
from scipy.signal import fftconvolve
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------- closed forms
def reff(r, w): return r - np.sqrt(r * w)
def kstar(r, w, gamma, I=1.0):
    return (6 * np.log(gamma) / (I * reff(r, w) ** 2)) ** (1 / 3)

# ---------------------------------------------------------------- Row 1: independent QP
def qp_evidence(k, r, rho_m, rho_p, extra=60):
    """min sum_{t=-N..k} (x_t - f_t)^2 over x,b with increments in [-rho_m, rho_p], f = b - r (t)_+ (Lambda=0).
       Solved as a bounded least-squares problem in the increments (independent of the D-reduction)."""
    w = rho_m + rho_p
    N = int(np.ceil(k * (np.sqrt(r / w) - 1))) + extra
    ts = np.arange(-N, k + 1)
    n = len(ts)
    L = np.tril(np.ones((n, n)))
    A = np.hstack([L, -L])
    rhs = -r * np.clip(ts, 0, None)
    lb = np.full(2 * n, -rho_m); ub = np.full(2 * n, rho_p)
    res = lsq_linear(A, rhs, bounds=(lb, ub), method='bvls', tol=1e-12, max_iter=5000)
    d = A @ res.x - rhs
    return float((d ** 2).sum())

def sampled_v_cost(n, r, w, I=1.0):
    """cost sum delta_t^2 of the integer V twin of horizon n: uniform fall over L = ceil(c/w) steps to -c at the onset
       (rate c/L <= w, feasible), then rise at rate m = r - w.  Returns (cost, L)."""
    m = r - w
    c = n * w * (np.sqrt(r / w) - 1)
    L = int(np.ceil(c / w - 1e-12)) if c > 0 else 0
    pre = -c * np.arange(1, L + 1) / max(L, 1) if L > 0 else np.array([])        # t = -L+1 .. 0 (value at t=0 is -c)
    post = -c + m * np.arange(1, n + 1)                                           # t = 1 .. n
    return float((pre ** 2).sum() + (post ** 2).sum()), L

# ---------------------------------------------------------------- Row 2: finite lower bound (Theorem 2)
def finite_lower_bound(gamma, r, w, hist=0, I=1.0, nmax=None):
    """max_{n,l}  n * (1 - e^l * delta - Phibar((l-G_n)/sqrt(2 G_n)))   with
       G_n = (I/2) * cost(V_n),  K = n + ceil(c/w),  delta = ln2 / floor((2 gamma - c/w)/K)."""
    ks = kstar(r, w, gamma, I)
    if nmax is None: nmax = int(3 * ks) + 5
    best = (0.0, None, None)
    lg = np.log(gamma)
    for n in range(max(2, int(0.2 * ks)), nmax):
        cost, cw = sampled_v_cost(n, r, w, I)
        G = 0.5 * I * cost
        K = n + cw
        J = int((2 * gamma - hist - cw) // K)
        if J < 1: continue
        delta = np.log(2) / J * 1.0000001
        # optimise l on a grid (log(1/delta) caps e^l delta < 1)
        ls = np.linspace(0.05, lg + 5, 400)
        val = 1 - np.exp(ls) * delta - norm.sf((ls - G) / np.sqrt(2 * G))
        i = int(np.argmax(val))
        lb = n * val[i]
        if lb > best[0]: best = (lb, n, float(ls[i]))
    return best

# ---------------------------------------------------------------- templates (isotonic cone residuals)
_iso = IsotonicRegression(increasing=True)
def make_template(j, beta, P):
    s = np.ones(P); s[P - j:] = 1 - beta
    mu = np.cumsum(s)
    v = mu - _iso.fit_transform(np.arange(P), mu)
    return v / np.linalg.norm(v), mu

def window_len(j, beta): return int(np.ceil(4 * j * np.sqrt(max(beta, 1.0)))) + 20

def geometric_lags(jmax, ratio=1.03, jmin=4):
    lags, j = [], float(jmin)
    while j <= jmax:
        lags.append(int(round(j))); j *= ratio
    return sorted(set(lags))

def build_templates(lags, betas):
    T = []
    for b in betas:
        for j in lags:
            P = window_len(j, b)
            v, _ = make_template(j, b, P)
            T.append((j, b, P, v[::-1].copy()))
    return T

# ---------------------------------------------------------------- scan on simulated least favourable stream
def run_scan(templates, q, w, H, Tmax, nruns, seed, thresholds):
    """returns dict gamma -> list of delays (T - nu + 1), plus pre-onset alarm counts.
       onset nu = H.  S_t = max over templates of the normalised statistic."""
    rng = np.random.default_rng(seed)
    Pmax = max(t[2] for t in templates)
    t0 = H - 200
    Ttot = H + Tmax
    out = {g: [] for g in thresholds}; pre = {g: 0 for g in thresholds}; cens = {g: 0 for g in thresholds}
    tt = np.arange(Ttot)
    mean = np.where(tt <= H, w * tt, w * H + w * (1 - q) * (tt - H))
    for _ in range(nruns):
        z = mean + rng.standard_normal(Ttot)
        S = np.full(Ttot - t0, -np.inf)
        for (j, b, P, vr) in templates:
            seg = z[t0 - P + 1:]
            o = fftconvolve(seg, vr, mode='valid')          # length Ttot - t0
            np.maximum(S, o, out=S)
        for g, x in thresholds.items():
            hit = np.nonzero(S > x)[0]
            if len(hit) == 0: cens[g] += 1; continue
            t_alarm = hit[0] + t0
            if t_alarm < H: pre[g] += 1; continue           # alarm before onset: excluded (conditioning on T >= nu)
            out[g].append(t_alarm - H + 1)
    return out, pre, cens

def gauss_threshold(gamma, ntempl): return float(np.sqrt(2 * np.log(4 * gamma * ntempl)))

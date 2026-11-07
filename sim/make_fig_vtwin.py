"""Figure 1 of Part I: the least favourable normal path (the V twin) for r = 4w, from the exact quadratic program.

Solves  min sum_{t=-N..k} (x_t - f_t)^2  over normal paths x, b (increments in [-rho_-, rho_+]) with
f = b - r (t)_+  (Lambda = 0), as a bounded least-squares problem in the increments (the same program as
qp_evidence in vtwin_sim.py), and plots (a) the base, the faulty path and the twin, (b) the gap delta = x - f.
Deterministic (no random numbers).  Usage:  python3 make_fig_vtwin.py [out.png]
"""
import sys
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_M, RHO_P, Q, K = 0.5, 0.5, 4.0, 30          # rho_-, rho_+, r/w, horizon k
W = RHO_M + RHO_P; R = Q * W
M = R - W; REFF = R - np.sqrt(R * W)
C = K * W * (np.sqrt(Q) - 1.0)                   # lead: the twin leaves the base C/W steps before the onset
EXTRA = 8

def qp_paths(k, r, rho_m, rho_p, extra):
    w = rho_m + rho_p
    N = int(np.ceil(k * (np.sqrt(r / w) - 1))) + extra
    ts = np.arange(-N, k + 1); n = len(ts)
    L = np.tril(np.ones((n, n)))
    A = np.hstack([L, -L])                       # x = L dx, b = L db  (both start at 0 at t = -N)
    rhs = -r * np.clip(ts, 0, None)
    res = lsq_linear(A, rhs, bounds=(np.full(2 * n, -rho_m), np.full(2 * n, rho_p)),
                     method="bvls", tol=1e-12, max_iter=5000)
    x = L @ res.x[:n]; b = L @ res.x[n:]
    f = b - r * np.clip(ts, 0, None)
    return ts, x, b, f

ts, x, b, f = qp_paths(K, R, RHO_M, RHO_P, EXTRA)
d = x - f
E = float((d ** 2).sum())
print(f"r={R:g} w={W:g} k={K}: E(k) [QP] = {E:.3f}, (1/3) r_eff^2 k^3 = {REFF**2 * K**3 / 3:.3f}, "
      f"lead c/w = {C / W:.1f}, delta(k) = {d[-1]:.3f} (k r_eff = {K * REFF:.3f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), dpi=180)
a = ax[0]
a.plot(ts, b, "--", color="0.55", lw=2, label=r"base $b$ (rises at $\rho_+$)")
a.plot(ts, f, color="#d62728", lw=2.6, label=r"faulty path $f=b-r(t)_+$")
a.plot(ts, x, color="#1f77b4", lw=2.6, label=r"normal twin $x$ (falls at $\rho_-$)")
for t0 in (-C / W, K): a.axvline(t0, color="k", ls=":", lw=0.9)
a.axvline(0, color="k", lw=0.9)
a.set_xlabel(r"time $t$ (onset at 0)"); a.set_ylabel("mean level")
a.set_title("(a) the twin starts falling before the onset"); a.legend(loc="lower left", frameon=False)
bx = ax[1]
bx.plot(ts, d, color="k", lw=2.6)
bx.axhline(0, color="0.6", lw=0.8); bx.axvline(0, color="k", lw=0.9)
for t0 in (-C / W, K): bx.axvline(t0, color="k", ls=":", lw=0.9)
bx.annotate(r"$\delta(k)=k\,r_{\mathrm{eff}}$", xy=(K, d[-1]), xytext=(K - 16, d[-1]),
            arrowprops=dict(arrowstyle="->", lw=0.9), va="center", ha="right")
bx.text(-3, 0.6 * d[-1], r"slope $m=r-w$", ha="right"); bx.text(-28, 0.7 * d[np.searchsorted(ts, 0)], r"slope $-w$")
bx.set_xlabel(r"time $t$ (onset at 0)"); bx.set_ylabel(r"separation $\delta=x-f$")
bx.set_title(r"(b) $E_0(k)=\sum\delta^2\approx\frac{1}{3}r_{\mathrm{eff}}^2k^3$")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "fig_vtwin.png"
fig.savefig(out); print("wrote", out)

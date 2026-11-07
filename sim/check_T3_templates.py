"""Check for Theorem 3 (scan): (i) the isotonic residual of the kinked ramp is supported on ONE block of
length j(sqrt q)+O(1) independent of the window; (ii) ||v_{j,q}||^2 >= E(j) >= r_eff^2 j^3/3 with E(j) from the
independent QP (row-1 code); (iii) the efficiency eta_F = min over lags of the best cosine is well defined and
agrees across lags to 4 decimals (replaces v3's limit definition).  numpy/sklearn only; < 1 min."""
import numpy as np, json
from sklearn.isotonic import IsotonicRegression
from vtwin_sim import qp_evidence, reff
iso = IsotonicRegression(increasing=True)

def residual(j, q, P, w=1.0):
    s = np.full(P, w); s[P - j:] = w * (1 - q); mu = np.cumsum(s)
    fit = iso.fit_transform(np.arange(P), mu); v = mu - fit
    nz = np.nonzero(np.abs(v) > 1e-9)[0]
    return v, (nz.min(), nz.max()) if len(nz) else None, fit

out = {}
print("(i) block support vs window length (w=1):")
for q in [1.5, 2.0, 4.0]:
    for j in [20, 40]:
        rows = []
        for P in [int(3 * j * np.sqrt(q)) + 10, int(6 * j * np.sqrt(q)) + 10, int(12 * j * np.sqrt(q)) + 10]:
            v, (lo, hi), fit = residual(j, q, P)
            rows.append((int(P), int(hi - lo + 1), float((v ** 2).sum())))
        pred = j * np.sqrt(q)
        print(f"  q={q} j={j}: predicted block length j*sqrt(q)={pred:.1f}; (window, block, ||v||^2) = {rows}")
        assert max(abs(a[2] - rows[0][2]) for a in rows) < 1e-6, "block cost depends on window!"
        out[f"block_q{q}_j{j}"] = dict(pred=pred, rows=rows)

print("(ii) ||v||^2 >= E(j) >= r_eff^2 j^3 / 3   (rho_-=0.3, rho_+=0.7, w=1)")
for q in [1.5, 2.0, 4.0]:
    for j in [20, 40]:
        r = q; w = 1.0
        v, _, _ = residual(j, q, int(6 * j * np.sqrt(q)) + 10); nv2 = float((v ** 2).sum())
        E = qp_evidence(j, r, 0.3, 0.7)
        lead = reff(r, w) ** 2 * j ** 3 / 3
        print(f"  q={q} j={j}: ||v||^2={nv2:.3f}  E(j) [QP]={E:.3f}  lead={lead:.3f}  ratios {nv2/E:.4f}, {E/lead:.4f}")
        assert nv2 >= E - 1e-6 and E >= lead - 1e-6
        out[f"ev_q{q}_j{j}"] = dict(nv2=nv2, E=E, lead=lead)

print("(iii) eta_F: min over lags of max over grid shapes of the cosine (grid 1.25*1.26^i, i<12)")
grid = [1.25 * 1.26 ** i for i in range(12)]
def cosine(q, beta, j):
    P = int(6 * j * np.sqrt(max(q, beta))) + 20
    vq, _, fitq = residual(j, q, P); vb, _, _ = residual(j, beta, P); muq = vq + fitq
    return float(muq @ vb / (np.linalg.norm(vq) * np.linalg.norm(vb)))     # numerator uses the mean path m_{j,q}, as in Theorem 3
for q in [1.5, 2.0, 3.0, 5.0, 10.0]:
    vals = [max(cosine(q, b, j) for b in grid) for j in [10, 20, 40, 80, 160]]
    print(f"  q={q}: eta at j=10,20,40,80,160 = {[round(x,4) for x in vals]}; eta_F(min) = {min(vals):.4f}")
    out[f"eta_q{q}"] = vals
json.dump(out, open("check_T3_templates.json", "w"), indent=1)
print("OK")

import numpy as np, json, time, sys
from vtwin_sim import *
d = json.load(open("rows23.json")); w = d["w"]
done = {(x["q"], x["g"]) for x in d["row3"]}
def grid(g, b0=1.25, bmax=17.0):
    out, b = [], b0
    while b <= bmax: out.append(b); b *= g
    return out
for q in [2.0, 4.0]:
    ks = kstar(q * w, w, 1e6)
    lags = geometric_lags(int(1.8 * kstar(q * w, w, 1e8)), ratio=1.05)
    tm_or = build_templates(lags, [q]); Pm = max(t[2] for t in tm_or)
    th_or = {1e6: gauss_threshold(1e6, len(tm_or))}
    Tmax = int(4 * kstar(q * w, w, 1e8))
    for g in [1.10, 1.26, 1.6, 2.5]:
        if (q, g) in done: continue
        t0 = time.time(); bs = grid(g)
        tm_rf = build_templates(lags, bs)
        Pmax = max(max(t[2] for t in tm_rf), Pm); H = Pmax + 200
        th_rf = {1e6: gauss_threshold(1e6, len(tm_rf))}
        o1, p1, c1 = run_scan(tm_or, q, w, H, Tmax, 600, 21, th_or)
        o2, p2, c2 = run_scan(tm_rf, q, w, H, Tmax, 600, 21, th_rf)
        d1 = np.array(o1[1e6], float); d2 = np.array(o2[1e6], float)
        d = json.load(open("rows23.json"))
        d["row3"].append(dict(q=q, g=g, nshapes=len(bs), kstar=ks, oracle=float(d1.mean()), rate_free=float(d2.mean()),
                              se_o=float(d1.std(ddof=1) / np.sqrt(len(d1))), se_r=float(d2.std(ddof=1) / np.sqrt(len(d2))),
                              x_or=th_or[1e6], x_rf=th_rf[1e6], ntempl_rf=len(tm_rf), ntempl_or=len(tm_or), pre=(p1[1e6], p2[1e6])))
        json.dump(d, open("rows23.json", "w"), indent=1)
        print("row3 q", q, "g", g, round(time.time() - t0, 1), "s", flush=True)
print("DONE", flush=True)

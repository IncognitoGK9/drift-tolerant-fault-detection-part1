import json, numpy as np
from vtwin_sim import *
d=json.load(open('rows23.json')); w=d['w']
out=[]
for r in d['row2']:
    q,g=r['q'],r['gamma']
    ks8=kstar(q*w,w,1e8)
    lags=geometric_lags(int(1.8*ks8),ratio=1.03)
    Pmax=window_len(max(lags),q); H=Pmax+200
    lb,n,l=finite_lower_bound(g,q*w,w,hist=H)
    out.append(dict(q=q,gamma=g,kstar=r['kstar'],lb=lb,n_lb=n,mean=r['mean'],se=r['se'],H=H,nruns=r['nruns'],pre=r['pre'],cens=r['cens']))
json.dump(out,open('row2_final.json','w'),indent=1)
for o in out:
    print(f"q={o['q']:<5} g={o['gamma']:.0e} k*={o['kstar']:.1f} LB={o['lb']:.1f} (n={o['n_lb']}) mean={o['mean']:.1f}+-{o['se']:.1f} mean/LB={o['mean']/o['lb']:.2f} mean/k*={o['mean']/o['kstar']:.3f} LB/k*={o['lb']/o['kstar']:.2f} H={o['H']}")

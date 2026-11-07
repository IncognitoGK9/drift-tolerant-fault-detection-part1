import numpy as np, sys
sys.argv=["x"]; from sim_directed_scan import template, KAPPA_GRID
def vec(j,kap,n):
    a,c,nv=template(j,kap); p=np.arange(-n+1,1); q=p+j; m=np.where(q<=0,q,(1-kap)*q).astype(float)
    v=np.where(p>=a,m-c,0.0); return m,v
for k in [1.5,2,3,5,10]:
    effs=[]
    for j in [20,40,80]:
        n=int(j*(np.sqrt(16)+1)*2)+40
        m,vstar=vec(j,k,n); d=np.linalg.norm(vstar)
        best=max((m@vec(j,kap,n)[1])/(np.linalg.norm(vec(j,kap,n)[1])*d) for kap in KAPPA_GRID)
        effs.append(best)
    print(f"k={k}: efficiency of rate-free grid at j=20,40,80: "+", ".join(f"{e:.4f}" for e in effs))

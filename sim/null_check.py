"""False-alarm check of the rate-free LFP scan under normal paths (no fault), w as for k=3.
Paths (transformed coordinates): apex (flat), least favourable base (rising at w), switching slopes in [0,w].
Reports the fraction of runs alarming within 2*gamma steps at gamma = 1e3 (Theorem bound: <= 1/4)."""
import numpy as np, sys
sys.argv=["x"]; from sim_directed_scan import drift_family, KAPPA_GRID, JMAX
rng=np.random.default_rng(7); TR=300; G=1e3; HOR=int(2*G); B=600
k=3.0; w=np.sqrt(6*15/(40**3*k*(np.sqrt(k)-1)**2))
F=drift_family(KAPPA_GRID); x=np.sqrt(2*np.log(4*G*JMAX*len(KAPPA_GRID)))
N=B+HOR; idx=np.arange(N,dtype=float)
def slopes(kind):
    if kind=="apex": return np.zeros((TR,N))
    if kind=="lf": return np.full((TR,N),w)
    s=np.zeros((TR,N)); 
    for i in range(TR):
        t=0
        while t<N:
            L=rng.geometric(1/150); s[i,t:t+L]=rng.choice([0,w,rng.uniform(0,w)]); t+=L
    return s
for kind in ["apex","lf","switch"]:
    Y=np.cumsum(slopes(kind),1)+rng.standard_normal((TR,N))
    S0=np.concatenate([np.zeros((TR,1)),np.cumsum(Y,1)],1); S1=np.concatenate([np.zeros((TR,1)),np.cumsum(Y*idx,1)],1)
    seg=lambda S,lo,hi:S[:,hi+1]-S[:,lo]; alarm=np.zeros(TR,bool); mx=np.full(TR,-np.inf)
    jj=F[:,0].astype(int); kap=F[:,1]; a=F[:,2].astype(int); c=F[:,3]; nv=F[:,4]
    for e in range(B,N):
        ok=(e+a>=0)&(jj<=e); j_,k_,a_,c_,n_=jj[ok],kap[ok],a[ok],c[ok],nv[ok]; ee=np.full_like(j_,e)
        Z=(seg(S1,e+a_,e-j_)+(-e+j_-c_)*seg(S0,e+a_,e-j_)+(1-k_)*seg(S1,e-j_+1,ee)+((1-k_)*(j_-e)-c_)*seg(S0,e-j_+1,ee))/n_
        m=Z.max(1); mx=np.maximum(mx,m); alarm|=m>x
    print(f"{kind:7s}: P(alarm within 2*gamma={HOR}) = {alarm.mean():.3f}   (bound 0.25); median max Z = {np.median(mx):.2f} vs threshold {x:.2f}",flush=True)

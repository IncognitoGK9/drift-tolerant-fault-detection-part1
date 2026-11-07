import numpy as np, json
from vtwin_sim import *
w=1.0; rows=[]
for (fm,fp) in [(0.2,0.8),(0.5,0.5),(0.9,0.1)]:
    for q in [1.3,2.0,4.0]:
        for k in [20,33,47]:
            r=q*w; qp=qp_evidence(k,r,fm*w,fp*w); re=reff(r,w)
            lead=re**2*k**3/3; corr=lead+0.5*re**2*k**2; vt,L=sampled_v_cost(k,r,w)
            rows.append(dict(fm=fm,q=q,k=k,cw=k*(np.sqrt(q)-1),frac=(k*(np.sqrt(q)-1))%1,qp=qp,lead=lead,corr=corr,vt=vt))
import pandas as pd
df=pd.DataFrame(rows)
df['e_lead']=df.qp/df.lead-1; df['e_corr']=df.qp/df['corr']-1; df['e_vt']=df.vt/df.qp-1
out=dict(max_lead=float(df.e_lead.abs().max()), max_corr=float(df.e_corr.abs().max()), max_vt=float(df.e_vt.abs().max()), min_vt=float(df.e_vt.min()),
         spread_asym=float((df.groupby(['q','k']).qp.max()/df.groupby(['q','k']).qp.min()-1).max()), n=len(df),
         n_noninteger=int((df.frac.between(0.01,0.99)).sum()), lead_at_k47=float(df[df.k==47].e_lead.abs().max()))
print(out); json.dump(out,open('row1.json','w'),indent=1)

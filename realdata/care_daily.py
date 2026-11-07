import numpy as np, glob, json, os, pandas as pd
OUT=os.path.expanduser("~/mnt/TWA_data/Claude outputs/p3a_care_daily"); rows=[]
for farm in ["A","B","C"]:
    for f in sorted(glob.glob(f"{farm}/*.npz")):
        z=np.load(f,allow_pickle=True); m=json.loads(str(z["meta"]))
        if m.get("event",{}).get("event_label")!="normal": continue
        P=[k for k in z.files if k.startswith("power_") and k.endswith("_avg")]; W=[k for k in z.files if k.startswith("wind_speed_") and k.endswith("_avg")]
        st=z["status_type_id"]; ok0=(st==0)
        best=None
        for p in P:
            for w in W:
                a=z[p][ok0]; b=z[w][ok0]; g=np.isfinite(a)&np.isfinite(b)
                if g.sum()<1000: continue
                c=pd.Series(a[g]).rank().corr(pd.Series(b[g]).rank())
                if best is None or c>best[0]: best=(c,p,w)
        if best is None: continue
        c,p,w=best; t=z["time_min"]; day=(t-t.min())//1440; pw=z[p].astype(float); ws=z[w].astype(float)
        ok=ok0&np.isfinite(pw)&np.isfinite(ws)&(ws>=3.5)&(ws<=25)
        nd=int(day.max())+1; trd=int(0.6*nd); tr=ok&(day<trd)
        bins=np.floor(ws/0.5).astype(int); med=pd.Series(pw[tr]).groupby(bins[tr]).median()
        pexp=pd.Series(bins).map(med).values.astype(float); ok&=np.isfinite(pexp)&(pexp>0)
        d=pd.DataFrame({"day":day[ok],"p":pw[ok],"e":pexp[ok]}).groupby("day").agg(p=("p","sum"),e=("e","sum"),n=("p","size"))
        d=d[d.n>=36]; d["perf_index"]=d.p/d.e
        name=f"care_{farm}_{m['id']}_asset{m['asset_id']}"; d[["perf_index","n"]].to_csv(f"{OUT}/{name}.csv")
        rows.append((name,len(d),round(float(c),3),p,w))
for r in rows: print(r)
print(len(rows),"normal datasets written")

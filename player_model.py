import numpy as np
import pandas as pd

def features(history, p):
    d=history.sort_values('GAME_DATE').copy()
    # Aggregate by date before lagging: no same-day information.
    t=d.groupby(['TEAM_ID','GAME_DATE'],as_index=False)[['PTS_opp','poss']].sum()
    g=t.groupby('TEAM_ID')
    t['dr']=100*(g.PTS_opp.cumsum()-t.PTS_opp)/(g.poss.cumsum()-t.poss)
    t['gp']=g.cumcount()
    d=d.merge(t[['TEAM_ID','GAME_DATE','dr','gp']],on=['TEAM_ID','GAME_DATE'])
    opp=d[['GAME_ID','TEAM_ID_opp','dr','gp','poss']].rename(columns={'TEAM_ID_opp':'TEAM_ID'})
    p=p.copy(); p['GAME_ID']=p.GAME_ID.astype(str); opp['GAME_ID']=opp.GAME_ID.astype(str)
    p['MIN']=pd.to_numeric(p.MIN,errors='coerce'); p['PTS']=pd.to_numeric(p.PTS,errors='coerce')
    p=p.merge(opp,on=['GAME_ID','TEAM_ID'],validate='many_to_one')
    return p[(p.MIN>0)&(p.gp>=3)&(p.poss>0)].dropna(subset=['dr','PTS'])

def project(p, roster, target_dr, pace, excluded=()):
    rows=[]
    for player in roster.itertuples():
        if player.PLAYER_NAME in excluded: continue
        q=p[p.PLAYER_ID==player.PLAYER_ID].sort_values('GAME_DATE')
        minutes=float(player.MIN)
        if not np.isfinite(minutes) or minutes<=0: continue
        baseline=float(player.PTS)/minutes
        slope=0.; adjusted=baseline; similar=0
        if len(q):
            x=q.dr.to_numpy()-target_dr; y=(q.PTS/q.MIN*100/q.poss).to_numpy()
            weights=np.exp(-np.abs(x)/6)*np.minimum(q.MIN.to_numpy()/20,1)
            effective=weights.sum(); similar=int((abs(x)<=5).sum())
            # Regularized local linear fit of scoring per minute per 100 possessions.
            X=np.column_stack([np.ones(len(x)),x]); prior=baseline*100/pace
            beta=np.linalg.solve(X.T@(weights[:,None]*X)+np.diag([12.,1500.]),X.T@(weights*y)+[12*prior,0])
            adjusted=max(0,float(beta[0])*pace/100); slope=float(beta[1])*pace/100
        rows.append(dict(player=player.PLAYER_NAME,minutes=minutes,baseline=baseline,rate=adjusted,games=len(q),similar=similar,slope=slope))
    r=pd.DataFrame(rows)
    if len(r)<5: raise ValueError('En az 5 uygun oyuncu gerekli')
    # Cap individual playing time and distribute regulation minutes.
    m=r.minutes.to_numpy(); lo,hi=0.,1000.
    for _ in range(60):
        mid=(lo+hi)/2
        if np.minimum(48,m*mid).sum()<240: lo=mid
        else: hi=mid
    r['minutes']=np.minimum(48,m*hi); r['expected']=r.minutes*r.rate; r['normal']=r.minutes*r.baseline
    return r

def simulate_players(r,n,rng,shared):
    means=r.expected.to_numpy(); noise=rng.gamma(12,1/12,size=(n,len(means)))
    return rng.poisson(np.maximum(0,means[None,:]*shared[:,None]*noise))

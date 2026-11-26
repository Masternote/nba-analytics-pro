import numpy as np
import pandas as pd
import requests
import streamlit as st
from nba_api.stats.endpoints import leaguegamefinder, leaguedashplayerstats
@st.cache_data(ttl=1800)
def logs(season):
    d=leaguegamefinder.LeagueGameFinder(season_nullable=season,season_type_nullable='Regular Season',player_or_team_abbreviation='T',league_id_nullable='00',timeout=35).get_data_frames()[0]
    d['GAME_DATE']=pd.to_datetime(d.GAME_DATE)
    d=d.drop_duplicates(['GAME_ID','TEAM_ID'])
    d=d.merge(d[['GAME_ID','TEAM_ID','PTS','FGA','FTA','OREB','TOV']],on='GAME_ID',suffixes=('','_opp'))
    d=d[d.TEAM_ID!=d.TEAM_ID_opp].copy()
    d['poss']=(d.FGA+.44*d.FTA-d.OREB+d.TOV+d.FGA_opp+.44*d.FTA_opp-d.OREB_opp+d.TOV_opp)/2
    return d[d.poss>0].sort_values('GAME_DATE')
def form(d,team_id,cutoff):
    s=d[(d.TEAM_ID==team_id)&(d.GAME_DATE<pd.Timestamp(cutoff))]
    rows=[]
    for n in [0,5,10,20]:
        q=s.tail(n) if n else s
        if len(q): rows.append(dict(window=n,GP=len(q),off=100*q.PTS.sum()/q.poss.sum(),defense=100*q.PTS_opp.sum()/q.poss.sum(),pace=q.poss.mean()))
    return pd.DataFrame(rows)
def blend(f):
    f=f.copy(); w=f.window.map({0:.5,5:.15,10:.2,20:.15})*np.where(f.window==0,1,np.minimum(1,f.GP/f.window.replace(0,1)))
    return np.average(f[['off','defense','pace']],axis=0,weights=w)
@st.cache_data(ttl=1800)
def players(season,cutoff):
    return leaguedashplayerstats.LeagueDashPlayerStats(season=season,season_type_all_star='Regular Season',date_to_nullable=cutoff.strftime('%m/%d/%Y'),per_mode_detailed='PerGame',timeout=30).get_data_frames()[0]
@st.cache_data(ttl=300)
def injuries():
    r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries',timeout=15); r.raise_for_status(); return r.json()['injuries']
def backtest(d,limit=300):
    out=[]
    for gid,g in d.groupby('GAME_ID',sort=False):
        if len(g)!=2: continue
        h=g[~g.MATCHUP.str.contains('@',regex=False)]; a=g[g.MATCHUP.str.contains('@',regex=False)]
        if len(h)!=1 or len(a)!=1: continue
        h=h.iloc[0]; a=a.iloc[0]; date=h.GAME_DATE
        prior=d[d.GAME_DATE<date]; fh=form(prior,h.TEAM_ID,date); fa=form(prior,a.TEAM_ID,date)
        if fh.empty or fa.empty or min(fh.GP.iloc[0],fa.GP.iloc[0])<5: continue
        base=100*prior.PTS.sum()/prior.poss.sum(); x,y=blend(fh),blend(fa); pace=(x[2]+y[2])/2
        ph=pace*x[0]*y[1]/base/100+1; pa=pace*y[0]*x[1]/base/100-1
        out.append(dict(date=date,home=h.TEAM_NAME,away=a.TEAM_NAME,pred_home=ph,pred_away=pa,actual_home=h.PTS,actual_away=a.PTS))
    return pd.DataFrame(out).tail(limit)

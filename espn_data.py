import requests, pandas as pd, streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import date,timedelta
from nba_api.stats.static import teams
ROOT='https://site.api.espn.com/apis/site/v2/sports/basketball/nba/'

def get(path,**params):
    r=requests.get(ROOT+path,params=params,timeout=15); r.raise_for_status(); return r.json()

def norm(s): return s.lower().replace('los angeles clippers','la clippers').replace('la lakers','los angeles lakers')
IDS={norm(t['full_name']):t['id'] for t in teams.get_teams()}
def tid(t): return IDS[norm(t['displayName'])]
@st.cache_data(ttl=3600)
def board(start,end):
    if start == end:
        return get('scoreboard', dates=f'{start:%Y%m%d}', limit=100)['events']
    try:
        return get('scoreboard', dates=f'{start:%Y%m%d}-{end:%Y%m%d}', limit=100)['events']
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 400:
            raise
        events = {}
        day = start
        while day <= end:
            for event in board(day, day):
                events[event['id']] = event
            day += timedelta(days=1)
        return list(events.values())

def number(x): return float(x)
def made_attempt(x): return float(x.split('-')[-1])
def minutes(x):
    if ':' in str(x):
        a,b=str(x).split(':'); return float(a)+float(b)/60
    return float(x)
@st.cache_data(ttl=86400)
def box(gid):
    j=get('summary',event=gid); b=j['boxscore']; rows=[]; ps=[]
    c=j['header']['competitions'][0]; day=pd.Timestamp(c['date']).tz_convert('America/New_York').tz_localize(None).normalize()
    for t in b['teams']:
        team=t['team']; i=tid(team); s={x['name']:x['displayValue'] for x in t['statistics']}; comp=next(x for x in c['competitors'] if str(x['id'])==str(team['id']))
        rows.append(dict(GAME_ID=str(gid),GAME_DATE=day,TEAM_ID=i,TEAM_NAME=team['displayName'],MATCHUP='vs.' if comp['homeAway']=='home' else '@',PTS=float(comp['score']),FGA=made_attempt(s['fieldGoalsMade-fieldGoalsAttempted']),FTA=made_attempt(s['freeThrowsMade-freeThrowsAttempted']),OREB=number(s['offensiveRebounds']),TOV=number(s['totalTurnovers'] if 'totalTurnovers' in s else s['turnovers'])))
    for t in b.get('players',[]):
        i=tid(t['team']);
        for group in t['statistics']:
            labels=group['labels']
            if not {'MIN','PTS'}.issubset(labels): continue
            for a in group['athletes']:
                if a.get('didNotPlay') or not a.get('stats'): continue
                s=dict(zip(labels,a['stats'])); m=minutes(s['MIN'])
                if m<=0: continue
                ps.append(dict(GAME_ID=str(gid),GAME_DATE=day,TEAM_ID=i,PLAYER_ID=int(a['athlete']['id']),PLAYER_NAME=a['athlete']['displayName'],MIN=m,PTS=float(s['PTS']),AST=float(s.get('AST',0)),REB=float(s.get('REB',0))))
    if len(rows)!=2 or not ps: raise ValueError('Eksik kutu skoru')
    return rows,ps
@st.cache_data(ttl=1800,show_spinner=False)
def dataset(cutoff):
    start=date(cutoff.year if cutoff.month>=10 else cutoff.year-1,10,1); events={}
    while start<=cutoff:
        end=min(start+timedelta(days=6),cutoff)
        for e in board(start,end):
            if e['status']['type'].get('completed') and e.get('season',{}).get('type')==2: events[e['id']]=e
        start=end+timedelta(days=1)
    if not events: raise ValueError('Tamamlanmış normal sezon maçı yok')
    rows=[]; ps=[]; errors=[]; progress=st.progress(0,text='ESPN geçmiş maçları alınıyor')
    def safe(g):
        try: return box(g)
        except Exception as e: return str(e)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for k,result in enumerate(pool.map(safe,events)):
            if isinstance(result,str): errors.append(result)
            else: rows.extend(result[0]); ps.extend(result[1])
            if k%10==0: progress.progress((k+1)/len(events),text=f'{k+1}/{len(events)} maç')
    progress.empty()
    if errors: raise ValueError(f'{len(errors)}/{len(events)} maç verisi eksik; ilk hata: {errors[0]}')
    d=pd.DataFrame(rows); p=pd.DataFrame(ps); d=d.merge(d[['GAME_ID','TEAM_ID','PTS','FGA','FTA','OREB','TOV']],on='GAME_ID',suffixes=('','_opp')); d=d[d.TEAM_ID!=d.TEAM_ID_opp].copy(); d['poss']=(d.FGA+.44*d.FTA-d.OREB+d.TOV+d.FGA_opp+.44*d.FTA_opp-d.OREB_opp+d.TOV_opp)/2
    d=d[(d.poss>0)&(d.GAME_DATE<=pd.Timestamp(cutoff))].sort_values('GAME_DATE'); p=p[p.GAME_DATE<=pd.Timestamp(cutoff)]
    return d,p

def roster(p):
    p=p.sort_values('GAME_DATE'); latest=p.drop_duplicates('PLAYER_ID',keep='last')[['PLAYER_ID','TEAM_ID']]; p=p.merge(latest,on=['PLAYER_ID','TEAM_ID']); r=p.groupby(['TEAM_ID','PLAYER_ID','PLAYER_NAME'],as_index=False).agg(GP=('PTS','size'),MIN=('MIN','mean'),PTS=('PTS','mean'),AST=('AST','mean'),REB=('REB','mean')); r['TEAM_ABBREVIATION']=r.TEAM_ID.map({t['id']:t['abbreviation'] for t in teams.get_teams()}); return r

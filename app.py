from espn_data import dataset, roster as espn_roster
import streamlit as st
from player_model import features, project, simulate_players
from nba_api.stats.endpoints import playergamelogs
from extras import logs,form,blend,players,injuries,backtest
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams

@st.cache_data(ttl=300)
def schedule(day):
    # ESPN scoreboard dates use US Eastern time. Fetch adjacent dates for Turkey.
    tz=ZoneInfo('Europe/Istanbul')
    games={}
    for offset in (-1,0,1):
        d=day+timedelta(days=offset)
        r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',params={'dates':d.strftime('%Y%m%d'),'limit':100},timeout=12)
        r.raise_for_status()
        for e in r.json()['events']:
            start=datetime.fromisoformat(e['date'].replace('Z','+00:00')).astimezone(tz)
            if start.date()!=day: continue
            c=e['competitions'][0]
            teams={t['homeAway']:t['team']['displayName'] for t in c['competitors']}
            games[e['id']]=(start,teams['home'],teams['away'],e['status']['type']['description'],e['id'])
    return sorted(games.values())
@st.cache_data(ttl=1800)
def ratings(cutoff):
    year=cutoff.year if cutoff.month>=10 else cutoff.year-1
    season=f'{year}-{(year+1)%100:02d}'
    df=leaguedashteamstats.LeagueDashTeamStats(season=season,season_type_all_star='Regular Season',measure_type_detailed_defense='Advanced',date_to_nullable=cutoff.strftime('%m/%d/%Y'),timeout=25).get_data_frames()[0]
    df=df[['TEAM_ID','TEAM_NAME','GP','OFF_RATING','DEF_RATING','PACE']].copy()
    for c in ['GP','OFF_RATING','DEF_RATING','PACE']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(); df=df[(df.GP>0)&(df.OFF_RATING>0)&(df.DEF_RATING>0)&(df.PACE>0)]
    if df.empty: raise ValueError('Geçerli takım verisi yok')
    return df,season

def normalize(name):
    return name.lower().replace('los angeles clippers','la clippers').replace('la lakers','los angeles lakers').strip()

st.set_page_config(page_title='NBA Analytics Pro',layout='wide')
st.title('NBA Analytics Pro v4.2')
st.warning('Deneysel model; gerçek takım istatistikleri kullanılır ancak model henüz geçmiş maçlarla kalibre edilmemiştir. Sakatlık raporu ve oyuncu verileri ayrı gösterilir; oyuncu modeli henüz kalibre edilmemiştir.')
mode=st.radio('Maç seçimi',['Maç takvimi','Takım seçimi'],horizontal=True)
now=datetime.now(ZoneInfo('America/New_York'))
day=st.date_input('Maç tarihi (Türkiye saati)',datetime.now(ZoneInfo('Europe/Istanbul')).date())
if st.button('Verileri yenile'):
    schedule.clear(); ratings.clear(); dataset.clear(); st.session_state.pop('result',None)
if mode=='Maç takvimi':
    try: games=schedule(day)
    except Exception as exc:
        st.error(f'Maç takvimi alınamadı: {exc}'); st.stop()
    if not games: st.info('Bu tarihte NBA maçı bulunamadı.'); st.stop()
    game=st.selectbox('NBA maçını seç',games,format_func=lambda g:f'{g[0]:%H:%M} | {g[2]} @ {g[1]} — {g[3]}')
    start,home,away,_,event_id=game
    match_day=start.astimezone(ZoneInfo('America/New_York')).date()
else:
    event_id=None
    names=sorted(t['full_name'] for t in nba_teams.get_teams())
    home=st.selectbox('Ev sahibi',names); away=st.selectbox('Deplasman',[x for x in names if x!=home])
    match_day=day-timedelta(days=1) # conservative cutoff for Turkish calendar date
cutoff=min(match_day-timedelta(days=1),now.date()-timedelta(days=1))
try:
    history, raw_players = dataset(cutoff)
    season=f'{cutoff.year if cutoff.month>=10 else cutoff.year-1}'
    data=history.groupby(['TEAM_ID','TEAM_NAME'],as_index=False).agg(GP=('PTS','size'),pts=('PTS','sum'),opp=('PTS_opp','sum'),poss=('poss','sum'))
    data['OFF_RATING']=100*data.pts/data.poss; data['DEF_RATING']=100*data.opp/data.poss; data['PACE']=data.poss/data.GP
    def team(name):
        rows=data[data.TEAM_NAME.map(normalize)==normalize(name)]
        if len(rows)!=1: raise ValueError(f'{name} için sezon verisi bulunamadı')
        r=rows.iloc[0]
        return name,float(r.OFF_RATING),float(r.DEF_RATING),float(r.PACE)
    h=team(home); v=team(away)
    base=float(np.average(data.OFF_RATING,weights=data.GP))
except Exception as exc:
    st.error(f'NBA istatistikleri alınamadı. Örnek veriyle simülasyon yapılmayacak. {exc}')
    st.info('Verileri yenile ile tekrar deneyin. ESPN geçmiş maç verileri eksik veya erişilemez olabilir.')
    st.stop()
try:
    history=history
    prior=history[history.GAME_DATE<=pd.Timestamp(cutoff)]
    for name in [home,away]:
        tid=int(data.loc[data.TEAM_NAME.map(normalize)==normalize(name),'TEAM_ID'].iloc[0])
        f=form(prior,tid,cutoff+timedelta(days=1))
        if f.empty: raise ValueError('Form verisi yok')
        st.write(name+' — son maç formu'); st.dataframe(f,hide_index=True)
        x=blend(f)
        if name==home: h=(name,*x)
        else: v=(name,*x)
    base=100*prior.PTS.sum()/prior.poss.sum()
except Exception as exc:
    st.error(f'Form verisi alınamadı: {exc}'); st.stop()
with st.expander('Oyuncu ve sakatlık raporu'):
    try:
        p=espn_roster(raw_players)
        ids=data[data.TEAM_NAME.map(normalize).isin([normalize(home),normalize(away)])].TEAM_ID
        st.dataframe(p[p.TEAM_ID.isin(ids)][['PLAYER_NAME','TEAM_ABBREVIATION','GP','MIN','PTS','AST','REB']].sort_values('MIN',ascending=False),hide_index=True)
    except Exception as exc: st.warning(f'Oyuncu verisi yok: {exc}')
    if match_day>=now.date():
        try:
            report=injuries(); found=False
            for t in report:
                if normalize(t.get('displayName','')) in [normalize(home),normalize(away)]:
                    found=True; st.write(t.get('displayName'))
                    st.dataframe([{'Oyuncu':i.get('athlete',{}).get('displayName'),'Durum':i.get('status'),'Tarih':i.get('date')} for i in t.get('injuries',[])])
            if not found: st.info('Eşleşen sakatlık kaydı yok; bu, kadronun eksiksiz olduğu anlamına gelmez.')
        except Exception as exc: st.warning(f'Sakatlık raporu alınamadı: {exc}')
    else: st.info('Geçmiş tarihli sakatlık arşivi mevcut değil.')
market_spread=market_total=None
if event_id and match_day>=now.date() and game[3].lower()=='scheduled':
    try:
        r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary',params={'event':event_id},timeout=12); r.raise_for_status()
        odds=r.json().get('pickcenter',[])
        if odds:
            o=odds[0]; market_total=float(o['overUnder']) if o.get('overUnder') is not None else None
            st.write('Piyasa kaydı:',o.get('details'), 'Toplam:',market_total,'Kaynak:',o.get('provider',{}).get('name','ESPN'))
            st.caption('Handikap yönü sağlayıcıya göre değişebildiğinden otomatik fark hesabına dahil edilmez.')
        else: st.info('Piyasa verisi bulunamadı.')
    except Exception as exc: st.warning(f'Piyasa verisi alınamadı: {exc}')
with st.expander('Geçmiş maç backtesti'):
    if st.button('Backtest çalıştır'):
        b=backtest(prior)
        if b.empty: st.warning('Yeterli geçmiş maç yok.')
        else:
            st.write('Maç sayısı:',len(b),'Kazanan doğruluğu:',float(((b.pred_home>b.pred_away)==(b.actual_home>b.actual_away)).mean()))
            st.write('Skor MAE:',float(np.abs(b[['pred_home','pred_away']].to_numpy()-b[['actual_home','actual_away']].to_numpy()).mean()))
            st.write('Toplam sayı MAE:',float(abs(b.pred_home+b.pred_away-b.actual_home-b.actual_away).mean()))
            st.dataframe(b); st.download_button('Backtest CSV',b.to_csv(index=False),'backtest.csv')
            st.caption('Backtest yalnızca Model A içindir. Yalnızca önceki günlerin takım verileri; sakatlık ve tarihsel piyasa çizgileri dahil değil.')
@st.cache_data(ttl=1800)
def player_history(season, cutoff):
    return playergamelogs.PlayerGameLogs(season_nullable=season,season_type_nullable='Regular Season',date_to_nullable=cutoff.strftime('%m/%d/%Y'),timeout=60).get_data_frames()[0]
model=st.radio('Model',['A — Takım','B — Oyuncu / rakip savunması','C — Birleşik'],index=1)
projections=[]
if model[0]!='A':
    try:
        raw=raw_players.copy()
        raw['GAME_DATE']=pd.to_datetime(raw.GAME_DATE)
        f=features(prior,raw[raw.GAME_DATE<=pd.Timestamp(cutoff)])
        roster=espn_roster(raw_players)
        pace=(h[3]+v[3])/2
        for name, opponent in [(home,v),(away,h)]:
            tid=int(data.loc[data.TEAM_NAME.map(normalize)==normalize(name),'TEAM_ID'].iloc[0])
            r=roster[roster.TEAM_ID==tid]
            excluded=st.multiselect(name+' — oynamayacak oyuncular (rapora göre kontrol edin)',r.PLAYER_NAME.tolist(),key=str(tid))
            pr=project(f,r,opponent[2],pace,excluded)
            projections.append(pr)
            st.subheader(name+' — oyuncu sayı beklentisi')
            st.dataframe(pr.round(2),hide_index=True)
            st.caption(f'Rakip savunma reytingi: {opponent[2]:.2f} | Takım oyuncu toplamı: {pr.expected.sum():.2f}')
        st.caption('Oyuncu modeli: geçmiş rakiplerin maç öncesi savunma reytingi, dakika ve tempo; az veri için sezon ortalamasına düzenlileştirme. Kadro sezon istatistiklerinden tahmin edilir. Model C ağırlığı %50/%50, deneysel.')
    except Exception as exc:
        st.error(f'Oyuncu modeli verisi alınamadı: {exc}'); st.stop()

match_key=str((home,away,day,cutoff,h,v,base))+model+str([r.to_json() for r in projections])
st.caption(f'Kaynak: ESPN kutu skorları (tahmini pozisyon sayısı) | {season} normal sezon | ABD tarihi {cutoff} dahil. Aynı günün maçları dahil edilmez.')
st.dataframe(pd.DataFrame([h,v],columns=['Takım','Offensive Rating','Defensive Rating','Pace']),hide_index=True)
st.caption(f'Lig ortalama hücum reytingi (maç ağırlıklı): {base:.2f}')
n=st.selectbox('Simülasyon adedi',[5000,10000,25000,50000,100000],index=1)

adv=2.0
sd=12.0
st.caption('Model varsayımları: ev sahibi avantajı 2 sayı, skor standart sapması 12 sayı, korelasyon 0.2; henüz kalibre edilmedi.')
if st.button('Simülasyonu başlat',type='primary'):
    pace=(h[3]+v[3])/2
    mu=np.array([pace*h[1]*v[2]/base/100+adv/2,pace*v[1]*h[2]/base/100-adv/2])
    rng=np.random.default_rng()
    x=np.maximum(0,np.rint(rng.multivariate_normal(mu,[[sd**2,.2*sd**2],[.2*sd**2,sd**2]],n))).astype(int)
    if model[0]!='A':
        shared=rng.lognormal(-0.04**2/2,0.04,n)
        ph=simulate_players(projections[0],n,rng,shared); pa=simulate_players(projections[1],n,rng,shared)
        px=np.column_stack([ph.sum(axis=1),pa.sum(axis=1)])
        if model[0]=='B': x=px
        else:
            choose=rng.random(n)<.5; x[choose]=px[choose]
        st.session_state.player_summary=pd.concat([r.assign(team=name,sim_mean=z.mean(axis=0),sim_min=z.min(axis=0),sim_max=z.max(axis=0)) for r,name,z in [(projections[0],home,ph),(projections[1],away,pa)]])
    # Resolve regulation ties using illustrative overtime scoring.
    while (tie:=x[:,0]==x[:,1]).any():
        x[tie]+=rng.poisson(11,size=(tie.sum(),2))
    df=pd.DataFrame(x,columns=['home','away']); st.session_state.result=(df,h[0],v[0],match_key)
if 'result' in st.session_state and st.session_state.result[3]==match_key:
    df,home,away,_=st.session_state.result
    if model[0]!='A' and 'player_summary' in st.session_state:
        st.subheader('Oyuncu modeli — normal süre simülasyonları')
        st.dataframe(st.session_state.player_summary.round(2),hide_index=True)
        st.download_button('Oyuncu CSV',st.session_state.player_summary.to_csv(index=False),'players.csv')
    diff=df.home-df.away; total=df.home+df.away
    m=st.columns(3); m[0].metric(home+' kazanma',f'{(diff>0).mean():.1%}'); m[1].metric(away+' kazanma',f'{(diff<0).mean():.1%}'); m[2].metric('Toplam sayı beklentisi',f'{total.mean():.2f}')
    stats=df.agg(['min','max','mean','median']).T; stats.index=[home,away]; st.dataframe(stats)
    modes=df.value_counts().head(10).reset_index(name='tekrar'); st.subheader('En sık kesin skorlar'); st.dataframe(modes)
    st.write(f'Model ev sahibi handikapı (ortalama fark): {-diff.mean():+.2f}')
    if market_total is not None:
        st.write(f'Model toplamı − piyasa toplamı: {total.mean()-market_total:+.2f}; üst: {(total>market_total).mean():.1%}, alt: {(total<market_total).mean():.1%}')
    st.subheader('Toplam skor dağılımı'); st.bar_chart(total.value_counts().sort_index())
    st.download_button('CSV indir',df.to_csv(index=False),'simulations.csv','text/csv')
    st.download_button('JSON indir',df.to_json(orient='records'),'simulations.json','application/json')

import streamlit as st
import numpy as np
import pandas as pd
st.set_page_config(page_title='NBA Analytics Pro',layout='wide')
st.title('NBA Analytics Pro v1.0')
st.warning('Deneysel model: veriler manuel girilir. Varsayılan değerler örnektir; model geçmiş maçlarla kalibre edilmemiştir.')
a,b=st.columns(2)
def team(c,label,default):
    with c:
        name=st.text_input(label,default)
        o=st.number_input('Offensive Rating',80.,150.,115.,key=label+'o')
        d=st.number_input('Defensive Rating',80.,150.,115.,key=label+'d')
        p=st.number_input('Pace',75.,125.,100.,key=label+'p')
    return name,o,d,p
h=team(a,'Ev sahibi','Boston Celtics'); v=team(b,'Deplasman','LA Lakers')
n=st.selectbox('Simülasyon adedi',[5000,10000,25000,50000,100000],index=1)
base=st.number_input('Lig ortalama rating',90.,135.,115.)
adv=st.number_input('Ev sahibi avantajı (sayı)',0.,10.,2.)
sd=st.number_input('Takım skor standart sapması (varsayım)',5.,25.,12.)
if st.button('Simülasyonu başlat',type='primary'):
    pace=(h[3]+v[3])/2
    mu=np.array([pace*h[1]*v[2]/base/100+adv/2,pace*v[1]*h[2]/base/100-adv/2])
    rng=np.random.default_rng()
    x=np.maximum(0,np.rint(rng.multivariate_normal(mu,[[sd**2,.2*sd**2],[.2*sd**2,sd**2]],n))).astype(int)
    # Resolve regulation ties using illustrative overtime scoring.
    while (tie:=x[:,0]==x[:,1]).any():
        x[tie]+=rng.poisson(11,size=(tie.sum(),2))
    df=pd.DataFrame(x,columns=['home','away']); st.session_state.result=(df,h[0],v[0])
if 'result' in st.session_state:
    df,home,away=st.session_state.result
    diff=df.home-df.away; total=df.home+df.away
    m=st.columns(3); m[0].metric(home+' kazanma',f'{(diff>0).mean():.1%}'); m[1].metric(away+' kazanma',f'{(diff<0).mean():.1%}'); m[2].metric('Toplam sayı beklentisi',f'{total.mean():.2f}')
    stats=df.agg(['min','max','mean','median']).T; stats.index=[home,away]; st.dataframe(stats)
    modes=df.value_counts().head(10).reset_index(name='tekrar'); st.subheader('En sık kesin skorlar'); st.dataframe(modes)
    st.write(f'Model ev sahibi handikapı (ortalama fark): {-diff.mean():+.2f}')
    st.subheader('Toplam skor dağılımı'); st.bar_chart(total.value_counts().sort_index())
    st.download_button('CSV indir',df.to_csv(index=False),'simulations.csv','text/csv')
    st.download_button('JSON indir',df.to_json(orient='records'),'simulations.json','application/json')

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. APP CONFIGURATION & ASSETS
# ============================================
st.set_page_config(
    page_title="PL Matchday Official",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOGO DATABASE ---
TEAM_LOGOS = {
    "Arsenal": "https://images.fotmob.com/image_resources/logo/teamlogo/9825.png",
    "Aston Villa": "https://images.fotmob.com/image_resources/logo/teamlogo/10252.png",
    "Bournemouth": "https://images.fotmob.com/image_resources/logo/teamlogo/8678.png",
    "Brentford": "https://images.fotmob.com/image_resources/logo/teamlogo/9937.png",
    "Brighton": "https://images.fotmob.com/image_resources/logo/teamlogo/10204.png",
    "Burnley": "https://images.fotmob.com/image_resources/logo/teamlogo/8191.png",
    "Chelsea": "https://images.fotmob.com/image_resources/logo/teamlogo/8455.png",
    "Crystal Palace": "https://images.fotmob.com/image_resources/logo/teamlogo/9826.png",
    "Everton": "https://images.fotmob.com/image_resources/logo/teamlogo/8668.png",
    "Fulham": "https://images.fotmob.com/image_resources/logo/teamlogo/9879.png",
    "Ipswich": "https://images.fotmob.com/image_resources/logo/teamlogo/9850.png",
    "Leeds": "https://images.fotmob.com/image_resources/logo/teamlogo/8463.png",
    "Leicester": "https://images.fotmob.com/image_resources/logo/teamlogo/8197.png",
    "Liverpool": "https://images.fotmob.com/image_resources/logo/teamlogo/8650.png",
    "Luton": "https://images.fotmob.com/image_resources/logo/teamlogo/8346.png",
    "Man City": "https://images.fotmob.com/image_resources/logo/teamlogo/8456.png",
    "Man United": "https://images.fotmob.com/image_resources/logo/teamlogo/10260.png",
    "Newcastle": "https://images.fotmob.com/image_resources/logo/teamlogo/10261.png",
    "Nott'm Forest": "https://images.fotmob.com/image_resources/logo/teamlogo/10203.png",
    "Sheffield United": "https://images.fotmob.com/image_resources/logo/teamlogo/8657.png",
    "Southampton": "https://images.fotmob.com/image_resources/logo/teamlogo/8466.png",
    "Sunderland": "https://images.fotmob.com/image_resources/logo/teamlogo/8472.png",
    "Tottenham": "https://images.fotmob.com/image_resources/logo/teamlogo/8586.png",
    "West Ham": "https://images.fotmob.com/image_resources/logo/teamlogo/8654.png",
    "Wolves": "https://images.fotmob.com/image_resources/logo/teamlogo/8602.png"
}

def get_logo(team_name):
    return TEAM_LOGOS.get(team_name, "https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg")

# ============================================
# 2. PREMIER LEAGUE THEME CSS
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;900&display=swap');

    :root {
        --pl-purple: #38003c;
        --pl-green: #00ff85;
        --pl-pink: #e90052;
        --pl-white: #ffffff;
    }

    /* GLOBAL BACKGROUND */
    .stApp {
        background-color: var(--pl-purple);
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        color: var(--pl-white);
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
    }

    /* DROPDOWNS & INPUTS */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: white !important;
        color: var(--pl-purple) !important;
        border-radius: 4px;
        border: none;
    }
    div[data-baseweb="select"] span {
        color: var(--pl-purple) !important; 
        font-weight: 700;
    }
    div[data-baseweb="menu"] {
        background-color: white !important;
    }
    div[data-baseweb="option"] {
        color: var(--pl-purple) !important;
    }

    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(90deg, #e90052 0%, #ff0055 100%);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 0, 82, 0.4);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 0, 82, 0.6);
        background: #ff0a6c;
    }

    /* METRICS */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid var(--pl-green);
    }
    div[data-testid="stMetricValue"] {
        color: var(--pl-green) !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: white !important;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* TABLES (DATAFRAMES) */
    div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stDataFrame"] div[role="grid"] {
        color: var(--pl-purple);
    }
    thead tr th {
        background-color: var(--pl-purple) !important;
        color: white !important;
    }

    /* CUSTOM LAYOUT CLASSES */
    .team-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    .logo-img {
        height: 160px;
        margin: 20px 0;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
        transition: transform 0.3s;
    }
    .logo-img:hover { transform: scale(1.05); }
    
    .elo-badge {
        background: white;
        color: var(--pl-purple);
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 900;
        margin-top: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# 3. GOD MODE BACKEND
# ============================================

class GodModeEngine:
    def __init__(self):
        self.SEASONS = ['1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']
        self.ODDS_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
        self.master_df = None
        self.model = None
        self.scaler = None
        self.features = ['Elo_Diff', 'EMA_SOT_Diff', 'EMA_Corn_Diff', 'Eff_Trend_Diff']
        self.current_teams = []
        self.curr_elo_dict = {}

    def load_data(self):
        dfs = []
        for s in self.SEASONS:
            try:
                c = requests.get(self.ODDS_URL.format(s)).content
                df = pd.read_csv(io.StringIO(c.decode('latin-1')))
                df = df.dropna(how='all')
                
                # Capture current season teams
                if s == '2526' or s == '2425': 
                    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                    self.current_teams = sorted(teams)
                
                cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC']
                df = df[[c for c in cols if c in df.columns]]
                dfs.append(df)
            except: pass
        
        if not dfs: return False
        
        df = pd.concat(dfs, ignore_index=True)
        col_map = {'Date':'date', 'HomeTeam':'home_team', 'AwayTeam':'away_team', 
                   'FTHG':'home_goals', 'FTAG':'away_goals', 
                   'HST':'home_shots_on_target', 'AST':'away_shots_on_target', 
                   'HC':'home_corners', 'AC':'away_corners'}
        df.rename(columns=col_map, inplace=True)
        
        # --- FIX: ROBUST DATE PARSING ---
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date']) # Drop invalid dates immediately
        df = df.sort_values('date').reset_index(drop=True)
        
        for c in ['home_shots_on_target', 'away_shots_on_target', 'home_corners', 'away_corners']:
            df[c] = df[c].fillna(df[c].mean())
            
        self.master_df = df
        return True

    def engineer_features(self):
        df = self.master_df.copy()
        
        # ELO
        df['home_elo'] = 1500.0
        df['away_elo'] = 1500.0
        curr_elo = {t: 1500.0 for t in pd.concat([df['home_team'], df['away_team']]).unique()}
        k = 20

        for i, row in df.iterrows():
            h, a = row['home_team'], row['away_team']
            h_elo, a_elo = curr_elo.get(h, 1500), curr_elo.get(a, 1500)
            df.at[i, 'home_elo'] = h_elo
            df.at[i, 'away_elo'] = a_elo

            if row['home_goals'] > row['away_goals']: res = 1
            elif row['home_goals'] == row['away_goals']: res = 0.5
            else: res = 0
            dr = h_elo - a_elo
            e_h = 1 / (1 + 10**(-dr/400))
            curr_elo[h] += k * (res - e_h)
            curr_elo[a] += k * ((1-res) - (1-e_h))
            
        self.curr_elo_dict = curr_elo 

        # EMA (Form)
        def create_stream(df):
            h = df[['date', 'home_team', 'home_goals', 'home_shots_on_target', 'home_corners']].copy()
            h.columns = ['date', 'team', 'goals', 'sot', 'corners']
            a = df[['date', 'away_team', 'away_goals', 'away_shots_on_target', 'away_corners']].copy()
            a.columns = ['date', 'team', 'goals', 'sot', 'corners']
            return pd.concat([h, a]).sort_values(['team', 'date'])

        stream = create_stream(df)
        cols = ['goals', 'sot', 'corners']
        stream_ema = stream.groupby('team')[cols].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
        stream = pd.concat([stream, stream_ema.add_prefix('ema_')], axis=1)

        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'h_ema_goals', 'ema_sot':'h_ema_sot', 'ema_corners':'h_ema_corn'}).drop(columns=['team'])
        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'a_ema_goals', 'ema_sot':'a_ema_sot', 'ema_corners':'a_ema_corn'}).drop(columns=['team'])

        # DIFFS
        df['Elo_Diff'] = df['home_elo'] - df['away_elo']
        df['EMA_SOT_Diff'] = df['h_ema_sot'] - df['a_ema_sot']
        df['EMA_Corn_Diff'] = df['h_ema_corn'] - df['a_ema_corn']
        
        h_eff = df['h_ema_goals'] / (df['h_ema_sot'] + 0.1)
        a_eff = df['a_ema_goals'] / (df['a_ema_sot'] + 0.1)
        df['Eff_Trend_Diff'] = h_eff - a_eff

        conditions = [df['home_goals'] > df['away_goals'], df['home_goals'] == df['away_goals']]
        df['target'] = np.select(conditions, [2, 1], default=0)
        
        self.master_df = df.dropna(subset=self.features).copy()

    def train_trinity_model(self):
        df = self.master_df
        X = df[self.features]
        y = df['target']
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        weights = np.exp(np.linspace(0, 4, len(X)))
        
        # TRINITY ENSEMBLE
        lr = LogisticRegression(C=0.05, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_mod = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, 
                                    objective='multi:softmax', num_class=3, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_mod)],
            voting='soft', weights=[1, 1, 3]
        )
        self.model.fit(X_scaled, y, sample_weight=weights)

    def predict_match(self, h_team, a_team):
        row_h = self.master_df[(self.master_df['home_team'] == h_team) | (self.master_df['away_team'] == h_team)].iloc[-1]
        row_a = self.master_df[(self.master_df['home_team'] == a_team) | (self.master_df['away_team'] == a_team)].iloc[-1]
        
        def get_stat(row, team, stat): return row[f'h_{stat}'] if row['home_team'] == team else row[f'a_{stat}']
        
        h_elo = self.curr_elo_dict.get(h_team, 1500)
        a_elo = self.curr_elo_dict.get(a_team, 1500)

        elo_diff = h_elo - a_elo
        sot_diff = get_stat(row_h, h_team, 'ema_sot') - get_stat(row_a, a_team, 'ema_sot')
        corn_diff = get_stat(row_h, h_team, 'ema_corn') - get_stat(row_a, a_team, 'ema_corn')
        eff_diff = (get_stat(row_h, h_team, 'ema_goals')/(get_stat(row_h, h_team, 'ema_sot')+0.1)) - (get_stat(row_a, a_team, 'ema_goals')/(get_stat(row_a, a_team, 'ema_sot')+0.1))
        
        input_vec = pd.DataFrame([[elo_diff, sot_diff, corn_diff, eff_diff]], columns=self.features)
        input_scaled = self.scaler.transform(input_vec)
        
        probs = self.model.predict_proba(input_scaled)[0]
        return {'A': probs[0], 'D': probs[1], 'H': probs[2], 'H_Elo': int(h_elo), 'A_Elo': int(a_elo)}

    def get_history(self, n=20):
        # Validation on last N matches in dataset
        recent = self.master_df.tail(n).copy()
        X = recent[self.features]
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
        results = []
        for i, (idx, row) in enumerate(recent.iterrows()):
            p_a, p_d, p_h = probs[i]
            if p_h > p_a and p_h > p_d: pred = "H"
            elif p_a > p_h and p_a > p_d: pred = "A"
            else: pred = "D"
            
            actual = "H" if row['target']==2 else "A" if row['target']==0 else "D"
            match_name = f"{row['home_team']} vs {row['away_team']}"
            
            # --- FIX: SAFE DATE FORMATTING ---
            date_str = "N/A"
            if pd.notnull(row['date']):
                try:
                    date_str = row['date'].strftime('%d %b')
                except: pass

            results.append({
                "Date": date_str,
                "Match": match_name,
                "Prediction": pred,
                "Result": actual,
                "Status": "✅" if pred == actual else "❌"
            })
        return pd.DataFrame(results)

# ============================================
# 4. INITIALIZATION
# ============================================
from datetime import date

@st.cache_resource
def load_pl_v6_safe_dates(version):
    eng = GodModeEngine()
    if eng.load_data():
        eng.engineer_features()
        eng.train_trinity_model()
        return eng
    return None

engine = load_pl_v6_safe_dates(version=date.today().isoformat())
if not engine: st.stop()

# ============================================
# 5. UI & PAGES
# ============================================

# HEADER
st.markdown("""
<div style="background-color: #38003c; padding: 20px; border-bottom: 4px solid #00ff85; margin-bottom: 25px;">
    <h1 style="color: white; margin:0; font-size: 3rem;">PREMIER LEAGUE <span style="color: #00ff85">PREDICTOR</span></h1>
    <p style="color: #e0e0e0; margin:0; font-size: 1.1rem;">OFFICIAL MATCHDAY INSIGHTS</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR NAV
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=150)
    page = st.radio("MENU", ["Match Centre", "Standings (Elo)", "History"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Ver: 6.0 - Robust")
with st.sidebar:
    if st.button("🔄 Force data refresh"):
        st.cache_resource.clear()
        st.rerun()


current_teams = engine.current_teams

# --- PAGE 1: MATCH CENTRE ---
if page == "Match Centre":
    col1, col2, col3 = st.columns([1, 0.3, 1])
    
    # Session state for logos
    if 'h_team' not in st.session_state: st.session_state.h_team = current_teams[0]
    if 'a_team' not in st.session_state: st.session_state.a_team = current_teams[1]

    with col1:
        st.markdown('<div class="team-col">', unsafe_allow_html=True)
        st.markdown("### HOME CLUB")
        h_team = st.selectbox("H", current_teams, index=current_teams.index(st.session_state.h_team), key="h_sel", label_visibility="collapsed")
        st.markdown(f'<img src="{get_logo(h_team)}" class="logo-img">', unsafe_allow_html=True)
        
        h_elo = int(engine.curr_elo_dict.get(h_team, 1500))
        st.markdown(f'<div class="elo-badge">ELO: {h_elo}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #e90052 !important; font-size: 4rem !important;'>VS</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="team-col">', unsafe_allow_html=True)
        st.markdown("### AWAY CLUB")
        a_team = st.selectbox("A", current_teams, index=current_teams.index(st.session_state.a_team), key="a_sel", label_visibility="collapsed")
        st.markdown(f'<img src="{get_logo(a_team)}" class="logo-img">', unsafe_allow_html=True)
        
        a_elo = int(engine.curr_elo_dict.get(a_team, 1500))
        st.markdown(f'<div class="elo-badge">ELO: {a_elo}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("PREDICT MATCH OUTCOME"):
        if h_team == a_team:
            st.warning("Please select different clubs.")
        else:
            with st.spinner("Analyzing stats..."):
                pred = engine.predict_match(h_team, a_team)
                
                st.markdown("---")
                st.markdown(f"<h3 style='text-align: center; margin-bottom: 20px;'>FULL TIME PROBABILITY</h3>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("HOME WIN", f"{pred['H']*100:.1f}%")
                c2.metric("DRAW", f"{pred['D']*100:.1f}%")
                c3.metric("AWAY WIN", f"{pred['A']*100:.1f}%")
                
                bar_html = f"""
                <div style="margin-top: 20px; width: 100%; height: 30px; display: flex; border-radius: 15px; overflow: hidden; background: #333;">
                    <div style="width: {pred['H']*100}%; background: #00ff85; display:flex; align-items:center; justify-content:center; color:#38003c; font-weight:bold;">{int(pred['H']*100)}%</div>
                    <div style="width: {pred['D']*100}%; background: #e0e0e0; display:flex; align-items:center; justify-content:center; color:#333; font-weight:bold;">D</div>
                    <div style="width: {pred['A']*100}%; background: #e90052; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">{int(pred['A']*100)}%</div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

# --- PAGE 2: STANDINGS (ELO) ---
elif page == "Standings (Elo)":
    st.markdown("### 🏆 LIVE POWER RANKINGS (ELO)")
    st.caption("Calculated based on match performance over the last 10 seasons.")
    
    elo_data = pd.DataFrame(list(engine.curr_elo_dict.items()), columns=['Club', 'Rating'])
    elo_data = elo_data[elo_data['Club'].isin(current_teams)]
    elo_data['Rating'] = elo_data['Rating'].astype(int)
    elo_data = elo_data.sort_values('Rating', ascending=False).reset_index(drop=True)
    elo_data.index += 1
    
    st.dataframe(
        elo_data,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.ProgressColumn(
                "Rating",
                format="%d",
                min_value=1300,
                max_value=2100,
            ),
        },
        height=800
    )

# --- PAGE 3: HISTORY ---
elif page == "History":
    st.markdown("### ⏪ PREDICTION HISTORY (LAST 20 GAMES)")
    st.caption("How the model performed on the most recent matches in the database.")
    
    hist_df = engine.get_history(20)
    
    # Calculate accuracy
    acc = len(hist_df[hist_df['Status']=='✅']) / len(hist_df)
    st.metric("Recent Accuracy", f"{acc*100:.0f}%")
    
    def highlight(s):
        return ['background-color: #00ff85; color: #38003c' if v == '✅' else 'background-color: #e90052; color: white' if v == '❌' else '' for v in s]

    st.dataframe(hist_df.style.apply(highlight, subset=['Status']), use_container_width=True)



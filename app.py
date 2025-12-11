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
# 1. APP CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ENGLISH PREMIER LEAGUE PREDICTOR",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOGO DATABASE (FotMob High-Res) ---
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
# 2. HIGH-VOLTAGE CSS THEME
# ============================================
st.markdown("""
    <style>
    /* IMPORT GAMING FONT */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700;800&display=swap');

    :root {
        --neon-green: #39ff14;
        --neon-blue: #00f3ff;
        --neon-pink: #ff0055;
        --dark-bg: #0b0c10;
        --card-bg: rgba(31, 40, 51, 0.8);
    }

    /* GLOBAL STYLES */
    .stApp {
        background-color: var(--dark-bg);
        background-image: radial-gradient(circle at center top, #1f2833 0%, #0b0c10 70%);
        color: white;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.3);
    }

    /* CUSTOM DROPDOWNS (DARK MODE) */
    div[data-baseweb="select"] > div {
        background-color: #1f2833 !important;
        color: white !important;
        border: 2px solid #45a29e;
        border-radius: 0px; /* Sharp Edges */
    }
    div[data-baseweb="select"] span {
        color: white !important;
        font-weight: 700;
        font-size: 1.2rem;
    }
    div[data-baseweb="menu"] {
        background-color: #0b0c10 !important;
        border: 1px solid var(--neon-blue);
    }

    /* INPUT FIELDS (ODDS) */
    div[data-baseweb="input"] > div {
        background-color: #1f2833 !important;
        color: white !important;
        border: 1px solid #66fcf1;
    }
    input { color: white !important; font-weight: bold; }

    /* ANGLED BUTTONS (CYBERPUNK STYLE) */
    div.stButton > button {
        background: linear-gradient(45deg, var(--neon-blue), #45a29e);
        color: black;
        border: none;
        clip-path: polygon(10% 0%, 100% 0%, 100% 80%, 90% 100%, 0% 100%, 0% 20%);
        padding: 15px 0;
        font-weight: 900;
        font-size: 22px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 0 15px rgba(102, 252, 241, 0.3);
    }
    div.stButton > button:hover {
        background: white;
        transform: scale(1.02);
        box-shadow: 0 0 30px var(--neon-blue);
    }

    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* METRICS OVERRIDE */
    div[data-testid="stMetric"] {
        background-color: #1f2833;
        border-left: 5px solid var(--neon-green);
        padding: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: var(--neon-green) !important;
        font-size: 2rem !important;
        text-shadow: 0 0 10px var(--neon-green);
    }
    div[data-testid="stMetricLabel"] { color: #ccc !important; }

    /* LAYOUT HELPERS */
    .team-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .logo-img {
        height: 150px;
        filter: drop-shadow(0 0 20px rgba(102, 252, 241, 0.2));
        transition: transform 0.3s;
    }
    .logo-img:hover { transform: scale(1.1); }
    
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# 3. GOD MODE BACKEND ENGINE
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
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
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
            
        # EMA
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
        self.curr_elo_dict = curr_elo # Store for UI display

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
        # Fetch latest stats
        row_h = self.master_df[(self.master_df['home_team'] == h_team) | (self.master_df['away_team'] == h_team)].iloc[-1]
        row_a = self.master_df[(self.master_df['home_team'] == a_team) | (self.master_df['away_team'] == a_team)].iloc[-1]
        
        def get_stat(row, team, stat): return row[f'h_{stat}'] if row['home_team'] == team else row[f'a_{stat}']
        
        # Elo from our live dictionary to be most current
        h_elo = self.curr_elo_dict.get(h_team, 1500)
        a_elo = self.curr_elo_dict.get(a_team, 1500)

        elo_diff = h_elo - a_elo
        sot_diff = get_stat(row_h, h_team, 'ema_sot') - get_stat(row_a, a_team, 'ema_sot')
        corn_diff = get_stat(row_h, h_team, 'ema_corn') - get_stat(row_a, a_team, 'ema_corn')
        
        h_eff = get_stat(row_h, h_team, 'ema_goals') / (get_stat(row_h, h_team, 'ema_sot') + 0.1)
        a_eff = get_stat(row_a, a_team, 'ema_goals') / (get_stat(row_a, a_team, 'ema_sot') + 0.1)
        eff_diff = h_eff - a_eff
        
        input_vec = pd.DataFrame([[elo_diff, sot_diff, corn_diff, eff_diff]], columns=self.features)
        input_scaled = self.scaler.transform(input_vec)
        
        probs = self.model.predict_proba(input_scaled)[0]
        return {'A': probs[0], 'D': probs[1], 'H': probs[2], 'H_Elo': int(h_elo), 'A_Elo': int(a_elo)}

# ============================================
# 4. INIT ENGINE
# ============================================
@st.cache_resource
def load_elite_engine():
    eng = GodModeEngine()
    if eng.load_data():
        eng.engineer_features()
        eng.train_trinity_model()
        return eng
    return None

engine = load_elite_engine()
if not engine: st.stop()

# ============================================
# 5. UI LAYOUT
# ============================================

# HEADER
st.markdown("""
<div style="text-align: center; margin-bottom: 40px;">
    <h1 style="font-size: 4rem; margin: 0; color: #fff; letter-spacing: 5px;">MATCHDAY <span style="color: #39ff14;">ELITE</span></h1>
    <p style="color: #66fcf1; letter-spacing: 2px; font-weight: bold;">AI-POWERED PREMIER LEAGUE INTELLIGENCE</p>
</div>
""", unsafe_allow_html=True)

# MAIN ARENA
col1, col2, col3 = st.columns([1, 0.3, 1])

# --- HOME ---
with col1:
    st.markdown('<div class="team-container">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #66fcf1;'>HOME SQUAD</h3>", unsafe_allow_html=True)
    h_team = st.selectbox("Select Home", engine.current_teams, index=0, label_visibility="collapsed", key="h_sel")
    st.markdown(f'<img src="{get_logo(h_team)}" class="logo-img">', unsafe_allow_html=True)
    
    # Live Elo Tag
    h_elo = int(engine.curr_elo_dict.get(h_team, 1500))
    st.markdown(f"""
        <div style="margin-top: 15px; border: 1px solid #39ff14; color: #39ff14; padding: 5px 15px; font-weight: bold; border-radius: 20px;">
            PWR: {h_elo}
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- VS ---
with col2:
    st.markdown("<div style='height: 80px'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 5rem !important; color: #ff0055; text-shadow: 0 0 30px #ff0055;'>VS</h1>", unsafe_allow_html=True)

# --- AWAY ---
with col3:
    st.markdown('<div class="team-container">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: #ff0055;'>AWAY SQUAD</h3>", unsafe_allow_html=True)
    a_team = st.selectbox("Select Away", engine.current_teams, index=1, label_visibility="collapsed", key="a_sel")
    st.markdown(f'<img src="{get_logo(a_team)}" class="logo-img">', unsafe_allow_html=True)
    
    # Live Elo Tag
    a_elo = int(engine.curr_elo_dict.get(a_team, 1500))
    st.markdown(f"""
        <div style="margin-top: 15px; border: 1px solid #ff0055; color: #ff0055; padding: 5px 15px; font-weight: bold; border-radius: 20px;">
            PWR: {a_elo}
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ODDS & ACTION
st.write("")
st.write("")
st.markdown("### 🎲 LIVE ODDS (OPTIONAL FOR VALUE CHECK)")
c1, c2, c3 = st.columns(3)
with c1: h_odds = st.number_input("Home Win Odds", 1.0, 20.0, 2.00, 0.01)
with c2: d_odds = st.number_input("Draw Odds", 1.0, 20.0, 3.50, 0.01)
with c3: a_odds = st.number_input("Away Win Odds", 1.0, 20.0, 4.00, 0.01)

st.write("")
if st.button("INITIATE PREDICTION SEQUENCE"):
    if h_team == a_team:
        st.warning("ERROR: SELECT DIFFERENT TEAMS")
    else:
        with st.spinner("CRUNCHING TACTICAL DATA..."):
            pred = engine.predict_match(h_team, a_team)
            
            # --- RESULTS SECTION ---
            st.markdown("---")
            
            # 1. WIN PROBABILITY BAR
            p_h, p_d, p_a = pred['H'], pred['D'], pred['A']
            st.markdown(f"""
            <div style="margin-bottom: 10px; display: flex; justify-content: space-between; font-weight: bold; color: #ccc;">
                <span>{h_team.upper()} ({int(p_h*100)}%)</span>
                <span>DRAW ({int(p_d*100)}%)</span>
                <span>{a_team.upper()} ({int(p_a*100)}%)</span>
            </div>
            <div style="width: 100%; height: 35px; background: #333; display: flex; border-radius: 5px; overflow: hidden; box-shadow: 0 0 15px rgba(0,0,0,0.5);">
                <div style="width: {p_h*100}%; background: #39ff14; box-shadow: 0 0 20px #39ff14;"></div>
                <div style="width: {p_d*100}%; background: #2c3e50;"></div>
                <div style="width: {p_a*100}%; background: #ff0055; box-shadow: 0 0 20px #ff0055;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            
            # 2. METRICS ROW
            m1, m2, m3 = st.columns(3)
            m1.metric("HOME CONFIDENCE", f"{p_h*100:.1f}%")
            m2.metric("DRAW RISK", f"{p_d*100:.1f}%")
            m3.metric("AWAY CONFIDENCE", f"{p_a*100:.1f}%")
            
            # 3. VALUE BETTING CARD
            st.markdown("<br>", unsafe_allow_html=True)
            ev_h = (p_h * h_odds) - 1
            ev_d = (p_d * d_odds) - 1
            ev_a = (p_a * a_odds) - 1
            best_ev = max(ev_h, ev_d, ev_a)
            
            if best_ev > 0.03: # 3% Edge Threshold
                if best_ev == ev_h: 
                    rec_text = f"BET HOME: {h_team.upper()}"
                    border_col = "#39ff14"
                elif best_ev == ev_d: 
                    rec_text = "BET DRAW"
                    border_col = "#66fcf1"
                else: 
                    rec_text = f"BET AWAY: {a_team.upper()}"
                    border_col = "#ff0055"
                
                st.markdown(f"""
                <div style="background: rgba(0,0,0,0.6); border: 2px solid {border_col}; padding: 20px; text-align: center; border-radius: 10px;">
                    <h2 style="color: {border_col}; margin: 0; text-shadow: 0 0 10px {border_col};">💎 HIGH VALUE DETECTED</h2>
                    <h1 style="color: white; margin: 10px 0;">{rec_text}</h1>
                    <p style="color: #ccc; margin: 0;">CALCULATED EDGE: <span style="color: white; font-weight: bold;">+{best_ev*100:.1f}%</span> ROI</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(0,0,0,0.6); border: 2px solid #555; padding: 20px; text-align: center; border-radius: 10px;">
                    <h3 style="color: #888; margin: 0;">🚫 NO VALUE PLAY</h3>
                    <p style="color: #666; margin: 5px 0;">Bookmaker odds are accurate. No statistical edge found.</p>
                </div>
                """, unsafe_allow_html=True)

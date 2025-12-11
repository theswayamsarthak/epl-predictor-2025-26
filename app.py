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
    page_title="FC26 GOD MODE",
    page_icon="🧠",
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
# 2. CSS STYLING (FC26 THEME)
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700;800&display=swap');
    :root { --neon-green: #32f99a; --neon-pink: #ff0055; --dark-bg: #0e0e10; }
    
    .stApp { background-color: var(--dark-bg); color: white; font-family: 'Rajdhani', sans-serif; }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; font-weight: 800; }
    
    /* DROPDOWNS */
    div[data-baseweb="select"] > div { background-color: #1a1a1d !important; color: white !important; border: 1px solid #333; }
    div[data-baseweb="select"] span { color: white !important; }
    
    /* INPUTS */
    div[data-baseweb="input"] > div { background-color: #1a1a1d !important; color: white !important; border: 1px solid #333; }
    input { color: white !important; }

    /* LOGO & ALIGNMENT */
    .team-col { display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .team-logo { height: 140px; margin: 15px 0; filter: drop-shadow(0 0 10px rgba(255,255,255,0.2)); }
    
    /* BUTTONS */
    div.stButton > button {
        background: var(--neon-green); color: black; border: none;
        clip-path: polygon(5% 0%, 100% 0%, 100% 90%, 95% 100%, 0% 100%, 0% 10%);
        padding: 15px 30px; font-weight: 800; font-size: 20px; width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover { background: white; box-shadow: 0 0 20px var(--neon-green); }

    /* METRICS */
    div[data-testid="stMetric"] { background: #1a1a1d; border-left: 4px solid var(--neon-green); padding: 10px; }
    div[data-testid="stMetricValue"] { color: var(--neon-green) !important; font-size: 24px !important; }
    
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
        # 1. FETCH REMOTE DATA
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
                
                # Select only needed cols
                cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC']
                df = df[[c for c in cols if c in df.columns]]
                dfs.append(df)
            except: pass
        
        if not dfs: return False
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Standardize Names
        col_map = {'Date':'date', 'HomeTeam':'home_team', 'AwayTeam':'away_team', 
                   'FTHG':'home_goals', 'FTAG':'away_goals', 
                   'HST':'home_shots_on_target', 'AST':'away_shots_on_target', 
                   'HC':'home_corners', 'AC':'away_corners'}
        df.rename(columns=col_map, inplace=True)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing stats with avg
        for c in ['home_shots_on_target', 'away_shots_on_target', 'home_corners', 'away_corners']:
            df[c] = df[c].fillna(df[c].mean())
            
        self.master_df = df
        return True

    def engineer_features(self):
        df = self.master_df.copy()
        
        # 1. ELO CALCULATION
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
            
        # 2. EMA CALCULATION (Form)
        def create_stream(df):
            h = df[['date', 'home_team', 'home_goals', 'home_shots_on_target', 'home_corners']].copy()
            h.columns = ['date', 'team', 'goals', 'sot', 'corners']
            a = df[['date', 'away_team', 'away_goals', 'away_shots_on_target', 'away_corners']].copy()
            a.columns = ['date', 'team', 'goals', 'sot', 'corners']
            return pd.concat([h, a]).sort_values(['team', 'date'])

        stream = create_stream(df)
        cols = ['goals', 'sot', 'corners']
        stream_ema = stream.groupby('team')[cols].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean()) # Span 5 is smoother
        stream = pd.concat([stream, stream_ema.add_prefix('ema_')], axis=1)

        # Merge EMA back
        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'h_ema_goals', 'ema_sot':'h_ema_sot', 'ema_corners':'h_ema_corn'}).drop(columns=['team'])
        
        df = df.merge(stream[['date', 'team', 'ema_goals', 'ema_sot', 'ema_corners']], 
                      left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').rename(columns={'ema_goals':'a_ema_goals', 'ema_sot':'a_ema_sot', 'ema_corners':'a_ema_corn'}).drop(columns=['team'])

        # 3. DIFFERENTIALS
        df['Elo_Diff'] = df['home_elo'] - df['away_elo']
        df['EMA_SOT_Diff'] = df['h_ema_sot'] - df['a_ema_sot']
        df['EMA_Corn_Diff'] = df['h_ema_corn'] - df['a_ema_corn']
        
        h_eff = df['h_ema_goals'] / (df['h_ema_sot'] + 0.1)
        a_eff = df['a_ema_goals'] / (df['a_ema_sot'] + 0.1)
        df['Eff_Trend_Diff'] = h_eff - a_eff

        # Target
        conditions = [df['home_goals'] > df['away_goals'], df['home_goals'] == df['away_goals']]
        df['target'] = np.select(conditions, [2, 1], default=0)
        
        self.master_df = df.dropna(subset=self.features).copy()

    def train_trinity_model(self):
        df = self.master_df
        X = df[self.features]
        y = df['target']
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # TIME DECAY WEIGHTS
        weights = np.exp(np.linspace(0, 4, len(X)))
        
        # TRINITY ENSEMBLE
        lr = LogisticRegression(multi_class='multinomial', C=0.05, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_mod = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, 
                                    objective='multi:softmax', num_class=3, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_mod)],
            voting='soft', weights=[1, 1, 3] # Heavy weight on XGBoost
        )
        
        self.model.fit(X_scaled, y, sample_weight=weights)

    def predict_match(self, h_team, a_team):
        # Get latest data
        row_h = self.master_df[(self.master_df['home_team'] == h_team) | (self.master_df['away_team'] == h_team)].iloc[-1]
        row_a = self.master_df[(self.master_df['home_team'] == a_team) | (self.master_df['away_team'] == a_team)].iloc[-1]
        
        def get_stat(row, team, stat): return row[f'h_{stat}'] if row['home_team'] == team else row[f'a_{stat}']
        def get_elo(row, team): return row['home_elo'] if row['home_team'] == team else row['away_elo']

        # Calc Inputs
        elo_diff = get_elo(row_h, h_team) - get_elo(row_a, a_team)
        sot_diff = get_stat(row_h, h_team, 'ema_sot') - get_stat(row_a, a_team, 'ema_sot')
        corn_diff = get_stat(row_h, h_team, 'ema_corn') - get_stat(row_a, a_team, 'ema_corn')
        
        h_eff = get_stat(row_h, h_team, 'ema_goals') / (get_stat(row_h, h_team, 'ema_sot') + 0.1)
        a_eff = get_stat(row_a, a_team, 'ema_goals') / (get_stat(row_a, a_team, 'ema_sot') + 0.1)
        eff_diff = h_eff - a_eff
        
        input_vec = pd.DataFrame([[elo_diff, sot_diff, corn_diff, eff_diff]], columns=self.features)
        input_scaled = self.scaler.transform(input_vec)
        
        probs = self.model.predict_proba(input_scaled)[0]
        return {'A': probs[0], 'D': probs[1], 'H': probs[2], 
                'H_Elo': int(get_elo(row_h, h_team)), 'A_Elo': int(get_elo(row_a, a_team))}

# ============================================
# 4. INITIALIZATION
# ============================================
@st.cache_resource
def load_god_mode():
    eng = GodModeEngine()
    eng.load_data()
    eng.engineer_features()
    eng.train_trinity_model()
    return eng

engine = load_god_mode()

if not engine: st.stop()

# ============================================
# 5. UI LAYOUT
# ============================================

st.markdown("<h1 style='text-align: center; color: #32f99a;'>GOD MODE <span style='color: white;'>PREDICTOR</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>POWERED BY TRINITY ENSEMBLE (XGBOOST + RF + LR)</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 0.2, 1])

# HOME SELECTION
with col1:
    st.markdown('<div class="team-col">', unsafe_allow_html=True)
    st.markdown("### HOME")
    h_team = st.selectbox("Select Home", engine.current_teams, index=0, label_visibility="collapsed")
    st.image(get_logo(h_team), width=130)
    st.markdown('</div>', unsafe_allow_html=True)

# VS
with col2:
    st.markdown("<br><br><h1 style='text-align: center; color: #ff0055;'>VS</h1>", unsafe_allow_html=True)

# AWAY SELECTION
with col3:
    st.markdown('<div class="team-col">', unsafe_allow_html=True)
    st.markdown("### AWAY")
    a_team = st.selectbox("Select Away", engine.current_teams, index=1, label_visibility="collapsed")
    st.image(get_logo(a_team), width=130)
    st.markdown('</div>', unsafe_allow_html=True)

# ODDS INPUT (For Value Betting)
st.markdown("### 💰 ENTER BOOKMAKER ODDS (Optional)")
c_o1, c_o2, c_o3 = st.columns(3)
with c_o1: h_odds = st.number_input("Home Odds", value=2.00, step=0.01)
with c_o2: d_odds = st.number_input("Draw Odds", value=3.20, step=0.01)
with c_o3: a_odds = st.number_input("Away Odds", value=3.50, step=0.01)

st.write("")
if st.button("RUN TRINITY PREDICTION"):
    if h_team == a_team:
        st.warning("Teams must be different.")
    else:
        with st.spinner("Calculating Exponential Moving Averages..."):
            pred = engine.predict_match(h_team, a_team)
            
            # DISPLAY RESULTS
            st.markdown("---")
            
            # 1. ELO COMPARISON
            c1, c2 = st.columns(2)
            c1.metric(f"{h_team} Elo", pred['H_Elo'])
            c2.metric(f"{a_team} Elo", pred['A_Elo'])
            
            # 2. PROBABILITIES
            st.markdown("### 📊 WIN PROBABILITY")
            p_h, p_d, p_a = pred['H'], pred['D'], pred['A']
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("HOME WIN", f"{p_h*100:.1f}%")
            col_b.metric("DRAW", f"{p_d*100:.1f}%")
            col_c.metric("AWAY WIN", f"{p_a*100:.1f}%")
            
            # Visual Bar
            bar_html = f"""
            <div style="width: 100%; height: 30px; display: flex; border-radius: 5px; overflow: hidden; margin-top: 10px;">
                <div style="width: {p_h*100}%; background: #32f99a;"></div>
                <div style="width: {p_d*100}%; background: #555;"></div>
                <div style="width: {p_a*100}%; background: #ff0055;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; color: #888; font-size: 12px;">
                <span>HOME</span><span>AWAY</span>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            
            # 3. VALUE BETTING CALCULATOR
            st.markdown("### 💎 BETTING VALUE DETECTOR")
            
            ev_h = (p_h * h_odds) - 1
            ev_d = (p_d * d_odds) - 1
            ev_a = (p_a * a_odds) - 1
            
            best_ev = max(ev_h, ev_d, ev_a)
            
            if best_ev > 0.05: # 5% Edge
                if best_ev == ev_h: rec = f"BET HOME ({h_team})"
                elif best_ev == ev_d: rec = "BET DRAW"
                else: rec = f"BET AWAY ({a_team})"
                
                st.success(f"✅ **VALUE FOUND:** {rec} (EV: {best_ev*100:.1f}%)")
                st.caption("The model thinks the odds are higher than the real risk.")
            else:
                st.error("🛑 NO VALUE BET FOUND")
                st.caption("The bookmaker odds are too stingy compared to our calculated probability. Stay away.")

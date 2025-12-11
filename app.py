import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. EA FC THEME CONFIGURATION
# ============================================
st.set_page_config(
    page_title="FC26 PREDICTOR",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME CSS (THE "GAME MENU" LOOK) ---
st.markdown("""
    <style>
    /* IMPORT FONTS: Rajdhani (Futuristic/Gaming) */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');

    /* 1. BACKGROUND: Dark Navy/Black Geometric */
    .stApp {
        background-color: #0e0e10;
        background-image: radial-gradient(circle at 50% 0%, #1a2c38 0%, #0e0e10 70%);
        color: white;
    }

    /* 2. TYPOGRAPHY: All Caps, Sharp */
    h1, h2, h3, .stButton button, .css-10trblm {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        font-weight: 800 !important;
        letter-spacing: 1.5px;
    }
    
    h1 { font-size: 3.5rem !important; color: white; text-shadow: 0px 0px 10px rgba(50, 249, 154, 0.5); }
    h2 { font-size: 2rem !important; border-left: 5px solid #32f99a; padding-left: 15px; }
    
    /* 3. NAVIGATION (SIDEBAR) */
    [data-testid="stSidebar"] {
        background-color: #08080a;
        border-right: 1px solid #333;
    }
    
    /* 4. "ULTIMATE TEAM" CARD STYLE METRICS */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e24 0%, #121214 100%);
        border: 1px solid #333;
        border-top: 3px solid #32f99a; /* The Voltage Green Accent */
        padding: 15px;
        border-radius: 0px; /* Sharp corners like FC menus */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #32f99a !important;
        font-size: 2.5rem !important;
        font-family: 'Rajdhani', sans-serif;
    }

    /* 5. THE "TRIANGLE" BUTTONS */
    div.stButton > button {
        background-color: #32f99a;
        color: #000;
        border: none;
        clip-path: polygon(10% 0%, 100% 0%, 100% 100%, 0% 100%); /* Angled Cut */
        padding: 15px 30px;
        font-size: 20px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #fff;
        transform: scale(1.02);
        box-shadow: 0 0 15px #32f99a;
    }

    /* 6. DATAFRAMES (Leaderboards) */
    div[data-testid="stDataFrame"] {
        background-color: #16161a;
        border: 1px solid #333;
    }
    
    /* Hide Streamlit Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. LOGIC ENGINE (Updated for History)
# ============================================

class EloTracker:
    def __init__(self, k_factor=20):
        self.ratings = {}
        self.k = k_factor
        self.base_rating = 1500

    def get_rating(self, team):
        return self.ratings.get(team, self.base_rating)

    def update_ratings(self, home_team, away_team, result):
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)
        e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
        new_h = r_h + self.k * (result - e_h)
        new_a = r_a + self.k * ((1 - result) - (1 - e_h))
        self.ratings[home_team] = new_h
        self.ratings[away_team] = new_a

class EPLPredictor:
    def __init__(self):
        self.urls = [
            "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
            "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        ]
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        self.le_team = LabelEncoder()
        self.elo = EloTracker(k_factor=20)
        self.matches = None
        self.history = [] # Stores match-by-match prediction accuracy

    def fetch_data(self):
        frames = []
        for url in self.urls:
            try:
                s = requests.get(url).content
                df = pd.read_csv(io.StringIO(s.decode('latin-1')))
                df = df.dropna(how='all')
                frames.append(df)
            except: pass
        
        if not frames: return False

        self.data = pd.concat(frames, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.matches = self.data[self.data['FTR'].notna()].copy()
        return True

    def run_training_cycle(self):
        # We process matches one by one to simulate "live" history
        all_teams = pd.concat([self.matches['HomeTeam'], self.matches['AwayTeam']]).unique()
        self.le_team.fit(all_teams)
        le_res = LabelEncoder()
        self.matches['Result_Code'] = le_res.fit_transform(self.matches['FTR'])
        
        features = []
        
        for idx, row in self.matches.iterrows():
            h_team, a_team = row['HomeTeam'], row['AwayTeam']
            
            # 1. CAPTURE STATE BEFORE MATCH (For Training/Prediction)
            h_elo = self.elo.get_rating(h_team)
            a_elo = self.elo.get_rating(a_team)
            
            h_recent = self._get_recent(h_team, idx)
            a_recent = self._get_recent(a_team, idx)
            
            f_vec = {
                'Elo_Diff': h_elo - a_elo,
                'Home_Elo': h_elo,
                'Away_Elo': a_elo,
                'H_Form_Pts': self._get_pts(h_recent, h_team),
                'A_Form_Pts': self._get_pts(a_recent, a_team),
                'H_GD': self._get_gd(h_recent, h_team),
                'A_GD': self._get_gd(a_recent, a_team)
            }
            features.append(f_vec)
            
            # 2. UPDATE ELO (After Match)
            res_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, res_val)

        self.X = pd.DataFrame(features)
        self.y = self.matches['Result_Code']
        self.model.fit(self.X, self.y)

    def evaluate_recent_performance(self, n_games=20):
        # Simulates how the model WOULD have predicted the last N games
        # Note: In a real prod app, we'd use a proper test set. 
        # Here we just check model confidence vs reality on recent data.
        recent = self.matches.tail(n_games).copy()
        features_subset = self.X.tail(n_games)
        
        preds = self.model.predict(features_subset)
        probs = self.model.predict_proba(features_subset)
        
        results = []
        for i, (idx, row) in enumerate(recent.iterrows()):
            home = row['HomeTeam']
            away = row['AwayTeam']
            actual = row['FTR']
            
            # Probability for Home Win (Index 2 in sklearn default usually, but check classes)
            # Classes are typically A(0), D(1), H(2)
            p_away, p_draw, p_home = probs[i]
            
            predicted_outcome = "H" if p_home > p_away and p_home > p_draw else "A" if p_away > p_home and p_away > p_draw else "D"
            
            is_correct = (predicted_outcome == actual)
            
            results.append({
                "Date": row['Date'].strftime('%Y-%m-%d'),
                "Match": f"{home} vs {away}",
                "Actual": actual,
                "Pred": predicted_outcome,
                "Confidence": max(p_home, p_draw, p_away),
                "Correct": "✅" if is_correct else "❌"
            })
        return pd.DataFrame(results)

    def _get_recent(self, team, idx):
        return self.matches[((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team)) & (self.matches.index < idx)].tail(5)

    def _get_pts(self, matches, team):
        if matches.empty: return 1.0
        pts = 0
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team:
                pts += 3 if m['FTR'] == 'H' else 1 if m['FTR'] == 'D' else 0
            else:
                pts += 3 if m['FTR'] == 'A' else 1 if m['FTR'] == 'D' else 0
        return pts / len(matches)

    def _get_gd(self, matches, team):
        if matches.empty: return 0
        gd = 0
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team: gd += (m['FTHG'] - m['FTAG'])
            else: gd += (m['FTAG'] - m['FTHG'])
        return gd / len(matches)

    def predict_future(self, h_team, a_team):
        if h_team not in self.le_team.classes_ or a_team not in self.le_team.classes_: return None
        h_elo = self.elo.get_rating(h_team)
        a_elo = self.elo.get_rating(a_team)
        
        # We simply take the latest known form from the very last match in DB
        last_idx = self.matches.index[-1] + 1
        h_recent = self._get_recent(h_team, last_idx)
        a_recent = self._get_recent(a_team, last_idx)
        
        vec = pd.DataFrame([{
            'Elo_Diff': h_elo - a_elo,
            'Home_Elo': h_elo,
            'Away_Elo': a_elo,
            'H_Form_Pts': self._get_pts(h_recent, h_team),
            'A_Form_Pts': self._get_pts(a_recent, a_team),
            'H_GD': self._get_gd(h_recent, h_team),
            'A_GD': self._get_gd(a_recent, a_team)
        }])
        
        probs = self.model.predict_proba(vec)[0]
        return {'A': probs[0], 'D': probs[1], 'H': probs[2]}

# ============================================
# 3. INITIALIZATION
# ============================================

@st.cache_resource
def load_app():
    eng = EPLPredictor()
    if eng.fetch_data():
        eng.run_training_cycle()
        return eng
    return None

engine = load_app()

# ============================================
# 4. NAVIGATION & UI
# ============================================

# Sidebar Navigation (The "Menu")
with st.sidebar:
    st.markdown("## ⚽ FC PREDICTOR")
    page = st.radio("NAVIGATION", ["KICK OFF", "LEADERBOARDS", "MATCH REPLAY"], label_visibility="collapsed")
    st.markdown("---")
    st.info("DATA: FOOTBALL-DATA.CO.UK")
    st.write("v25.2.1 • LIVE BUILD")

if not engine:
    st.error("Server Offline. Could not fetch season data.")
    st.stop()

teams = sorted(engine.le_team.classes_)

# --------------------------------------------
# PAGE 1: KICK OFF (PREDICTOR)
# --------------------------------------------
if page == "KICK OFF":
    st.markdown("<h1 style='text-align: center'>MATCHDAY <span style='color:#32f99a'>CENTRE</span></h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. THE SELECTION ARENA
    col1, col2, col3 = st.columns([1, 0.2, 1])
    
    with col1:
        st.markdown("### HOME CLUB")
        h_team = st.selectbox("H", teams, index=0, label_visibility="collapsed", key="h_sel")
        st.markdown(f"<div style='background: #1e1e24; padding: 20px; border-top: 4px solid #32f99a; text-align: center;'><h2>{h_team}</h2></div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<br><br><h1 style='text-align: center; color: #555;'>VS</h1>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("### AWAY CLUB")
        a_team = st.selectbox("A", teams, index=1, label_visibility="collapsed", key="a_sel")
        st.markdown(f"<div style='background: #1e1e24; padding: 20px; border-top: 4px solid #ff0055; text-align: center;'><h2>{a_team}</h2></div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # 2. THE ACTION BUTTON
    c_btn1, c_btn2, c_btn3 = st.columns([1,2,1])
    with c_btn2:
        if st.button("SIMULATE MATCH", use_container_width=True):
            if h_team == a_team:
                st.warning("SELECT DIFFERENT TEAMS")
            else:
                pred = engine.predict_future(h_team, a_team)
                
                # 3. THE REVEAL
                st.markdown("---")
                
                # Custom HTML for the "Winner Card"
                win_prob = max(pred.values())
                winner = "HOME" if pred['H'] == win_prob else "AWAY" if pred['A'] == win_prob else "DRAW"
                color = "#32f99a" if winner == "HOME" else "#ff0055" if winner == "AWAY" else "#888"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                    <div style="background: rgba(255,255,255,0.05); border: 2px solid {color}; padding: 30px; border-radius: 0px; text-align: center; min-width: 300px;">
                        <h4 style="margin:0; color: #aaa;">PROJECTED RESULT</h4>
                        <h1 style="margin:0; font-size: 4rem !important; color: {color};">{winner} WIN</h1>
                        <p style="margin:0; font-size: 1.2rem;">CONFIDENCE: {int(win_prob*100)}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 4. STATS BREAKDOWN
                m1, m2, m3 = st.columns(3)
                m1.metric("HOME", f"{int(pred['H']*100)}%")
                m2.metric("DRAW", f"{int(pred['D']*100)}%")
                m3.metric("AWAY", f"{int(pred['A']*100)}%")

# --------------------------------------------
# PAGE 2: LEADERBOARDS (ELO)
# --------------------------------------------
elif page == "LEADERBOARDS":
    st.title("GLOBAL RANKINGS")
    st.write("LIVE ELO RATINGS TRACKER")
    
    # Convert ratings dict to DataFrame
    elo_data = pd.DataFrame(list(engine.elo.ratings.items()), columns=['Club', 'Rating'])
    elo_data['Rating'] = elo_data['Rating'].astype(int)
    elo_data = elo_data.sort_values('Rating', ascending=False).reset_index(drop=True)
    elo_data.index += 1 # Rank starts at 1
    
    # Display top 3 visually
    top3 = elo_data.head(3)
    c1, c2, c3 = st.columns(3)
    
    with c2:
        st.markdown(f"<div style='text-align:center; padding: 20px; border: 2px solid #ffd700; background: #1a1a1a;'>🥇 1ST<br><h1>{top3.iloc[0]['Club']}</h1><h2>{top3.iloc[0]['Rating']}</h2></div>", unsafe_allow_html=True)
    with c1:
        st.markdown(f"<div style='text-align:center; padding: 20px; border: 2px solid #c0c0c0; background: #1a1a1a; margin-top: 20px;'>🥈 2ND<br><h3>{top3.iloc[1]['Club']}</h3><h3>{top3.iloc[1]['Rating']}</h3></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align:center; padding: 20px; border: 2px solid #cd7f32; background: #1a1a1a; margin-top: 20px;'>🥉 3RD<br><h3>{top3.iloc[2]['Club']}</h3><h3>{top3.iloc[2]['Rating']}</h3></div>", unsafe_allow_html=True)

    st.write("")
    st.write("### FULL STANDINGS")
    
    # Styled Table
    st.dataframe(
        elo_data, 
        use_container_width=True,
        column_config={
            "Rating": st.column_config.ProgressColumn(
                "Skill Rating",
                format="%d",
                min_value=1300,
                max_value=2200,
            ),
        }
    )

# --------------------------------------------
# PAGE 3: MATCH REPLAY (HISTORY)
# --------------------------------------------
elif page == "MATCH REPLAY":
    st.title("PERFORMANCE REVIEW")
    st.write("MODEL PREDICTIONS VS REALITY (LAST 20 GAMES)")
    
    # Run the evaluation
    with st.spinner("ANALYZING MATCH FOOTAGE..."):
        history_df = engine.evaluate_recent_performance(n_games=20)
    
    # Metrics
    acc = len(history_df[history_df['Correct'] == '✅']) / len(history_df)
    
    k1, k2 = st.columns(2)
    k1.metric("RECENT ACCURACY", f"{int(acc*100)}%")
    k2.metric("GAMES ANALYZED", "20")
    
    st.table(history_df[['Date', 'Match', 'Actual', 'Pred', 'Correct']])

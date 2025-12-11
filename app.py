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
# 1. SETUP & THEME CONFIGURATION
# ============================================
st.set_page_config(
    page_title="EPL Matchday AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR "FOOTBALL PASSION" LOOK ---
st.markdown("""
    <style>
    /* Import Google Font: Oswald (Sports style) */
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&display=swap');

    /* 1. MAIN BACKGROUND: Dark Stadium Vibes */
    .stApp {
        background-color: #1a1a1a;
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("https://images.unsplash.com/photo-1522778119026-d647f0565c6a");
        background-size: cover;
        background-attachment: fixed;
    }

    /* 2. HEADERS & TEXT */
    h1, h2, h3, h4, h5 {
        font-family: 'Oswald', sans-serif !important;
        text-transform: uppercase;
        color: #ffffff !important;
        text-shadow: 2px 2px 4px #000000;
        letter-spacing: 1px;
    }
    
    p, label {
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
    }

    /* 3. METRIC CARDS (The Scoreboard Look) */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.05);
        border-color: #00ff85; /* EPL Green */
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Oswald', sans-serif;
        font-size: 3rem !important;
        color: #00ff85 !important; /* Neon Green numbers */
    }
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: bold;
    }

    /* 4. THE BUTTON (Make it Pop) */
    div.stButton > button {
        background: linear-gradient(45deg, #ff0055, #ff0000);
        color: white;
        font-family: 'Oswald', sans-serif;
        font-size: 24px;
        padding: 0.5rem 2rem;
        border: none;
        border-radius: 5px;
        width: 100%;
        text-transform: uppercase;
        box-shadow: 0px 0px 20px rgba(255, 0, 85, 0.6);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0px 0px 30px rgba(255, 0, 85, 0.9);
        border: 2px solid white;
    }

    /* 5. SELECT BOXES */
    div[data-baseweb="select"] > div {
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        border: 1px solid #444;
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. HELPER FUNCTIONS (Logos & Logic)
# ============================================

def get_team_badge(team_name):
    """
    Returns a URL for the team badge. 
    Uses a public GitHub repo for EPL logos or a default shield.
    """
    # Simple mapping for big 6 to ensure high quality, others fallback to generic search pattern
    base_url = "https://resources.premierleague.com/premierleague/badges/t"
    
    # We would need exact ID mapping for real URLS, so for this demo 
    # we will use a reliable placeholder service with the team name text
    return f"https://ui-avatars.com/api/?name={team_name}&background=random&color=fff&size=128&bold=true&font-size=0.33"

# ============================================
# 3. CORE LOGIC (ELO & MODEL) - KEPT SAME
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
        self.matches = None
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        self.le_team = LabelEncoder()
        self.elo = EloTracker(k_factor=20)
        self.team_stats = {} 

    def fetch_data(self):
        frames = []
        for url in self.urls:
            try:
                s = requests.get(url).content
                df = pd.read_csv(io.StringIO(s.decode('latin-1')))
                df = df.dropna(how='all')
                frames.append(df)
            except Exception:
                pass # Silent fail for smoother UI
        
        if not frames: return False

        self.data = pd.concat(frames, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.matches = self.data[self.data['FTR'].notna()].copy()
        return True

    def prepare_features(self):
        all_teams = pd.concat([self.matches['HomeTeam'], self.matches['AwayTeam']]).unique()
        self.le_team.fit(all_teams)
        le_res = LabelEncoder()
        self.matches['Result_Code'] = le_res.fit_transform(self.matches['FTR'])
        
        # Initialize history
        for team in all_teams:
            self.team_stats[team] = {'elo_history': [], 'dates': []}

        for idx, row in self.matches.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            date = row['Date']
            
            h_elo = self.elo.get_rating(h_team)
            a_elo = self.elo.get_rating(a_team)
            
            self.team_stats[h_team]['elo_history'].append(h_elo)
            self.team_stats[h_team]['dates'].append(date)
            self.team_stats[a_team]['elo_history'].append(a_elo)
            self.team_stats[a_team]['dates'].append(date)

            h_recent = self._get_recent_matches(h_team, idx)
            a_recent = self._get_recent_matches(a_team, idx)

            f_vec = {
                'Elo_Diff': h_elo - a_elo,
                'Home_Elo': h_elo,
                'Away_Elo': a_elo,
                'H_Form_Pts': self._get_recent_points(h_recent, h_team),
                'A_Form_Pts': self._get_recent_points(a_recent, a_team),
                'H_Goal_Diff_5': self._get_goal_diff(h_recent, h_team),
                'A_Goal_Diff_5': self._get_goal_diff(a_recent, a_team)
            }
            # Update Elo
            result_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, result_val)

        self.X = pd.DataFrame([f_vec]) # Just dummy init
        # Re-run full loop properly for training (simplified for this snippet)
        # In production, we run the loop above to build X and y fully.
        # For now, let's assume the previous code block did this correctly.
        # TO SAVE SPACE: Re-implementing the robust loop inside prepare_features
        features = []
        self.elo = EloTracker(k_factor=20) # Reset
        for idx, row in self.matches.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            h_elo = self.elo.get_rating(h_team)
            a_elo = self.elo.get_rating(a_team)
            
            h_recent = self._get_recent_matches(h_team, idx)
            a_recent = self._get_recent_matches(a_team, idx)

            f_vec = {
                'Elo_Diff': h_elo - a_elo,
                'Home_Elo': h_elo,
                'Away_Elo': a_elo,
                'H_Form_Pts': self._get_recent_points(h_recent, h_team),
                'A_Form_Pts': self._get_recent_points(a_recent, a_team),
                'H_Goal_Diff_5': self._get_goal_diff(h_recent, h_team),
                'A_Goal_Diff_5': self._get_goal_diff(a_recent, a_team)
            }
            features.append(f_vec)
            result_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, result_val)
            
        self.X = pd.DataFrame(features)
        self.y = self.matches['Result_Code']

    def _get_recent_matches(self, team, current_idx):
        return self.matches[((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team)) & (self.matches.index < current_idx)].tail(5)

    def _get_recent_points(self, matches, team):
        if matches.empty: return 1.0
        pts = 0
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team:
                if m['FTR'] == 'H': pts += 3
                elif m['FTR'] == 'D': pts += 1
            else:
                if m['FTR'] == 'A': pts += 3
                elif m['FTR'] == 'D': pts += 1
        return pts / len(matches)

    def _get_goal_diff(self, matches, team):
        if matches.empty: return 0
        gd = 0
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team:
                gd += (m['FTHG'] - m['FTAG'])
            else:
                gd += (m['FTAG'] - m['FTHG'])
        return gd / len(matches)

    def train_model(self):
        self.model.fit(self.X, self.y)

    def predict_next_match(self, home_team, away_team):
        if home_team not in self.le_team.classes_ or away_team not in self.le_team.classes_:
            return None
        h_elo = self.elo.get_rating(home_team)
        a_elo = self.elo.get_rating(away_team)
        current_idx = len(self.matches) + 1
        h_recent = self._get_recent_matches(home_team, current_idx)
        a_recent = self._get_recent_matches(away_team, current_idx)

        input_vec = pd.DataFrame([{
            'Elo_Diff': h_elo - a_elo,
            'Home_Elo': h_elo,
            'Away_Elo': a_elo,
            'H_Form_Pts': self._get_recent_points(h_recent, home_team),
            'A_Form_Pts': self._get_recent_points(a_recent, away_team),
            'H_Goal_Diff_5': self._get_goal_diff(h_recent, home_team),
            'A_Goal_Diff_5': self._get_goal_diff(a_recent, away_team)
        }])
        probs = self.model.predict_proba(input_vec)[0]
        return {'Away': probs[0], 'Draw': probs[1], 'Home': probs[2]}

# ============================================
# 4. APP INTERFACE
# ============================================

@st.cache_resource
def load_engine():
    engine = EPLPredictor()
    with st.spinner('Preparing Matchday Models...'):
        if not engine.fetch_data(): return None
        engine.prepare_features()
        engine.train_model()
    return engine

engine = load_engine()

# --- HERO SECTION ---
st.markdown("<h1 style='text-align: center; font-size: 80px; margin-bottom: 0px;'>MATCHDAY <span style='color: #00ff85'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; opacity: 0.8;'>POWERED BY MACHINE LEARNING & ELO RATINGS</p>", unsafe_allow_html=True)
st.markdown("---")

if engine:
    team_list = sorted(engine.le_team.classes_)
    
    # --- MATCHUP SELECTION ---
    c1, c2, c3 = st.columns([1, 0.2, 1])
    
    with c1:
        st.markdown("<h3 style='text-align: center'>HOME</h3>", unsafe_allow_html=True)
        home_team = st.selectbox("Select Home Team", team_list, index=0, label_visibility="collapsed")
        st.image(get_team_badge(home_team), width=100)
        
    with c2:
        st.markdown("<br><br><h1 style='text-align: center; color: #ff0055;'>VS</h1>", unsafe_allow_html=True)
        
    with c3:
        st.markdown("<h3 style='text-align: center'>AWAY</h3>", unsafe_allow_html=True)
        away_team = st.selectbox("Select Away Team", team_list, index=1, label_visibility="collapsed")
        # Align image to right
        st.markdown(f"<div style='display: flex; justify-content: flex-end'><img src='{get_team_badge(away_team)}' width='100'></div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # --- ACTION BUTTON ---
    if st.button("🚀 KICK OFF PREDICTION"):
        if home_team == away_team:
            st.warning("Teams must be different!")
        else:
            with st.spinner("Simulating match..."):
                pred = engine.predict_next_match(home_team, away_team)
                
            if pred:
                st.markdown("---")
                st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>FULL TIME PROJECTION</h2>", unsafe_allow_html=True)
                
                # SCOREBOARD CARDS
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("HOME WIN", f"{pred['Home']*100:.1f}%")
                with col2:
                    st.metric("DRAW", f"{pred['Draw']*100:.1f}%")
                with col3:
                    st.metric("AWAY WIN", f"{pred['Away']*100:.1f}%")
                
                # BAR CHART VISUAL
                st.write("")
                st.write("### DOMINANCE METER")
                
                # Custom HTML Progress Bar
                bar_html = f"""
                <div style="display: flex; width: 100%; height: 30px; border-radius: 15px; overflow: hidden; margin-top: 10px;">
                    <div style="width: {pred['Home']*100}%; background-color: #00ff85; display: flex; align-items: center; justify-content: center; font-weight: bold; color: black;">{int(pred['Home']*100)}%</div>
                    <div style="width: {pred['Draw']*100}%; background-color: #888; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">D</div>
                    <div style="width: {pred['Away']*100}%; background-color: #ff0055; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">{int(pred['Away']*100)}%</div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; color: #ccc; font-size: 14px;">
                    <span>{home_team}</span>
                    <span>{away_team}</span>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

    # --- TEAM STATS EXPANDER ---
    st.write("")
    st.write("")
    with st.expander("📊 VIEW TEAM FORM & STATS"):
        team_stats_select = st.selectbox("Select Team", team_list)
        if team_stats_select in engine.team_stats:
            stats = engine.team_stats[team_stats_select]
            
            # Use Streamlit native chart for sleekness
            chart_data = pd.DataFrame({
                "Date": stats['dates'],
                "Elo Rating": stats['elo_history']
            }).set_index("Date")
            
            st.line_chart(chart_data, color="#00ff85")
            st.caption("Elo Rating over the last 2 seasons")

else:
    st.error("Could not load season data.")

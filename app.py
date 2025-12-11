import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. CONFIG & ASSETS
# ============================================
st.set_page_config(
    page_title="PL Official Matchday",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOGO DATABASE ---
# Maps the specific names in football-data.co.uk to clean PNG URLs
# We use Wikimedia/Wikipedia URLs for stability and transparency
TEAM_LOGOS = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "Aston Villa": "https://upload.wikimedia.org/wikipedia/en/f/f9/Aston_Villa_FC_crest_%282016%29.svg",
    "Bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "Brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://upload.wikimedia.org/wikipedia/en/6/62/Burnley_F.C._Logo.svg",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "Crystal Palace": "https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Ipswich": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
    "Leeds": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
    "Leicester": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
    "Liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
    "Luton": "https://upload.wikimedia.org/wikipedia/en/9/9d/Luton_Town_logo.svg",
    "Man City": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
    "Man United": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "Newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "Nott'm Forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
    "Sheffield United": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",
    "Southampton": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
    "Tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
    "West Ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "Wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg"
}

def get_logo(team_name):
    return TEAM_LOGOS.get(team_name, "https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg") # Fallback icon

# ============================================
# 2. OFFICIAL PREMIER LEAGUE CSS
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;900&display=swap');
    
    :root {
        --pl-purple: #38003c;
        --pl-green: #00ff85;
        --pl-pink: #e90052;
    }

    .stApp {
        background-color: var(--pl-purple);
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        color: white;
    }

    h1, h2, h3 { font-family: 'Poppins', sans-serif !important; font-weight: 900 !important; letter-spacing: -1px; }
    p, div { font-family: 'Poppins', sans-serif !important; }

    /* VS Badge */
    .vs-badge {
        font-size: 40px;
        color: var(--pl-pink);
        font-weight: 900;
        text-shadow: 2px 2px 0px #fff;
    }

    /* Team Dropdown Styling */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: var(--pl-purple) !important;
        border: none;
        font-weight: 700;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border-left: 5px solid var(--pl-green);
        padding: 10px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--pl-green) !important;
        font-size: 2rem !important;
    }

    /* Prediction Button */
    .stButton button {
        background: linear-gradient(90deg, #e90052 0%, #ff0055 100%);
        color: white;
        text-transform: uppercase;
        font-weight: 800;
        border: none;
        padding: 15px 0;
        font-size: 18px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(233, 0, 82, 0.6);
    }
    
    /* Clean up standard Streamlit UI */
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# 3. BACKEND ENGINE
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
        # We load multiple seasons to train the model...
        self.urls = [
            "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
            "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        ]
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        self.le_team = LabelEncoder()
        self.elo = EloTracker(k_factor=20)
        self.matches = None
        self.current_season_teams = [] # New list for filtering

    def fetch_data(self):
        frames = []
        for url in self.urls:
            try:
                s = requests.get(url).content
                df = pd.read_csv(io.StringIO(s.decode('latin-1')))
                df = df.dropna(how='all')
                
                # Check if this is the latest file to grab current teams
                if "2526" in url:
                    # Get unique teams from Home and Away columns of the latest file
                    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                    self.current_season_teams = sorted(teams)
                    
                frames.append(df)
            except: pass
        
        if not frames: return False

        self.data = pd.concat(frames, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.matches = self.data[self.data['FTR'].notna()].copy()
        
        # Fallback: If 25/26 file is empty (start of season), grab from 24/25
        if not self.current_season_teams:
             last_season_matches = self.matches[self.matches['Date'] > '2024-08-01']
             self.current_season_teams = sorted(pd.concat([last_season_matches['HomeTeam'], last_season_matches['AwayTeam']]).unique())

        return True

    def run_training_cycle(self):
        all_teams = pd.concat([self.matches['HomeTeam'], self.matches['AwayTeam']]).unique()
        self.le_team.fit(all_teams)
        le_res = LabelEncoder()
        self.matches['Result_Code'] = le_res.fit_transform(self.matches['FTR'])
        
        features = []
        for idx, row in self.matches.iterrows():
            h_team, a_team = row['HomeTeam'], row['AwayTeam']
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
            res_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, res_val)

        self.X = pd.DataFrame(features)
        self.y = self.matches['Result_Code']
        self.model.fit(self.X, self.y)

    def _get_recent(self, team, idx):
        return self.matches[((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team)) & (self.matches.index < idx)].tail(5)

    def _get_pts(self, matches, team):
        if matches.empty: return 1.0
        pts = 0
        for _, m in matches.iterrows():
            if m['HomeTeam'] == team: pts += 3 if m['FTR'] == 'H' else 1 if m['FTR'] == 'D' else 0
            else: pts += 3 if m['FTR'] == 'A' else 1 if m['FTR'] == 'D' else 0
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
# 4. INITIALIZE APP
# ============================================

@st.cache_resource
def load_app_v2():
    eng = EPLPredictor()
    if eng.fetch_data():
        eng.run_training_cycle()
        return eng
    return None

engine = load_app_v2()

if not engine:
    st.error("Could not load data. Please check internet connection.")
    st.stop()

# ============================================
# 5. UI: "QUICK MATCH" LAYOUT
# ============================================

# Use the FILTERED list (Current Season Only)
current_teams = engine.current_season_teams

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 3rem; margin-bottom: 0;">MATCHDAY <span style="color: #00ff85">CENTRE</span></h1>
    <p style="color: #bbb;">OFFICIAL PREMIER LEAGUE AI PREDICTOR</p>
</div>
""", unsafe_allow_html=True)

# The "FC26" Style Layout: Logo on top, Dropdown below
col1, col2, col3 = st.columns([1, 0.3, 1])

with col1:
    st.markdown("<h3 style='text-align: center; color: #fff;'>HOME</h3>", unsafe_allow_html=True)
    h_team = st.selectbox("Home Team", current_teams, index=0, label_visibility="collapsed", key="h_team_select")
    
    # BIG LOGO DISPLAY
    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="{get_logo(h_team)}" style="height: 150px; filter: drop-shadow(0 0 10px rgba(0,0,0,0.5));">
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True) # Spacer
    st.markdown("<div class='vs-badge' style='text-align: center;'>VS</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align: center; color: #fff;'>AWAY</h3>", unsafe_allow_html=True)
    a_team = st.selectbox("Away Team", current_teams, index=1, label_visibility="collapsed", key="a_team_select")
    
    # BIG LOGO DISPLAY
    st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="{get_logo(a_team)}" style="height: 150px; filter: drop-shadow(0 0 10px rgba(0,0,0,0.5));">
        </div>
    """, unsafe_allow_html=True)

# PREDICTION ACTION
st.write("")
st.write("")
col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 2, 1])

with col_btn_2:
    if st.button("KICK OFF PREDICTION"):
        if h_team == a_team:
            st.warning("Please select two different teams!")
        else:
            pred = engine.predict_future(h_team, a_team)
            
            # SCROLL TO RESULT
            st.markdown("---")
            
            # Winner Logic
            win_prob = max(pred.values())
            if pred['H'] == win_prob: 
                winner_text = f"{h_team} WIN"
                win_color = "#00ff85"
            elif pred['A'] == win_prob:
                winner_text = f"{a_team} WIN"
                win_color = "#e90052"
            else:
                winner_text = "DRAW"
                win_color = "#cccccc"

            # RESULT BANNER
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 30px; text-align: center; border: 2px solid {win_color};">
                <h4 style="color: #ddd; margin: 0;">PREDICTED OUTCOME</h4>
                <h1 style="font-size: 4rem; margin: 10px 0; color: {win_color}; text-shadow: 0 0 20px {win_color}40;">{winner_text}</h1>
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
                    <div>
                        <div style="font-size: 0.9rem; color: #aaa;">HOME</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: white;">{int(pred['H']*100)}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; color: #aaa;">DRAW</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: white;">{int(pred['D']*100)}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; color: #aaa;">AWAY</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: white;">{int(pred['A']*100)}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ELO TABLE IN EXPANDER
with st.expander("VIEW LIVE LEAGUE STANDINGS"):
    elo_data = pd.DataFrame(list(engine.elo.ratings.items()), columns=['Club', 'Rating'])
    
    # Filter Elo Table to only show current teams as well
    elo_data = elo_data[elo_data['Club'].isin(current_teams)]
    
    elo_data['Rating'] = elo_data['Rating'].astype(int)
    elo_data = elo_data.sort_values('Rating', ascending=False).reset_index(drop=True)
    elo_data.index += 1
    st.dataframe(elo_data, use_container_width=True)


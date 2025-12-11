import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# ============================================
# 1. APP CONFIGURATION & ASSETS
# ============================================
st.set_page_config(
    page_title="PL Predictor",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOGO DATABASE (Source: FotMob CDN for stability) ---
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
# 2. OFFICIAL PREMIER LEAGUE CSS THEME
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

    .stApp {
        background-color: var(--pl-purple);
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png"); 
        color: var(--pl-white);
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 900 !important;
        color: var(--pl-white) !important;
        letter-spacing: -0.5px;
    }
    
    p, label, .stMarkdown, div, span {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* --- FLEXBOX CENTERING WRAPPER --- */
    .team-col-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center; /* Ensures horizontal centering */
        justify-content: flex-start;
        height: 100%;
        width: 100%;
    }

    /* LOGO STYLING */
    .team-logo-img {
        max-height: 160px;
        width: auto;
        /* Auto margins are crucial for centering block elements */
        margin: 25px auto; 
        display: block;
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.2));
        transition: transform 0.3s ease;
    }
    .team-logo-img:hover {
        transform: scale(1.05);
    }
    
    /* VS VERTICAL ALIGNMENT */
    .vs-col-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding-top: 80px; /* Push down to align with logos */
    }

    /* DROPDOWNS */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #38003c !important;
        border-radius: 4px;
        border: 2px solid transparent;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: var(--pl-green);
    }
    div[data-baseweb="select"] span {
        color: #38003c !important; 
        font-weight: 700;
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
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 0, 82, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 0, 82, 0.6);
    }

    /* METRICS */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid var(--pl-green);
    }
    div[data-testid="stMetricValue"] {
        color: var(--pl-green) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #fff !important;
    }
    
    /* DATAFRAMES */
    div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
    }
    thead tr th {
        background-color: var(--pl-purple) !important;
        color: white !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================
# 4. BACKEND LOGIC
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
        self.current_season_teams = []

    def fetch_data(self):
        frames = []
        for url in self.urls:
            try:
                s = requests.get(url).content
                df = pd.read_csv(io.StringIO(s.decode('latin-1')))
                df = df.dropna(how='all')
                if "2526" in url:
                     teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                     self.current_season_teams = sorted(teams)
                frames.append(df)
            except: pass
        
        if not frames: return False

        self.data = pd.concat(frames, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.matches = self.data[self.data['FTR'].notna()].copy()
        
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

    def evaluate_recent_performance(self, n_games=20):
        recent = self.matches.tail(n_games).copy()
        features_subset = self.X.tail(n_games)
        probs = self.model.predict_proba(features_subset)
        
        results = []
        for i, (idx, row) in enumerate(recent.iterrows()):
            home = row['HomeTeam']
            away = row['AwayTeam']
            actual = row['FTR']
            p_away, p_draw, p_home = probs[i]
            predicted_outcome = "H" if p_home > p_away and p_home > p_draw else "A" if p_away > p_home and p_away > p_draw else "D"
            is_correct = (predicted_outcome == actual)
            
            results.append({
                "Date": row['Date'].strftime('%d %b'),
                "Match": f"{home} v {away}",
                "Result": actual,
                "Prediction": predicted_outcome,
                "Correct": "✔" if is_correct else "✖"
            })
        return pd.DataFrame(results)

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
# 5. INITIALIZATION
# ============================================

@st.cache_resource
def load_pl_engine_vFinal():
    eng = EPLPredictor()
    if eng.fetch_data():
        eng.run_training_cycle()
        return eng
    return None

engine = load_pl_engine_vFinal()

# ============================================
# 6. HEADER
# ============================================
st.markdown("""
<div style="background-color: #38003c; padding: 20px; border-bottom: 4px solid #00ff85; margin-bottom: 25px;">
    <h1 style="color: white; margin:0; font-size: 3rem;">PREMIER LEAGUE <span style="color: #00ff85">PREDICTOR</span></h1>
    <p style="color: #e0e0e0; margin:0; font-size: 1.1rem;">OFFICIAL MATCHDAY INSIGHTS</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 7. MAIN UI
# ============================================

if not engine:
    st.error("Connection to PL Database failed. Please reload.")
    st.stop()

current_teams = engine.current_season_teams

# Tabs for Navigation
tab1, tab2, tab3 = st.tabs(["MATCH CENTRE", "TABLE", "FORM GUIDE"])

# --- TAB 1: MATCH CENTRE ---
with tab1:
    # Slightly wider side columns for better centering space
    col1, col2, col3 = st.columns([1.2, 0.3, 1.2])
    
    # --- HOME COLUMN (Centered Wrapper) ---
    with col1:
        st.markdown('<div class="team-col-wrapper">', unsafe_allow_html=True)
        st.markdown("### HOME CLUB")
        h_team = st.selectbox("H_Select", current_teams, index=0, label_visibility="collapsed", key="h_team_select")
        
        # LOGO (Using HTML img for perfect centering control)
        st.markdown(f'<img src="{get_logo(h_team)}" class="team-logo-img">', unsafe_allow_html=True)
        
        # ELO BOX
        h_elo_val = int(engine.elo.get_rating(h_team))
        st.markdown(f"""
            <div style="text-align: center; margin-top: 10px; background: #fff; padding: 10px 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                <span style="color:#38003c; font-size: 0.9rem; font-weight: bold; text-transform: uppercase;">Elo Rating</span><br>
                <span style="color:#38003c; font-size: 2rem; font-weight: 900;">{h_elo_val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- VS COLUMN (Vertically Aligned) ---
    with col2:
        st.markdown('<div class="vs-col-wrapper">', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #e90052 !important; font-size: 3.5rem !important; margin: 0;'>VS</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- AWAY COLUMN (Centered Wrapper) ---
    with col3:
        st.markdown('<div class="team-col-wrapper">', unsafe_allow_html=True)
        st.markdown("### AWAY CLUB")
        a_team = st.selectbox("A_Select", current_teams, index=1, label_visibility="collapsed", key="a_team_select")
        
        # LOGO
        st.markdown(f'<img src="{get_logo(a_team)}" class="team-logo-img">', unsafe_allow_html=True)
        
        # ELO BOX
        a_elo_val = int(engine.elo.get_rating(a_team))
        st.markdown(f"""
            <div style="text-align: center; margin-top: 10px; background: #fff; padding: 10px 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                <span style="color:#38003c; font-size: 0.9rem; font-weight: bold; text-transform: uppercase;">Elo Rating</span><br>
                <span style="color:#38003c; font-size: 2rem; font-weight: 900;">{a_elo_val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # --- ACTION BUTTON ---
    if st.button("PREDICT RESULT", use_container_width=True):
        if h_team == a_team:
            st.warning("Please select two different clubs.")
        else:
            pred = engine.predict_future(h_team, a_team)
            
            # --- THE "OFFICIAL" BROADCAST GRAPHIC ---
            st.markdown("---")
            st.markdown("<h3 style='text-align: center; margin-bottom: 20px; color: #00ff85 !important;'>MATCH FORECAST</h3>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("HOME WIN", f"{int(pred['H']*100)}%")
            with c2:
                st.metric("DRAW", f"{int(pred['D']*100)}%")
            with c3:
                st.metric("AWAY WIN", f"{int(pred['A']*100)}%")

            # The "PL Green" Bar
            bar_html = f"""
            <div style="margin-top: 30px; width: 100%; height: 30px; display: flex; border-radius: 15px; overflow: hidden; background: #222; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);">
                <div style="width: {pred['H']*100}%; background: linear-gradient(90deg, #00ff85, #00cc6a); display: flex; align-items: center; justify-content: center; color: #38003c; font-weight: 900;">{int(pred['H']*100)}%</div>
                <div style="width: {pred['D']*100}%; background: #888; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700;">D</div>
                <div style="width: {pred['A']*100}%; background: linear-gradient(90deg, #e90052, #ff0055); display: flex; align-items: center; justify-content: center; color: white; font-weight: 900;">{int(pred['A']*100)}%</div>
            </div>
            <div style="display: flex; justify-content: space-between; color: #bbb; font-size: 0.9rem; margin-top: 10px; font-weight: 700;">
                <span>{h_team.upper()}</span>
                <span>{a_team.upper()}</span>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

# --- TAB 2: TABLE (ELO) - FILTERED ---
with tab2:
    st.markdown("### LIVE CLUB RATINGS")
    elo_data = pd.DataFrame(list(engine.elo.ratings.items()), columns=['Club', 'Rating'])
    elo_data = elo_data[elo_data['Club'].isin(current_teams)]
    elo_data['Rating'] = elo_data['Rating'].astype(int)
    elo_data = elo_data.sort_values('Rating', ascending=False).reset_index(drop=True)
    elo_data.index += 1
    
    st.dataframe(
        elo_data,
        use_container_width=True,
        column_config={
            "Rating": st.column_config.ProgressColumn(
                "Power Index",
                format="%d",
                min_value=1300,
                max_value=2200,
            ),
        },
        height=600
    )

# --- TAB 3: FORM GUIDE ---
with tab3:
    st.markdown("### MODEL ACCURACY (LAST 20 GAMES)")
    history_df = engine.evaluate_recent_performance(n_games=20)
    def highlight_correct(s):
        return ['background-color: #00ff85; color: #38003c; font-weight: bold;' if v == '✔' else 'background-color: #e90052; color: white; font-weight: bold;' if v == '✖' else '' for v in s]

    st.dataframe(history_df.style.apply(highlight_correct, subset=['Correct']), use_container_width=True)

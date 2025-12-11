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
    page_title="FC26 PREDICTOR",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOGO DATABASE (High-Res PNGs) ---
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
    return TEAM_LOGOS.get(team_name, "https://upload.wikimedia.org/wikipedia/commons/d/d3/Soccerball.svg")

# ============================================
# 2. FC26 THEME CSS (Enhanced & Aligned)
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700;800&display=swap');

    :root {
        --neon-green: #32f99a;
        --neon-pink: #ff0055;
        --dark-bg: #0e0e10;
        --card-bg: #1a1a1d;
        --border-color: #333;
    }

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
    }

    .stApp {
        background-color: var(--dark-bg);
        background-image: radial-gradient(circle at 50% 0%, #1a2c38 0%, var(--dark-bg) 80%);
        color: white;
    }

    h1, h2, h3, h4 {
        text-transform: uppercase;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }
    
    h1 { text-shadow: 0px 0px 15px rgba(50, 249, 154, 0.3); }

    /* --- CUSTOM DROPDOWN STYLING --- */
    /* The main container of the selectbox */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #1a1a1d !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 0px !important; /* Sharp corners */
        transition: all 0.3s ease;
    }
    /* Hover effect */
    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--neon-green) !important;
        box-shadow: 0 0 10px rgba(50, 249, 154, 0.2);
    }
    /* The text inside */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span {
        color: white !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    /* The arrow icon */
    div[data-testid="stSelectbox"] svg {
        fill: var(--neon-green) !important;
    }
    /* The dropdown menu options */
    div[data-baseweb="menu"] {
        background-color: #1a1a1d !important;
        border: 2px solid var(--neon-green) !important;
    }
    div[data-baseweb="option"] {
        color: white !important;
    }
    div[data-baseweb="option"]:hover, div[data-baseweb="option"][aria-selected="true"] {
        background-color: rgba(50, 249, 154, 0.2) !important;
    }

    /* --- LAYOUT HELPERS --- */
    .centered-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        height: 100%;
    }
    .vs-col {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding-top: 40px; /* Adjustment for visual centering */
    }

    /* --- OTHER UI ELEMENTS --- */
    [data-testid="stSidebar"] {
        background-color: #08080a;
        border-right: 1px solid #222;
    }
    
    div.stButton > button {
        background-color: var(--neon-green);
        color: #000;
        border: none;
        clip-path: polygon(5% 0%, 100% 0%, 100% 90%, 95% 100%, 0% 100%, 0% 10%);
        padding: 15px 30px;
        font-size: 20px;
        font-weight: 800;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: white;
        box-shadow: 0 0 20px var(--neon-green);
    }

    #MainMenu, footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 3. BACKEND LOGIC (Unchanged)
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
                "Date": row['Date'].strftime('%d/%m'),
                "Match": f"{home} v {away}",
                "Real": actual,
                "AI Pred": predicted_outcome,
                "Status": "✅" if is_correct else "❌"
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
# 4. INITIALIZATION
# ============================================

@st.cache_resource
def load_fc_engine_v4():
    eng = EPLPredictor()
    if eng.fetch_data():
        eng.run_training_cycle()
        return eng
    return None

engine = load_fc_engine_v4()

# ============================================
# 5. MAIN NAVIGATION & UI
# ============================================

# --- SIDEBAR MENU ---
with st.sidebar:
    st.markdown("## ⚽ FC PREDICTOR")
    page = st.radio("GAME MODE", ["KICK OFF", "LEADERBOARDS", "MATCH REPLAY"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("ENGINE: LOGISTIC REGRESSION + ELO")

if not engine:
    st.error("CRITICAL ERROR: Data Engine Offline.")
    st.stop()

current_teams = engine.current_season_teams

# ==================================================
# PAGE 1: KICK OFF (Aligned & Styled)
# ==================================================
if page == "KICK OFF":
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>MATCHDAY <span style='color:#32f99a'>CENTRE</span></h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. THE SELECTION ARENA
    # Use columns with specific weighting for better spacing
    col_h, col_vs, col_a = st.columns([1, 0.3, 1])
    
    # Helper to keep track of current selection for instant logo update
    if 'h_team_curr' not in st.session_state: st.session_state.h_team_curr = current_teams[0]
    if 'a_team_curr' not in st.session_state: st.session_state.a_team_curr = current_teams[1]

    # --- HOME COLUMN ---
    with col_h:
        st.markdown('<div class="centered-col">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #888; margin-bottom: 15px;'>HOME CLUB</h3>", unsafe_allow_html=True)
        
        # Big Logo
        st.markdown(f"""
            <img src="{get_logo(st.session_state.h_team_curr)}" id="home_logo" style="height: 160px; filter: drop-shadow(0 0 20px rgba(50,249,154,0.3)); margin-bottom: 25px; transition: all 0.3s;">
        """, unsafe_allow_html=True)
        
        # Styled Selectbox
        h_team = st.selectbox("H", current_teams, index=current_teams.index(st.session_state.h_team_curr), label_visibility="collapsed", key="h_sel")
        st.session_state.h_team_curr = h_team # Update state

        # ELO Display
        h_elo_val = int(engine.elo.get_rating(h_team))
        st.markdown(f"""
            <div style="margin-top: 20px; text-align: center;">
                <span style="color:#888; font-size: 1rem; font-weight: 700;">ELO RATING</span><br>
                <span style="font-size: 2.5rem; font-weight: 800; color: #32f99a;">{h_elo_val}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # JS to update logo instantly on selection change
        st.markdown(f"""<script>
            var logo = document.getElementById("home_logo");
            logo.style.opacity = 0;
            setTimeout(function(){{
                logo.src = "{get_logo(h_team)}";
                logo.style.opacity = 1;
            }}, 150);
        </script>""", unsafe_allow_html=True)
        
    # --- VS COLUMN ---
    with col_vs:
        st.markdown('<div class="vs-col">', unsafe_allow_html=True)
        st.markdown("<h1 style='color: #444; font-size: 6rem !important; text-shadow: none; margin: 0;'>VS</h1>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- AWAY COLUMN ---
    with col_a:
        st.markdown('<div class="centered-col">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #888; margin-bottom: 15px;'>AWAY CLUB</h3>", unsafe_allow_html=True)
        
        # Big Logo
        st.markdown(f"""
            <img src="{get_logo(st.session_state.a_team_curr)}" id="away_logo" style="height: 160px; filter: drop-shadow(0 0 20px rgba(255,0,85,0.3)); margin-bottom: 25px; transition: all 0.3s;">
        """, unsafe_allow_html=True)
        
        # Styled Selectbox
        a_team = st.selectbox("A", current_teams, index=current_teams.index(st.session_state.a_team_curr), label_visibility="collapsed", key="a_sel")
        st.session_state.a_team_curr = a_team # Update state
        
        # ELO Display
        a_elo_val = int(engine.elo.get_rating(a_team))
        st.markdown(f"""
            <div style="margin-top: 20px; text-align: center;">
                <span style="color:#888; font-size: 1rem; font-weight: 700;">ELO RATING</span><br>
                <span style="font-size: 2.5rem; font-weight: 800; color: #ff0055;">{a_elo_val}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # JS to update logo instantly
        st.markdown(f"""<script>
            var logo = document.getElementById("away_logo");
            logo.style.opacity = 0;
            setTimeout(function(){{
                logo.src = "{get_logo(a_team)}";
                logo.style.opacity = 1;
            }}, 150);
        </script>""", unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # 2. THE ACTION BUTTON
    c_btn1, c_btn2, c_btn3 = st.columns([1,2,1])
    with c_btn2:
        if st.button("SIMULATE MATCH", use_container_width=True):
            if h_team == a_team:
                st.warning("CHOOSE DIFFERENT CLUBS")
            else:
                with st.spinner("RUNNING SIMULATION..."):
                    pred = engine.predict_future(h_team, a_team)
                
                # 3. THE REVEAL
                st.markdown("---")
                
                win_prob = max(pred.values())
                if pred['H'] == win_prob:
                    winner_text = f"{h_team.upper()} WIN"
                    accent_color = "#32f99a"
                elif pred['A'] == win_prob:
                    winner_text = f"{a_team.upper()} WIN"
                    accent_color = "#ff0055"
                else:
                    winner_text = "DRAW"
                    accent_color = "#888888"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: center; margin: 30px 0;">
                    <div style="background: linear-gradient(135deg, #1e1e24 0%, #121214 100%); border: 3px solid {accent_color}; padding: 40px 60px; text-align: center; box-shadow: 0 0 50px {accent_color}30; clip-path: polygon(5% 0%, 100% 0%, 100% 90%, 95% 100%, 0% 100%, 0% 10%);">
                        <h4 style="margin:0; color: #888; letter-spacing: 2px;">PREDICTED RESULT</h4>
                        <h1 style="margin: 10px 0; font-size: 5rem !important; color: {accent_color}; text-shadow: 0 0 25px {accent_color};">{winner_text}</h1>
                        <p style="margin:0; font-size: 1.5rem; color: white;">CONFIDENCE: <span style="color:{accent_color}; font-weight:800;">{int(win_prob*100)}%</span></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 4. STATS BREAKDOWN
                m1, m2, m3 = st.columns(3)
                m1.metric("HOME WIN", f"{int(pred['H']*100)}%")
                m2.metric("DRAW", f"{int(pred['D']*100)}%")
                m3.metric("AWAY WIN", f"{int(pred['A']*100)}%")

# ==================================================
# PAGE 2: LEADERBOARDS (Unchanged)
# ==================================================
elif page == "LEADERBOARDS":
    st.title("GLOBAL RANKINGS")
    st.write("LIVE ELO RATINGS TRACKER")
    
    elo_data = pd.DataFrame(list(engine.elo.ratings.items()), columns=['Club', 'Rating'])
    elo_data = elo_data[elo_data['Club'].isin(current_teams)]
    elo_data['Rating'] = elo_data['Rating'].astype(int)
    elo_data = elo_data.sort_values('Rating', ascending=False).reset_index(drop=True)
    elo_data.index += 1
    
    top3 = elo_data.head(3)
    c1, c2, c3 = st.columns(3)
    
    with c2:
        st.markdown(f"<div style='text-align:center; padding: 30px; border: 3px solid #ffd700; background: linear-gradient(to bottom, #1a1a1d, #ffd70020); clip-path: polygon(10% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 10%);'>🥇 1ST<br><h1 style='color:#ffd700; font-size: 2.5rem !important;'>{top3.iloc[0]['Club']}</h1><h2 style='color:white;'>{top3.iloc[0]['Rating']}</h2></div>", unsafe_allow_html=True)
    with c1:
        st.markdown(f"<div style='text-align:center; padding: 20px; border: 3px solid #c0c0c0; background: linear-gradient(to bottom, #1a1a1d, #c0c0c020); margin-top: 30px; clip-path: polygon(10% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 10%);'>🥈 2ND<br><h3>{top3.iloc[1]['Club']}</h3><h3>{top3.iloc[1]['Rating']}</h3></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align:center; padding: 20px; border: 3px solid #cd7f32; background: linear-gradient(to bottom, #1a1a1d, #cd7f3220); margin-top: 30px; clip-path: polygon(10% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 10%);'>🥉 3RD<br><h3>{top3.iloc[2]['Club']}</h3><h3>{top3.iloc[2]['Rating']}</h3></div>", unsafe_allow_html=True)

    st.write("")
    st.write("### FULL STANDINGS")
    
    st.dataframe(
        elo_data, 
        use_container_width=True,
        column_config={"Rating": st.column_config.ProgressColumn("Skill Rating", format="%d", min_value=1300, max_value=2200)},
        height=600
    )

# ==================================================
# PAGE 3: MATCH REPLAY (Unchanged)
# ==================================================
elif page == "MATCH REPLAY":
    st.title("PERFORMANCE REVIEW")
    st.write("MODEL ACCURACY (LAST 20 GAMES)")
    
    with st.spinner("ANALYZING MATCH FOOTAGE..."):
        history_df = engine.evaluate_recent_performance(n_games=20)
    
    acc = len(history_df[history_df['Status'] == '✅']) / len(history_df)
    
    k1, k2 = st.columns(2)
    k1.metric("RECENT ACCURACY", f"{int(acc*100)}%")
    k2.metric("GAMES ANALYZED", "20")
    
    st.dataframe(history_df, use_container_width=True)

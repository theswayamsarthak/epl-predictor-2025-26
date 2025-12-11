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
# 1. APP CONFIGURATION & THEME
# ============================================
st.set_page_config(
    page_title="PL Matchday Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premier League Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    :root {
        --pl-bg: #38003c;
        --pl-accent-green: #00ff85;
        --pl-accent-pink: #e90052;
        --pl-text-light: #ffffff;
        --pl-card-bg: #2c0030;
        --pl-secondary-text: #d6cbe2;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: var(--pl-bg);
        color: var(--pl-text-light);
    }

    h1, h2, h3 {
        color: var(--pl-text-light) !important;
        font-weight: 700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--pl-card-bg);
        border-right: 2px solid var(--pl-accent-green);
    }
    [data-testid="stSidebar"] .stRadio label {
        color: var(--pl-text-light) !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: transparent;
        border: 1px solid transparent;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"]:hover {
        background-color: rgba(0, 255, 133, 0.1);
        border-color: var(--pl-accent-green);
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: var(--pl-accent-green) !important;
        color: var(--pl-bg) !important;
    }

    /* Dropdowns & Inputs */
    .stSelectbox > div > div {
        background-color: var(--pl-card-bg) !important;
        color: var(--pl-text-light) !important;
        border: 1px solid var(--pl-accent-green);
        border-radius: 8px;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: var(--pl-text-light) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--pl-accent-green), #00cc6a);
        color: var(--pl-bg);
        font-weight: 700;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 255, 133, 0.4);
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        background-color: var(--pl-card-bg);
        border: 1px solid var(--pl-accent-pink);
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stDataFrame"] th {
        background-color: var(--pl-bg) !important;
        color: var(--pl-accent-green) !important;
        font-weight: 600;
        border-bottom: 2px solid var(--pl-accent-pink) !important;
    }
    [data-testid="stDataFrame"] td {
        color: var(--pl-text-light) !important;
        border-bottom: 1px solid #4a0e4f !important;
    }

    /* Custom Elements */
    .pl-header {
        background: linear-gradient(90deg, var(--pl-bg) 0%, var(--pl-card-bg) 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid var(--pl-accent-pink);
        margin-bottom: 30px;
        display: flex;
        align-items: center;
    }
    .pl-header-logo {
        width: 80px;
        margin-right: 20px;
    }
    .pl-card {
        background-color: var(--pl-card-bg);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 133, 0.2);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .prediction-bar {
        display: flex;
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        margin-top: 15px;
    }
    .bar-home { background-color: var(--pl-accent-green); color: var(--pl-bg); display: flex; align-items: center; justify-content: center; font-weight: bold;}
    .bar-draw { background-color: #888; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;}
    .bar-away { background-color: var(--pl-accent-pink); color: white; display: flex; align-items: center; justify-content: center; font-weight: bold;}
    
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. BACKEND: ELO & PREDICTION ENGINE
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
            # "https://www.football-data.co.uk/mmz4281/2526/E0.csv" # Uncomment for next season
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
                # Identify current season's teams from the latest URL
                if url == self.urls[-1]:
                    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()
                    self.current_season_teams = sorted(teams)
                frames.append(df)
            except Exception as e:
                st.warning(f"Could not fetch data from {url}: {e}")
        
        if not frames: return False

        self.data = pd.concat(frames, ignore_index=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True, errors='coerce')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.matches = self.data[self.data['FTR'].notna()].copy()
        
        # Fallback if current_season_teams is empty
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
            # Update Elo *after* getting features for prediction
            res_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, res_val)

        self.X = pd.DataFrame(features)
        self.y = self.matches['Result_Code']
        self.model.fit(self.X, self.y)

    def _get_recent(self, team, idx):
        # Get last 5 matches *before* the current match index
        return self.matches[((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team)) & (self.matches.index < idx)].tail(5)

    def _get_pts(self, matches, team):
        if matches.empty: return 1.0 # Default form
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
            if m['HomeTeam'] == team:
                gd += (m['FTHG'] - m['FTAG'])
            else:
                gd += (m['FTAG'] - m['FTHG'])
        return gd / len(matches)

    def predict_future(self, h_team, a_team):
        if h_team not in self.le_team.classes_ or a_team not in self.le_team.classes_: return None
        
        # Use current Elo and form from the *end* of the data
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
        # Assuming the label encoder sorts as A, D, H
        return {'A': probs[0], 'D': probs[1], 'H': probs[2]}

    def get_prediction_history(self, n=20):
        # Get the last N matches
        history_matches = self.matches.tail(n).copy().reset_index(drop=True)
        results = []
        
        # We need to simulate the state of the model *before* each match
        # This is computationally expensive to do perfectly (retraining).
        # A simpler approach for this demo is to use the features we already calculated
        # which represent the pre-match state.
        
        start_idx = self.matches.index[-n]
        features_subset = self.X.iloc[start_idx:].reset_index(drop=True)
        
        # Get predictions for these matches
        probs_all = self.model.predict_proba(features_subset)
        
        for i, row in history_matches.iterrows():
            probs = probs_all[i]
            # Determine predicted outcome (highest probability)
            if probs[2] > probs[0] and probs[2] > probs[1]: pred = 'H'
            elif probs[0] > probs[2] and probs[0] > probs[1]: pred = 'A'
            else: pred = 'D'
            
            correct = '✅' if pred == row['FTR'] else '❌'
            
            results.append({
                'Date': row['Date'].strftime('%d-%b-%y'),
                'Home Team': row['HomeTeam'],
                'Away Team': row['AwayTeam'],
                'Prediction': pred,
                'Actual': row['FTR'],
                'Correct?': correct
            })
            
        return pd.DataFrame(results)

# ============================================
# 3. APP INITIALIZATION
# ============================================
@st.cache_resource
def load_app():
    eng = EPLPredictor()
    with st.spinner("Booting up Premier League Engine..."):
        if eng.fetch_data():
            eng.run_training_cycle()
            return eng
    return None

engine = load_app()

if not engine:
    st.error("Failed to load data. Please check your connection and try again.")
    st.stop()

# ============================================
# 4. SIDEBAR NAVIGATION
# ============================================
st.sidebar.image("https://www.premierleague.com/resources/rebrand/v7.0.10/i/elements/pl-main-logo.png", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Match Predictor", "Elo Rankings", "Prediction History"])
st.sidebar.markdown("---")
st.sidebar.caption("Model: Logistic Regression with Elo & Form")
st.sidebar.caption("Data: football-data.co.uk")

# ============================================
# 5. PAGE IMPLEMENTATION
# ============================================

# --- PAGE 1: MATCH PREDICTOR ---
if page == "Match Predictor":
    st.markdown("""
        <div class="pl-header">
            <img src="https://www.premierleague.com/resources/rebrand/v7.0.10/i/elements/pl-main-logo.png" class="pl-header-logo">
            <div>
                <h1 style="margin:0;">Matchday <span style="color: var(--pl-accent-green);">Predictor</span></h1>
                <p style="color: var(--pl-secondary-text); margin:0;">AI-Powered forecasts for the upcoming fixtures.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    teams = engine.current_season_teams
    
    col1, col2, col3 = st.columns([1, 0.2, 1])
    
    with col1:
        st.markdown("### Home Team")
        h_team = st.selectbox("Select Home Team", teams, index=0)
        st.markdown(f"""
            <div class="pl-card">
                <h2 style="color: var(--pl-accent-green);">{h_team}</h2>
                <p>Elo Rating: <strong>{engine.elo.get_rating(h_team):.0f}</strong></p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h1 style='text-align: center; padding-top: 100px; color: var(--pl-accent-pink) !important;'>VS</h1>", unsafe_allow_html=True)

    with col3:
        st.markdown("### Away Team")
        a_team = st.selectbox("Select Away Team", teams, index=len(teams)-1)
        st.markdown(f"""
            <div class="pl-card">
                <h2 style="color: var(--pl-accent-pink);">{a_team}</h2>
                <p>Elo Rating: <strong>{engine.elo.get_rating(a_team):.0f}</strong></p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    if st.button("Predict Match Result", use_container_width=True):
        if h_team == a_team:
            st.warning("Please select two different teams.")
        else:
            preds = engine.predict_future(h_team, a_team)
            if preds:
                prob_h = preds['H'] * 100
                prob_d = preds['D'] * 100
                prob_a = preds['A'] * 100
                
                st.markdown("### Prediction Summary")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric(label=f"{h_team} Win", value=f"{prob_h:.1f}%", delta="Home")
                m2.metric(label="Draw", value=f"{prob_d:.1f}%")
                m3.metric(label=f"{a_team} Win", value=f"{prob_a:.1f}%", delta="-Away", delta_color="inverse")

                # Visual Bar
                st.markdown(f"""
                    <div class="prediction-bar">
                        <div class="bar-home" style="width: {prob_h}%;">{int(prob_h)}%</div>
                        <div class="bar-draw" style="width: {prob_d}%;">{int(prob_d)}%</div>
                        <div class="bar-away" style="width: {prob_a}%;">{int(prob_a)}%</div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px; color: var(--pl-secondary-text);">
                        <span>{h_team}</span>
                        <span>{a_team}</span>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.error("Prediction failed. Check team selection.")

# --- PAGE 2: ELO RANKINGS ---
elif page == "Elo Rankings":
    st.markdown("""
        <div class="pl-header">
            <div>
                <h1 style="margin:0;">Premier League <span style="color: var(--pl-accent-green);">Elo Rankings</span></h1>
                <p style="color: var(--pl-secondary-text); margin:0;">Live power rankings based on team performance.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Create DataFrame from ratings dictionary
    elo_df = pd.DataFrame(list(engine.elo.ratings.items()), columns=['Team', 'Rating'])
    # Filter for current season teams only
    elo_df = elo_df[elo_df['Team'].isin(engine.current_season_teams)]
    # Sort by rating
    elo_df = elo_df.sort_values(by='Rating', ascending=False).reset_index(drop=True)
    # Add ranking column
    elo_df.index = elo_df.index + 1
    elo_df.index.name = 'Rank'
    elo_df['Rating'] = elo_df['Rating'].round(0).astype(int)

    st.dataframe(elo_df, use_container_width=True)

# --- PAGE 3: PREDICTION HISTORY ---
elif page == "Prediction History":
    st.markdown("""
        <div class="pl-header">
            <div>
                <h1 style="margin:0;">Prediction <span style="color: var(--pl-accent-pink);">History</span></h1>
                <p style="color: var(--pl-secondary-text); margin:0;">Model performance over the last 20 matches.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    history_df = engine.get_prediction_history(n=20)
    
    # Calculate accuracy
    accuracy = (history_df['Correct?'] == '✅').mean()
    st.metric("Recent Accuracy (Last 20 Games)", f"{accuracy*100:.1f}%")
    
    st.dataframe(history_df, use_container_width=True)

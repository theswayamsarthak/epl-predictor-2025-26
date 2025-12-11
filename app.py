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
# 1. SETUP & CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Premier League AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide default menu and footer for a cleaner look
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ============================================
# 2. CORE LOGIC (ELO & MODEL)
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
        self.team_stats = {} # Store history for plotting

    def fetch_data(self):
        frames = []
        for url in self.urls:
            try:
                s = requests.get(url).content
                df = pd.read_csv(io.StringIO(s.decode('latin-1')))
                df = df.dropna(how='all')
                frames.append(df)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        
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
        
        features = []
        # Initialize history tracking for charts
        for team in all_teams:
            self.team_stats[team] = {'elo_history': [], 'dates': []}

        for idx, row in self.matches.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']
            date = row['Date']
            
            h_elo = self.elo.get_rating(h_team)
            a_elo = self.elo.get_rating(a_team)
            
            # Record history
            self.team_stats[h_team]['elo_history'].append(h_elo)
            self.team_stats[h_team]['dates'].append(date)
            self.team_stats[a_team]['elo_history'].append(a_elo)
            self.team_stats[a_team]['dates'].append(date)

            # Feature Vector
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
            
            # Update Elo
            result_val = 1.0 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0.0
            self.elo.update_ratings(h_team, a_team, result_val)

        self.X = pd.DataFrame(features)
        self.y = self.matches['Result_Code']

    def _get_recent_matches(self, team, current_idx):
        # Helper to get last 5 matches for a team before current_idx
        team_matches = self.matches[
            ((self.matches['HomeTeam'] == team) | (self.matches['AwayTeam'] == team)) & 
            (self.matches.index < current_idx)
        ]
        return team_matches.tail(5)

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
        
        # Get all past matches for current form
        all_matches = self.matches
        current_idx = len(all_matches) + 1 # Future match
        
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
# 3. THE STREAMLIT APP (UI)
# ============================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Data sourced from football-data.co.uk")
    st.write("This model uses Elo ratings and recent form (last 5 games) to predict outcomes.")
    st.markdown("---")
    st.write("**Model Version:** 1.0.0")
    st.write("**Engine:** Logistic Regression")

# --- MAIN LOADER ---
@st.cache_resource
def load_engine():
    engine = EPLPredictor()
    # Simple placeholder while loading
    with st.spinner('Crunching numbers from the 24/25 & 25/26 seasons...'):
        if not engine.fetch_data(): return None
        engine.prepare_features()
        engine.train_model()
    return engine

engine = load_engine()

# --- HEADER ---
st.title("⚽ Premier League Predictor")
st.markdown("#### AI-Powered Match Forecasting")
st.markdown("---")

if engine:
    # TABS FOR ORGANIZATION
    tab1, tab2 = st.tabs(["🔮 Match Predictor", "📈 Team Stats"])

    # --- TAB 1: PREDICTOR ---
    with tab1:
        st.write("### Select Teams")
        
        col_h, col_vs, col_a = st.columns([4, 1, 4])
        team_list = sorted(engine.le_team.classes_)

        with col_h:
            home_team = st.selectbox("Home Team", team_list, index=0, key="home")
        
        with col_vs:
            st.markdown("<h2 style='text-align: center; margin-top: 25px;'>VS</h2>", unsafe_allow_html=True)
        
        with col_a:
            away_team = st.selectbox("Away Team", team_list, index=1, key="away")

        st.markdown("---")

        if st.button("Analyze Matchup", type="primary", use_container_width=True):
            if home_team == away_team:
                st.warning("Please select two different teams.")
            else:
                pred = engine.predict_next_match(home_team, away_team)
                if pred:
                    # Determine Winner
                    winner_val = max(pred, key=pred.get)
                    
                    # Display Big Result
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Home Win", f"{pred['Home']*100:.1f}%", delta_color="normal")
                    c2.metric("Draw", f"{pred['Draw']*100:.1f}%", delta_color="off")
                    c3.metric("Away Win", f"{pred['Away']*100:.1f}%", delta_color="normal")

                    # Visual Progress Bar
                    st.write("")
                    st.write("##### Probability Distribution")
                    st.progress(pred['Home'], text=f"Home Strength: {pred['Home']*100:.0f}%")
                    st.progress(pred['Home'] + pred['Draw'], text=f"Draw Probability region")
                    
                    st.success(f"**Insight:** The model favors **{winner_val}** for this match.")

    # --- TAB 2: TEAM STATS ---
    with tab2:
        st.write("### Team Performance History")
        selected_team = st.selectbox("Select Team to View Elo History", team_list)
        
        if selected_team in engine.team_stats:
            stats = engine.team_stats[selected_team]
            dates = stats['dates']
            elos = stats['elo_history']
            
            # Plotting with Matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, elos, marker='o', linestyle='-', color='#00ff85') # EPL Green color
            ax.set_title(f"Elo Rating Progression: {selected_team}")
            ax.set_ylabel("Elo Rating")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
        else:
            st.info("Not enough data history for this team.")

else:
    st.error("Failed to load data. Please refresh the page.")
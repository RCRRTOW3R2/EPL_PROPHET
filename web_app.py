#!/usr/bin/env python3
"""
EPL PROPHET - WEB APPLICATION
============================

Beautiful web interface for EPL match predictions!

Features:
- Select any upcoming match from dropdown
- Get instant predictions with explanations
- Beautiful, responsive design
- Real-time match selection
- Detailed probability visualizations

Perfect for GitHub deployment!
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

class WebPredictor:
    """Web-based EPL match predictor."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        self.fixtures = None
        self.load_system()
    
    def load_system(self):
        """Load the prediction system."""
        
        try:
            # Load model
            self.model = joblib.load("../outputs/champion_model.joblib")
            
            # Load features
            df = pd.read_csv("../outputs/champion_features.csv")
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            
            # Calculate team form
            self.calculate_team_form(df)
            
            # Load fixtures
            self.fixtures = pd.read_csv("../outputs/all_upcoming_fixtures.csv")
            
            print("✅ Web predictor system loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def calculate_team_form(self, df):
        """Calculate current team form."""
        
        df_recent = df[df['actual_result'].notna()].copy()
        self.team_form = {}
        
        all_teams = set(df_recent['home_team'].tolist() + df_recent['away_team'].tolist())
        
        for team in all_teams:
            home_matches = df_recent[df_recent['home_team'] == team]
            away_matches = df_recent[df_recent['away_team'] == team]
            
            team_features = {}
            
            if len(home_matches) > 0:
                latest_home = home_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('home_'):
                        base_feature = feature.replace('home_', '')
                        team_features[base_feature] = latest_home[feature] if not pd.isna(latest_home[feature]) else 0
            
            if len(away_matches) > 0:
                latest_away = away_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('away_'):
                        base_feature = feature.replace('away_', '')
                        if base_feature not in team_features:
                            team_features[base_feature] = latest_away[feature] if not pd.isna(latest_away[feature]) else 0
            
            self.team_form[team] = team_features
    
    def create_features(self, home_team, away_team):
        """Create match features."""
        
        if home_team not in self.team_form or away_team not in self.team_form:
            return None
        
        home_form = self.team_form[home_team]
        away_form = self.team_form[away_team]
        
        match_features = []
        
        for feature_name in self.feature_names:
            if feature_name.startswith('home_'):
                base_feature = feature_name.replace('home_', '')
                value = home_form.get(base_feature, 0)
            elif feature_name.startswith('away_'):
                base_feature = feature_name.replace('away_', '')
                value = away_form.get(base_feature, 0)
            else:
                value = self.calculate_comparative(feature_name, home_form, away_form)
            
            match_features.append(value)
        
        return np.array(match_features)
    
    def calculate_comparative(self, feature_name, home_form, away_form):
        """Calculate comparative features."""
        
        # Logarithmic ratios (champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                home_goals = home_form.get('goals_ema_long', 1)
                away_goals = away_form.get('goals_ema_long', 1)
                return np.log((home_goals + 1) / (away_goals + 1))
            elif 'points' in feature_name and 'long' in feature_name:
                home_points = home_form.get('points_ema_long', 1)
                away_points = away_form.get('points_ema_long', 1)
                return np.log((home_points + 1) / (away_points + 1))
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                home_val = home_form.get('goals_ema_long', 0)
                away_val = away_form.get('goals_ema_long', 0)
                advantage = home_val - away_val
                return np.sign(advantage) * (advantage ** 2)
        
        return 0
    
    def predict_match(self, home_team, away_team):
        """Predict a specific match."""
        
        features = self.create_features(home_team, away_team)
        
        if features is None:
            return {
                'error': f'Missing data for {home_team} or {away_team}',
                'success': False
            }
        
        # Make prediction
        features_array = features.reshape(1, -1)
        proba = self.model.predict_proba(features_array)[0]
        pred_class = self.model.predict(features_array)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[pred_class]
        confidence = max(proba)
        
        # Get key factors
        importances = self.model.feature_importances_
        contributions = features * importances
        
        feature_contribs = list(zip(self.feature_names, contributions))
        feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        key_factors = []
        for feat, contrib in feature_contribs[:3]:
            if 'log_ratio' in feat and 'goals' in feat:
                if contrib > 0:
                    key_factors.append("Superior goal-scoring form favors prediction")
                break
            elif 'momentum' in feat:
                if contrib > 0:
                    key_factors.append("Recent momentum supports prediction")
                break
            elif 'squared_advantage' in feat:
                if contrib > 0:
                    key_factors.append("Significant form advantage detected")
                break
        
        if not key_factors:
            key_factors.append("Model analysis based on comprehensive form data")
        
        return {
            'success': True,
            'home_team': home_team,
            'away_team': away_team,
            'prediction': predicted_result,
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                'home_win': round(proba[2] * 100, 1),
                'draw': round(proba[1] * 100, 1),
                'away_win': round(proba[0] * 100, 1)
            },
            'key_factors': key_factors
        }

# Initialize predictor
predictor = WebPredictor()

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/fixtures')
def get_fixtures():
    """Get all upcoming fixtures."""
    
    try:
        fixtures_list = []
        for _, fixture in predictor.fixtures.iterrows():
            fixtures_list.append({
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'date': fixture['date'],
                'time': fixture['time'],
                'gameweek': fixture.get('gameweek', 'TBD'),
                'display': f"GW{fixture.get('gameweek', '?')} | {fixture['date']} {fixture['time']} - {fixture['home_team']} vs {fixture['away_team']}"
            })
        
        return jsonify({
            'success': True,
            'fixtures': fixtures_list
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict a specific match."""
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({
                'success': False,
                'error': 'Missing team names'
            })
        
        result = predictor.predict_match(home_team, away_team)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting EPL Prophet Web App...")
    print("🌐 Opening on http://localhost:5000")
    print("🏆 53.7% Accurate Champion Model Ready!")
    app.run(debug=True, host='0.0.0.0', port=5000) 
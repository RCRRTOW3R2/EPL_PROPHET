#!/usr/bin/env python3
"""
EPL PROPHET - DEPLOYMENT WEB APPLICATION
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

class DeploymentPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        self.fixtures = None
        self.load_system()
    
    def load_system(self):
        try:
            print("🏆 Loading deployment system...")
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            outputs_dir = os.path.join(base_dir, '..', 'outputs')
            
            # Load model
            model_path = os.path.join(outputs_dir, 'champion_model.joblib')
            if not os.path.exists(model_path):
                alt_models = ['enhanced_random_forest.joblib', 'phase2_short_term.joblib']
                for alt_model in alt_models:
                    alt_path = os.path.join(outputs_dir, alt_model)
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"   📁 Using model: {alt_model}")
                        break
            
            self.model = joblib.load(model_path)
            print("   ✅ Model loaded successfully")
            
            # Load features
            features_path = os.path.join(outputs_dir, 'champion_features.csv')
            if not os.path.exists(features_path):
                alt_features = ['phase2_enhanced_features.csv', 'recency_weighted_stock_features.csv']
                for alt_feature in alt_features:
                    alt_path = os.path.join(outputs_dir, alt_feature)
                    if os.path.exists(alt_path):
                        features_path = alt_path
                        break
            
            df = pd.read_csv(features_path)
            
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals', 'Unnamed: 0']
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            self.calculate_team_form(df)
            print(f"   ✅ Team form calculated")
            
            # Load fixtures
            fixtures_path = os.path.join(outputs_dir, 'all_upcoming_fixtures.csv')
            if os.path.exists(fixtures_path):
                self.fixtures = pd.read_csv(fixtures_path)
            else:
                self.create_sample_fixtures()
            
            print("🎉 Deployment system ready!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def create_sample_fixtures(self):
        sample_fixtures = [
            {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'date': '2025-01-26', 'time': '16:30', 'gameweek': 22},
            {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'date': '2025-01-26', 'time': '14:00', 'gameweek': 22},
            {'home_team': 'Manchester United', 'away_team': 'Tottenham', 'date': '2025-01-27', 'time': '16:30', 'gameweek': 22},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2025-01-27', 'time': '19:00', 'gameweek': 22},
            {'home_team': 'Brighton', 'away_team': 'West Ham', 'date': '2025-01-28', 'time': '20:00', 'gameweek': 22}
        ]
        self.fixtures = pd.DataFrame(sample_fixtures)
        print(f"   ✅ Created {len(sample_fixtures)} sample fixtures")
    
    def calculate_team_form(self, df):
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
    
    def predict_match(self, home_team, away_team):
        if home_team not in self.team_form or away_team not in self.team_form:
            return {'error': f'Missing data for {home_team} or {away_team}', 'success': False}
        
        try:
            # Create simplified features
            home_form = self.team_form[home_team]
            away_form = self.team_form[away_team]
            
            # Get available features
            available_features = []
            for feature_name in self.feature_names:
                if feature_name.startswith('home_'):
                    base_feature = feature_name.replace('home_', '')
                    value = home_form.get(base_feature, 0)
                elif feature_name.startswith('away_'):
                    base_feature = feature_name.replace('away_', '')
                    value = away_form.get(base_feature, 0)
                else:
                    value = 0
                
                available_features.append(value)
            
            features_array = np.array(available_features).reshape(1, -1)
            
            # Ensure feature count matches model
            if features_array.shape[1] != len(self.feature_names):
                # Pad or truncate to match
                target_size = len(self.feature_names)
                if features_array.shape[1] < target_size:
                    padding = np.zeros((1, target_size - features_array.shape[1]))
                    features_array = np.hstack([features_array, padding])
                else:
                    features_array = features_array[:, :target_size]
            
            proba = self.model.predict_proba(features_array)[0]
            pred_class = self.model.predict(features_array)[0]
            
            result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
            predicted_result = result_map[pred_class]
            confidence = max(proba)
            
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
                'key_factors': ["Model analysis based on team form and statistics"]
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}', 'success': False}

# Initialize predictor
print("🚀 Initializing deployment predictor...")
predictor = DeploymentPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fixtures')
def get_fixtures():
    try:
        fixtures_list = []
        for _, fixture in predictor.fixtures.iterrows():
            fixtures_list.append({
                'home_team': fixture['home_team'],
                'away_team': fixture['away_team'],
                'date': fixture.get('date', 'TBD'),
                'time': fixture.get('time', 'TBD'),
                'gameweek': fixture.get('gameweek', 'TBD'),
                'display': f"GW{fixture.get('gameweek', '?')} | {fixture.get('date', 'TBD')} {fixture.get('time', 'TBD')} - {fixture['home_team']} vs {fixture['away_team']}"
            })
        
        return jsonify({'success': True, 'fixtures': fixtures_list})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'error': 'Missing team names'})
        
        result = predictor.predict_match(home_team, away_team)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

if __name__ == '__main__':
    print("🚀 Starting EPL Prophet Deployment App...")
    print("🌐 Opening on http://localhost:8080")
    print("🏆 Ready for deployment!")
    app.run(debug=True, host='0.0.0.0', port=8080)

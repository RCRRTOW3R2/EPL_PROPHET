#!/usr/bin/env python3
"""
EPL PROPHET - DEPLOYMENT WEB APPLICATION
========================================

Fixed version for deployment with correct file paths!
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
    """Deployment-ready EPL match predictor."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        self.fixtures = None
        self.load_system()
    
    def load_system(self):
        """Load the prediction system with correct file paths."""
        
        try:
            print("🏆 Loading deployment system...")
            
            # Get the correct paths relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            outputs_dir = os.path.join(base_dir, '..', 'outputs')
            
            # Load champion model
            model_path = os.path.join(outputs_dir, 'champion_model.joblib')
            if not os.path.exists(model_path):
                # Try alternative model names
                alternative_models = [
                    'champion_rf.joblib',
                    'enhanced_random_forest.joblib',
                    'phase2_short_term.joblib'
                ]
                
                for alt_model in alternative_models:
                    alt_path = os.path.join(outputs_dir, alt_model)
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        print(f"   📁 Using model: {alt_model}")
                        break
                else:
                    raise FileNotFoundError("No suitable model file found")
            
            self.model = joblib.load(model_path)
            print("   ✅ Model loaded successfully")
            
            # Load features
            features_path = os.path.join(outputs_dir, 'champion_features.csv')
            if not os.path.exists(features_path):
                # Try alternative feature files
                alternative_features = [
                    'phase2_enhanced_features.csv',
                    'recency_weighted_stock_features.csv',
                    'enhanced_match_features_v2.csv'
                ]
                
                for alt_features in alternative_features:
                    alt_path = os.path.join(outputs_dir, alt_features)
                    if os.path.exists(alt_path):
                        features_path = alt_path
                        print(f"   📁 Using features: {alt_features}")
                        break
            
            df = pd.read_csv(features_path)
            
            # Get feature names
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals', 'Unnamed: 0']
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            # Calculate team form
            self.calculate_team_form(df)
            print(f"   ✅ Team form calculated for {len(self.team_form)} teams")
            
            # Load fixtures
            fixtures_path = os.path.join(outputs_dir, 'all_upcoming_fixtures.csv')
            if not os.path.exists(fixtures_path):
                fixtures_path = os.path.join(outputs_dir, 'upcoming_fixtures.csv')
            
            if os.path.exists(fixtures_path):
                self.fixtures = pd.read_csv(fixtures_path)
                print(f"   ✅ {len(self.fixtures)} fixtures loaded")
            else:
                print("   ⚠️  No fixture file found, creating sample data")
                self.create_sample_fixtures()
            
            print("🎉 Deployment system ready!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading system: {e}")
            print(f"   📁 Attempted paths: {base_dir}")
            print(f"   📁 Outputs dir: {outputs_dir}")
            return False
    
    def create_sample_fixtures(self):
        """Create sample fixtures if none found."""
        
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
        
        try:
            # Make prediction
            features_array = features.reshape(1, -1)
            proba = self.model.predict_proba(features_array)[0]
            pred_class = self.model.predict(features_array)[0]
            
            result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
            predicted_result = result_map[pred_class]
            confidence = max(proba)
            
            # Get key factors
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                contributions = features * importances
                
                feature_contribs = list(zip(self.feature_names, contributions))
                feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                key_factors = []
                for feat, contrib in feature_contribs[:3]:
                    if 'log_ratio' in feat and 'goals' in feat and contrib > 0:
                        key_factors.append("Superior goal-scoring form favors prediction")
                        break
                    elif 'momentum' in feat and contrib > 0:
                        key_factors.append("Recent momentum supports prediction")
                        break
                    elif 'squared_advantage' in feat and contrib > 0:
                        key_factors.append("Significant form advantage detected")
                        break
                
                if not key_factors:
                    key_factors.append("Model analysis based on comprehensive form data")
            else:
                key_factors = ["Model analysis based on comprehensive data"]
            
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
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }

# Initialize predictor
print("🚀 Initializing deployment predictor...")
predictor = DeploymentPredictor()

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
                'date': fixture.get('date', 'TBD'),
                'time': fixture.get('time', 'TBD'),
                'gameweek': fixture.get('gameweek', 'TBD'),
                'display': f"GW{fixture.get('gameweek', '?')} | {fixture.get('date', 'TBD')} {fixture.get('time', 'TBD')} - {fixture['home_team']} vs {fixture['away_team']}"
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

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

if __name__ == '__main__':
    print("🚀 Starting EPL Prophet Deployment App...")
    print("🌐 Opening on http://localhost:8080")
    print("🏆 Ready for deployment!")
    
    # Use port 8080 to avoid conflicts with AirPlay
    app.run(debug=True, host='0.0.0.0', port=8080) 
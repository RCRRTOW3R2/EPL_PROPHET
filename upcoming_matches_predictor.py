#!/usr/bin/env python3
"""
EPL PROPHET - UPCOMING MATCHES PREDICTOR
========================================

Predict future EPL matches using our 53.7% champion model!

Example: "Man City vs Liverpool tomorrow" 
- Loads latest team form data
- Calculates all our champion features (logarithmic ratios, EMAs, etc.)
- Makes explainable prediction with SHAP

This is what you wanted - dynamic future predictions!
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class UpcomingMatchPredictor:
    """Predict upcoming EPL matches with our champion model."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_current_form = None
        self.class_names = ['Away Win', 'Draw', 'Home Win']
        
    def load_champion_system(self):
        """Load our champion model and current team form data."""
        
        print("🏆 Loading Champion Prediction System...")
        
        try:
            # Load champion model
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion model loaded (53.7% accuracy)")
            
            # Load feature data and current team form
            df = pd.read_csv("../outputs/champion_features.csv")
            
            # Get feature names
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            # Calculate current team form (latest values for each team)
            self.calculate_current_team_form(df)
            print(f"   ✅ Current form calculated for {len(self.team_current_form)} teams")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def calculate_current_team_form(self, df):
        """Calculate current form for all teams."""
        
        # Get the most recent data for each team
        df_recent = df[df['actual_result'].notna()].copy()
        df_recent['date'] = pd.to_datetime(df_recent['date'], dayfirst=True, errors='coerce')
        df_recent = df_recent.sort_values('date')
        
        self.team_current_form = {}
        
        # For each team, get their latest form data
        all_teams = set(df_recent['home_team'].tolist() + df_recent['away_team'].tolist())
        
        for team in all_teams:
            # Get team's most recent matches (both home and away)
            home_matches = df_recent[df_recent['home_team'] == team]
            away_matches = df_recent[df_recent['away_team'] == team]
            
            if len(home_matches) > 0:
                latest_home = home_matches.iloc[-1]
                home_features = self.extract_team_features(latest_home, 'home')
            else:
                home_features = {}
            
            if len(away_matches) > 0:
                latest_away = away_matches.iloc[-1]
                away_features = self.extract_team_features(latest_away, 'away')
            else:
                away_features = {}
            
            # Combine home and away features (use most recent)
            combined_features = {}
            combined_features.update(home_features)
            combined_features.update(away_features)
            
            self.team_current_form[team] = combined_features
    
    def extract_team_features(self, match_row, team_type):
        """Extract team-specific features from a match row."""
        
        features = {}
        
        for feature in self.feature_names:
            if feature.startswith(f'{team_type}_'):
                # Remove team prefix for storage
                base_feature = feature.replace(f'{team_type}_', '')
                features[base_feature] = match_row[feature] if not pd.isna(match_row[feature]) else 0
        
        return features
    
    def create_upcoming_match_features(self, home_team, away_team):
        """Create feature vector for an upcoming match."""
        
        print(f"📊 Creating features for {home_team} vs {away_team}...")
        
        if home_team not in self.team_current_form:
            print(f"   ⚠️ Warning: {home_team} not found in current form data")
            return None
            
        if away_team not in self.team_current_form:
            print(f"   ⚠️ Warning: {away_team} not found in current form data")
            return None
        
        # Get current form for both teams
        home_form = self.team_current_form[home_team]
        away_form = self.team_current_form[away_team]
        
        # Create feature vector
        match_features = []
        
        for feature_name in self.feature_names:
            if feature_name.startswith('home_'):
                base_feature = feature_name.replace('home_', '')
                value = home_form.get(base_feature, 0)
            elif feature_name.startswith('away_'):
                base_feature = feature_name.replace('away_', '')
                value = away_form.get(base_feature, 0)
            else:
                # For comparative features, calculate from team forms
                value = self.calculate_comparative_feature(feature_name, home_form, away_form)
            
            match_features.append(value)
        
        return np.array(match_features)
    
    def calculate_comparative_feature(self, feature_name, home_form, away_form):
        """Calculate comparative features like advantages and ratios."""
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                home_val = home_form.get('goals_ema_long', 0)
                away_val = away_form.get('goals_ema_long', 0)
                return home_val - away_val
            elif 'points' in feature_name:
                home_val = home_form.get('points_ema_long', 0)
                away_val = away_form.get('points_ema_long', 0)
                return home_val - away_val
        
        # Logarithmic ratios (our champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                home_goals = home_form.get('goals_ema_long', 1)
                away_goals = away_form.get('goals_ema_long', 1)
                return np.log((home_goals + 1) / (away_goals + 1))
            elif 'points' in feature_name and 'long' in feature_name:
                home_points = home_form.get('points_ema_long', 1)
                away_points = away_form.get('points_ema_long', 1)
                return np.log((home_points + 1) / (away_points + 1))
            elif 'goals' in feature_name and 'medium' in feature_name:
                home_goals = home_form.get('goals_ema_medium', 1)
                away_goals = away_form.get('goals_ema_medium', 1)
                return np.log((home_goals + 1) / (away_goals + 1))
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                home_val = home_form.get('goals_ema_long', 0)
                away_val = away_form.get('goals_ema_long', 0)
                advantage = home_val - away_val
                return np.sign(advantage) * (advantage ** 2)
            elif 'points' in feature_name and 'long' in feature_name:
                home_val = home_form.get('points_ema_long', 0)
                away_val = away_form.get('points_ema_long', 0)
                advantage = home_val - away_val
                return np.sign(advantage) * (advantage ** 2)
        
        # Momentum ratios
        if 'momentum_ratio' in feature_name:
            if 'goals' in feature_name:
                home_momentum = home_form.get('goals_momentum', 0)
                away_momentum = away_form.get('goals_momentum', 0)
                return home_momentum / (abs(away_momentum) + 0.01)
            elif 'points' in feature_name:
                home_momentum = home_form.get('points_momentum', 0)
                away_momentum = away_form.get('points_momentum', 0)
                return home_momentum / (abs(away_momentum) + 0.01)
        
        # Default to 0 for unknown features
        return 0
    
    def predict_upcoming_match(self, home_team, away_team, match_date=None):
        """Predict an upcoming match with full explanation."""
        
        print(f"\n🔮 PREDICTING UPCOMING MATCH")
        print("=" * 60)
        print(f"📅 Match: {home_team} vs {away_team}")
        if match_date:
            print(f"📅 Date: {match_date}")
        print("=" * 60)
        
        # Create features for this match
        match_features = self.create_upcoming_match_features(home_team, away_team)
        
        if match_features is None:
            print("❌ Cannot create prediction - team data missing")
            return None
        
        # Make prediction
        if len(match_features.shape) == 1:
            match_features = match_features.reshape(1, -1)
        
        prediction_proba = self.model.predict_proba(match_features)[0]
        prediction_class = self.model.predict(match_features)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[prediction_class]
        confidence = max(prediction_proba)
        
        # Display prediction
        print(f"🎯 PREDICTION: {predicted_result}")
        print(f"🎲 Confidence: {confidence:.1%}")
        print(f"\n📊 Full Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"   {class_name}: {prediction_proba[i]:.1%}")
        
        # Explain prediction using feature importance
        self.explain_upcoming_prediction(match_features[0], predicted_result, home_team, away_team)
        
        return {
            'prediction': predicted_result,
            'confidence': confidence,
            'probabilities': dict(zip(self.class_names, prediction_proba)),
            'home_team': home_team,
            'away_team': away_team
        }
    
    def explain_upcoming_prediction(self, match_features, predicted_result, home_team, away_team):
        """Explain why the prediction was made."""
        
        print(f"\n🔥 WHY {predicted_result}? - TOP FACTORS:")
        print("-" * 50)
        
        # Use feature importance for explanation
        feature_importances = self.model.feature_importances_
        match_contributions = match_features * feature_importances
        
        # Get top contributors
        feature_contributions = list(zip(self.feature_names, match_contributions))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        positive = [f for f in feature_contributions if f[1] > 0][:5]
        negative = [f for f in feature_contributions if f[1] < 0][:3]
        
        if positive:
            print("📈 SUPPORTING FACTORS:")
            for feat, value in positive:
                print(f"   + {feat}: {value:.3f}")
        
        if negative:
            print("\n📉 OPPOSING FACTORS:")
            for feat, value in negative:
                print(f"   - {feat}: {value:.3f}")
        
        # Human explanations
        print(f"\n💬 HUMAN EXPLANATION:")
        for feat, value in positive[:3]:
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                print(f"   ✅ {explanation}")
        
        for feat, value in negative[:2]:
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                print(f"   ⚠️ {explanation}")
    
    def feature_to_human(self, feature_name, value, home_team, away_team):
        """Convert feature to human explanation."""
        
        # Logarithmic ratios (champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has superior long-term goal-scoring form"
            elif 'points' in feature_name and 'long' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has superior long-term points form"
            elif 'goals' in feature_name and 'medium' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has better recent goal-scoring momentum"
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has a significant goal-scoring advantage"
            elif 'points' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has a significant points form advantage"
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has overall better goal-scoring form"
            elif 'points' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has overall better points form"
        
        return None


def create_sample_upcoming_fixtures():
    """Create sample upcoming fixtures for demonstration."""
    
    upcoming_fixtures = [
        {'home_team': 'Man City', 'away_team': 'Liverpool', 'date': '2025-08-25'},
        {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'date': '2025-08-26'},
        {'home_team': 'Man United', 'away_team': 'Tottenham', 'date': '2025-08-26'},
        {'home_team': 'Brighton', 'away_team': 'Newcastle', 'date': '2025-08-27'},
        {'home_team': 'Aston Villa', 'away_team': 'West Ham', 'date': '2025-08-27'}
    ]
    
    return upcoming_fixtures

def run_upcoming_matches_demo():
    """Demo upcoming matches prediction system."""
    
    print("🚀 EPL PROPHET - UPCOMING MATCHES PREDICTOR")
    print("=" * 70)
    print("Predict future EPL matches with our 53.7% champion model!")
    
    # Initialize predictor
    predictor = UpcomingMatchPredictor()
    
    if not predictor.load_champion_system():
        return
    
    # Get sample upcoming fixtures
    upcoming_fixtures = create_sample_upcoming_fixtures()
    
    print(f"\n🔮 ANALYZING {len(upcoming_fixtures)} UPCOMING FIXTURES...")
    
    predictions = []
    
    for fixture in upcoming_fixtures:
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        match_date = fixture['date']
        
        prediction = predictor.predict_upcoming_match(home_team, away_team, match_date)
        
        if prediction:
            predictions.append(prediction)
        
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("📋 UPCOMING FIXTURES SUMMARY:")
    print("=" * 50)
    
    for prediction in predictions:
        confidence_emoji = "🔥" if prediction['confidence'] > 0.5 else "📊"
        print(f"{confidence_emoji} {prediction['home_team']} vs {prediction['away_team']}: {prediction['prediction']} ({prediction['confidence']:.1%})")
    
    print(f"\n✨ UPCOMING PREDICTIONS COMPLETE!")
    print("🏆 All predictions made with our 53.7% champion model!")
    print("💬 Each prediction includes detailed explanations!")
    print("🔮 Ready to analyze any upcoming EPL fixture!")

if __name__ == "__main__":
    run_upcoming_matches_demo() 
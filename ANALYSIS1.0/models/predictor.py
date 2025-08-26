#!/usr/bin/env python3
"""
EPL PROPHET - UPCOMING MATCHES PREDICTOR
Predict future EPL matches like "Man City vs Liverpool tomorrow"!
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

class UpcomingPredictor:
    """Predict upcoming EPL matches."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        
    def load_system(self):
        """Load champion model and team form data."""
        
        print("🏆 Loading Champion System...")
        
        try:
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion model loaded (53.7% accuracy)")
            
            df = pd.read_csv("../outputs/champion_features.csv")
            
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            self.calculate_team_form(df)
            print(f"   ✅ Current form for {len(self.team_form)} teams")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def calculate_team_form(self, df):
        """Calculate current form for all teams."""
        
        df_recent = df[df['actual_result'].notna()].copy()
        self.team_form = {}
        
        all_teams = set(df_recent['home_team'].tolist() + df_recent['away_team'].tolist())
        
        for team in all_teams:
            # Get most recent form values
            home_matches = df_recent[df_recent['home_team'] == team]
            away_matches = df_recent[df_recent['away_team'] == team]
            
            team_features = {}
            
            # From home matches
            if len(home_matches) > 0:
                latest_home = home_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('home_'):
                        base_feature = feature.replace('home_', '')
                        team_features[base_feature] = latest_home[feature] if not pd.isna(latest_home[feature]) else 0
            
            # From away matches
            if len(away_matches) > 0:
                latest_away = away_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('away_'):
                        base_feature = feature.replace('away_', '')
                        if base_feature not in team_features:  # Don't overwrite home data
                            team_features[base_feature] = latest_away[feature] if not pd.isna(latest_away[feature]) else 0
            
            self.team_form[team] = team_features
    
    def predict_match(self, home_team, away_team):
        """Predict upcoming match."""
        
        print(f"\n🔮 PREDICTING: {home_team} vs {away_team}")
        print("=" * 60)
        
        if home_team not in self.team_form:
            print(f"   ❌ {home_team} not found in data")
            return None
            
        if away_team not in self.team_form:
            print(f"   ❌ {away_team} not found in data")
            return None
        
        # Create feature vector
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
                # Comparative features
                value = self.calculate_comparative(feature_name, home_form, away_form)
            
            match_features.append(value)
        
        # Make prediction
        features_array = np.array(match_features).reshape(1, -1)
        prediction_proba = self.model.predict_proba(features_array)[0]
        prediction_class = self.model.predict(features_array)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[prediction_class]
        confidence = max(prediction_proba)
        
        # Display results
        print(f"🎯 PREDICTION: {predicted_result}")
        print(f"🎲 CONFIDENCE: {confidence:.1%}")
        print(f"\n📊 PROBABILITIES:")
        class_names = ['Away Win', 'Draw', 'Home Win']
        for i, name in enumerate(class_names):
            print(f"   {name}: {prediction_proba[i]:.1%}")
        
        # Simple explanation
        self.explain_prediction(match_features, home_team, away_team)
        
        return {
            'prediction': predicted_result,
            'confidence': confidence,
            'probabilities': dict(zip(class_names, prediction_proba))
        }
    
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
            if 'goals' in feature_name and 'long' in feature_name:
                home_val = home_form.get('goals_ema_long', 0)
                away_val = away_form.get('goals_ema_long', 0)
                advantage = home_val - away_val
                return np.sign(advantage) * (advantage ** 2)
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                home_val = home_form.get('goals_ema_long', 0)
                away_val = away_form.get('goals_ema_long', 0)
                return home_val - away_val
        
        return 0  # Default
    
    def explain_prediction(self, match_features, home_team, away_team):
        """Simple explanation of prediction."""
        
        print(f"\n💬 WHY THIS PREDICTION:")
        
        # Use feature importance
        importances = self.model.feature_importances_
        contributions = np.array(match_features) * importances
        
        feature_contributions = list(zip(self.feature_names, contributions))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_positive = [f for f in feature_contributions if f[1] > 0][:3]
        
        for feat, value in top_positive:
            if 'log_ratio' in feat and 'goals' in feat:
                if value > 0:
                    print(f"   ✅ {home_team} has better goal-scoring form")
                else:
                    print(f"   ✅ {away_team} has better goal-scoring form")
            elif 'points' in feat and 'advantage' in feat:
                if value > 0:
                    print(f"   ✅ {home_team} has better points form")
                else:
                    print(f"   ✅ {away_team} has better points form")
            elif 'momentum' in feat:
                print(f"   ✅ Recent momentum favors the prediction")

def demo_upcoming_predictions():
    """Demo upcoming match predictions."""
    
    print("🚀 EPL PROPHET - UPCOMING MATCHES DEMO")
    print("=" * 60)
    print("Predict future matches like 'Man City vs Liverpool tomorrow'!")
    
    predictor = UpcomingPredictor()
    
    if not predictor.load_system():
        return
    
    # Demo fixtures
    fixtures = [
        ('Man City', 'Liverpool'),
        ('Arsenal', 'Chelsea'),
        ('Man United', 'Tottenham'),
        ('Brighton', 'Newcastle')
    ]
    
    print(f"\n🔮 PREDICTING {len(fixtures)} UPCOMING FIXTURES...")
    
    predictions = []
    
    for home_team, away_team in fixtures:
        prediction = predictor.predict_match(home_team, away_team)
        if prediction:
            predictions.append((home_team, away_team, prediction))
        print("\n" + "="*60 + "\n")
    
    # Summary
    print("📋 UPCOMING FIXTURES SUMMARY:")
    print("=" * 40)
    
    for home, away, pred in predictions:
        emoji = "🔥" if pred['confidence'] > 0.5 else "📊"
        print(f"{emoji} {home} vs {away}: {pred['prediction']} ({pred['confidence']:.1%})")
    
    print(f"\n✨ DEMO COMPLETE!")
    print("🏆 Ready to predict any upcoming EPL fixture!")
    print("💬 Each prediction includes explanations!")

if __name__ == "__main__":
    demo_upcoming_predictions()

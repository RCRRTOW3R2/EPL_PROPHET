#!/usr/bin/env python3
"""
EPL PROPHET - FINAL UPCOMING MATCHES PREDICTOR
==============================================

WORKING VERSION: Predict future EPL matches with our 53.7% champion model!

Example: "Manchester City vs Liverpool tomorrow"
- Uses actual team names from our dataset
- Calculates champion features (logarithmic ratios, EMAs)
- Provides explainable predictions

This is what you wanted - dynamic future predictions!
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

class FinalPredictor:
    """Final upcoming match predictor with correct team names."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        
    def load_system(self):
        """Load champion system."""
        
        print("🏆 Loading Champion Prediction System...")
        
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
            home_matches = df_recent[df_recent['home_team'] == team]
            away_matches = df_recent[df_recent['away_team'] == team]
            
            team_features = {}
            
            # Get latest form from most recent match
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
        """Predict upcoming match with explanation."""
        
        print(f"\n🔮 PREDICTING UPCOMING MATCH")
        print("=" * 60)
        print(f"🏠 HOME: {home_team}")
        print(f"✈️  AWAY: {away_team}")
        print("=" * 60)
        
        if home_team not in self.team_form:
            print(f"   ❌ {home_team} not found in current form data")
            available_teams = list(self.team_form.keys())
            similar_teams = [t for t in available_teams if home_team.lower() in t.lower() or t.lower() in home_team.lower()]
            if similar_teams:
                print(f"   💡 Did you mean: {similar_teams}")
            return None
            
        if away_team not in self.team_form:
            print(f"   ❌ {away_team} not found in current form data")
            available_teams = list(self.team_form.keys())
            similar_teams = [t for t in available_teams if away_team.lower() in t.lower() or t.lower() in away_team.lower()]
            if similar_teams:
                print(f"   💡 Did you mean: {similar_teams}")
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
                # Comparative features (our champions!)
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
        print(f"\n📊 DETAILED PROBABILITIES:")
        class_names = ['Away Win', 'Draw', 'Home Win']
        for i, name in enumerate(class_names):
            bar_length = int(prediction_proba[i] * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {name:>10}: {prediction_proba[i]:.1%} {bar}")
        
        # Explain prediction
        self.explain_prediction(match_features, predicted_result, home_team, away_team)
        
        return {
            'prediction': predicted_result,
            'confidence': confidence,
            'probabilities': dict(zip(class_names, prediction_proba))
        }
    
    def calculate_comparative(self, feature_name, home_form, away_form):
        """Calculate comparative features (our champions!)."""
        
        # Logarithmic ratios (our breakthrough champions!)
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
        
        # Momentum ratios
        if 'momentum_ratio' in feature_name:
            if 'goals' in feature_name:
                home_momentum = home_form.get('goals_momentum', 0)
                away_momentum = away_form.get('goals_momentum', 0)
                return home_momentum / (abs(away_momentum) + 0.01)
        
        return 0
    
    def explain_prediction(self, match_features, predicted_result, home_team, away_team):
        """Explain prediction using our champion features."""
        
        print(f"\n🔥 WHY {predicted_result}? - CHAMPION FEATURES ANALYSIS:")
        print("-" * 55)
        
        # Get feature importance contributions
        importances = self.model.feature_importances_
        contributions = np.array(match_features) * importances
        
        feature_contributions = list(zip(self.feature_names, contributions, match_features))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("📈 TOP SUPPORTING FACTORS:")
        positive_count = 0
        for feat, contrib, value in feature_contributions:
            if contrib > 0 and positive_count < 5:
                print(f"   + {feat}: {contrib:.3f}")
                # Human explanation
                explanation = self.feature_to_human(feat, value, home_team, away_team)
                if explanation:
                    print(f"     💬 {explanation}")
                positive_count += 1
        
        print("\n�� TOP OPPOSING FACTORS:")
        negative_count = 0
        for feat, contrib, value in feature_contributions:
            if contrib < 0 and negative_count < 3:
                print(f"   - {feat}: {contrib:.3f}")
                # Human explanation
                explanation = self.feature_to_human(feat, value, home_team, away_team)
                if explanation:
                    print(f"     💬 {explanation}")
                negative_count += 1
    
    def feature_to_human(self, feature_name, value, home_team, away_team):
        """Convert features to human explanations."""
        
        # Logarithmic ratios (our champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                if value > 0:
                    return f"{home_team} has superior long-term goal-scoring form"
                else:
                    return f"{away_team} has superior long-term goal-scoring form"
            elif 'points' in feature_name and 'long' in feature_name:
                if value > 0:
                    return f"{home_team} has superior long-term points form"
                else:
                    return f"{away_team} has superior long-term points form"
            elif 'goals' in feature_name and 'medium' in feature_name:
                if value > 0:
                    return f"{home_team} has better recent goal-scoring momentum"
                else:
                    return f"{away_team} has better recent goal-scoring momentum"
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                if value > 0:
                    return f"{home_team} has a significant goal-scoring advantage"
                else:
                    return f"{away_team} has a significant goal-scoring advantage"
            elif 'points' in feature_name:
                if value > 0:
                    return f"{home_team} has a significant points form advantage"
                else:
                    return f"{away_team} has a significant points form advantage"
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                if value > 0:
                    return f"{home_team} has overall better goal-scoring form"
                else:
                    return f"{away_team} has overall better goal-scoring form"
        
        # Momentum
        if 'momentum' in feature_name:
            if value > 0:
                return f"Recent momentum favors the prediction"
        
        return None

def predict_specific_match(home_team, away_team):
    """Predict a specific upcoming match."""
    
    print("🚀 EPL PROPHET - UPCOMING MATCH PREDICTOR")
    print("=" * 70)
    print("Predict future EPL matches with our 53.7% champion model!")
    
    predictor = FinalPredictor()
    
    if not predictor.load_system():
        return None
    
    prediction = predictor.predict_match(home_team, away_team)
    
    if prediction:
        print(f"\n✨ PREDICTION COMPLETE!")
        print(f"🏆 Made with our 53.7% champion model")
        print(f"🔥 Powered by logarithmic ratio features")
        print(f"⚡ Based on latest team form and momentum")
    
    return prediction

def demo_big_matches():
    """Demo predictions for big upcoming matches."""
    
    print("🚀 EPL PROPHET - BIG MATCHES DEMO")
    print("=" * 60)
    
    predictor = FinalPredictor()
    if not predictor.load_system():
        return
    
    # Big matches with correct team names
    big_matches = [
        ('Manchester City', 'Liverpool'),
        ('Arsenal', 'Chelsea'),
        ('Manchester United', 'Tottenham'),
        ('Newcastle', 'Aston Villa'),
        ('Brighton', 'West Ham')
    ]
    
    print(f"\n🔮 PREDICTING {len(big_matches)} BIG UPCOMING MATCHES...")
    
    results = []
    
    for home_team, away_team in big_matches:
        prediction = predictor.predict_match(home_team, away_team)
        if prediction:
            results.append((home_team, away_team, prediction))
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("🏆 BIG MATCHES PREDICTIONS SUMMARY:")
    print("=" * 50)
    
    for home, away, pred in results:
        emoji = "🔥" if pred['confidence'] > 0.45 else "📊"
        print(f"{emoji} {home} vs {away}")
        print(f"    Prediction: {pred['prediction']} ({pred['confidence']:.1%} confidence)")
    
    print(f"\n✨ DEMO COMPLETE!")
    print("🏆 All predictions made with our champion model!")
    print("💬 Each includes detailed explanations!")
    print("🔮 Ready to predict ANY upcoming EPL fixture!")

if __name__ == "__main__":
    demo_big_matches()

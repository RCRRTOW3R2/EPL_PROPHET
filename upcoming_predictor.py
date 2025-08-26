#!/usr/bin/env python3
"""
EPL PROPHET - UPCOMING MATCHES PREDICTOR
========================================

Complete system to predict ALL upcoming EPL matches!

Loads fixtures from CSV file and predicts each match with:
- 53.7% champion model accuracy
- Full explanations using champion features
- Confidence levels and probabilities
- Save results to file

This is your complete "Man City vs Liverpool tomorrow" solution!
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class UpcomingMatchesPredictor:
    """Predict all upcoming EPL matches from CSV file."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        self.class_names = ['Away Win', 'Draw', 'Home Win']
        
    def load_champion_system(self):
        """Load our champion prediction system."""
        
        print("🏆 Loading Champion Prediction System...")
        
        try:
            # Load champion model
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion model loaded (53.7% accuracy)")
            
            # Load feature data
            df = pd.read_csv("../outputs/champion_features.csv")
            
            # Get feature names
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} champion features loaded")
            
            # Calculate current team form
            self.calculate_team_form(df)
            print(f"   ✅ Current form for {len(self.team_form)} teams")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading champion system: {e}")
            return False
    
    def calculate_team_form(self, df):
        """Calculate current form for all teams."""
        
        # Get completed matches only
        df_recent = df[df['actual_result'].notna()].copy()
        self.team_form = {}
        
        # Get all teams
        all_teams = set(df_recent['home_team'].tolist() + df_recent['away_team'].tolist())
        
        for team in all_teams:
            # Get team's most recent form data
            home_matches = df_recent[df_recent['home_team'] == team]
            away_matches = df_recent[df_recent['away_team'] == team]
            
            team_features = {}
            
            # Extract latest form from home matches
            if len(home_matches) > 0:
                latest_home = home_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('home_'):
                        base_feature = feature.replace('home_', '')
                        team_features[base_feature] = latest_home[feature] if not pd.isna(latest_home[feature]) else 0
            
            # Extract latest form from away matches
            if len(away_matches) > 0:
                latest_away = away_matches.iloc[-1]
                for feature in self.feature_names:
                    if feature.startswith('away_'):
                        base_feature = feature.replace('away_', '')
                        if base_feature not in team_features:
                            team_features[base_feature] = latest_away[feature] if not pd.isna(latest_away[feature]) else 0
            
            self.team_form[team] = team_features
    
    def create_match_features(self, home_team, away_team):
        """Create feature vector for a match."""
        
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
                # Comparative features (our champions!)
                value = self.calculate_comparative(feature_name, home_form, away_form)
            
            match_features.append(value)
        
        return np.array(match_features)
    
    def calculate_comparative(self, feature_name, home_form, away_form):
        """Calculate comparative features (our breakthrough champions!)."""
        
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
    
    def predict_single_match(self, home_team, away_team, match_date=None, match_time=None):
        """Predict a single upcoming match."""
        
        print(f"\n🔮 PREDICTING: {home_team} vs {away_team}")
        if match_date:
            print(f"📅 Date: {match_date} {match_time}")
        print("=" * 60)
        
        # Create features
        match_features = self.create_match_features(home_team, away_team)
        
        if match_features is None:
            print(f"   ❌ Cannot predict - missing team data")
            return None
        
        # Make prediction
        features_array = match_features.reshape(1, -1)
        prediction_proba = self.model.predict_proba(features_array)[0]
        prediction_class = self.model.predict(features_array)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[prediction_class]
        confidence = max(prediction_proba)
        
        # Display prediction
        print(f"🎯 PREDICTION: {predicted_result}")
        print(f"🎲 CONFIDENCE: {confidence:.1%}")
        
        # Show probability bars
        print(f"\n📊 PROBABILITIES:")
        for i, name in enumerate(self.class_names):
            bar_length = int(prediction_proba[i] * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {name:>10}: {prediction_proba[i]:.1%} {bar}")
        
        # Explain prediction
        self.explain_prediction(match_features, predicted_result, home_team, away_team)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date,
            'time': match_time,
            'prediction': predicted_result,
            'confidence': confidence,
            'away_win_prob': prediction_proba[0],
            'draw_prob': prediction_proba[1], 
            'home_win_prob': prediction_proba[2]
        }
    
    def explain_prediction(self, match_features, predicted_result, home_team, away_team):
        """Explain prediction using champion features."""
        
        print(f"\n🔥 WHY {predicted_result}? - CHAMPION ANALYSIS:")
        print("-" * 50)
        
        # Calculate feature contributions
        importances = self.model.feature_importances_
        contributions = match_features * importances
        
        feature_contributions = list(zip(self.feature_names, contributions, match_features))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("📈 TOP SUPPORTING FACTORS:")
        count = 0
        for feat, contrib, value in feature_contributions:
            if contrib > 0 and count < 3:
                print(f"   + {feat}: {contrib:.3f}")
                explanation = self.feature_to_human(feat, value, home_team, away_team)
                if explanation:
                    print(f"     💬 {explanation}")
                count += 1
        
        print("\n📉 TOP OPPOSING FACTORS:")
        count = 0
        for feat, contrib, value in feature_contributions:
            if contrib < 0 and count < 2:
                print(f"   - {feat}: {contrib:.3f}")
                explanation = self.feature_to_human(feat, value, home_team, away_team)
                if explanation:
                    print(f"     💬 {explanation}")
                count += 1
    
    def feature_to_human(self, feature_name, value, home_team, away_team):
        """Convert features to human explanations."""
        
        # Logarithmic ratios (our breakthrough champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has superior long-term goal-scoring form"
            elif 'points' in feature_name and 'long' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has superior long-term points form"
            elif 'goals' in feature_name and 'medium' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has better recent goal momentum"
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has significant goal-scoring advantage"
            elif 'points' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has significant form advantage"
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if value > 0 else away_team
                return f"{team} has better overall goal form"
        
        # Momentum
        if 'momentum' in feature_name and value != 0:
            return "Recent momentum influences prediction"
        
        return None
    
    def predict_all_upcoming_matches(self, fixtures_file="../outputs/upcoming_fixtures.csv"):
        """Predict all upcoming matches from CSV file."""
        
        print("🚀 EPL PROPHET - PREDICT ALL UPCOMING MATCHES")
        print("=" * 70)
        print("Using our 53.7% champion model on all fixtures!")
        
        # Load upcoming fixtures
        try:
            fixtures_df = pd.read_csv(fixtures_file)
            print(f"\n📁 Loaded {len(fixtures_df)} upcoming fixtures")
        except Exception as e:
            print(f"❌ Error loading fixtures: {e}")
            return None
        
        # Load champion system
        if not self.load_champion_system():
            return None
        
        print(f"\n🔮 PREDICTING {len(fixtures_df)} UPCOMING MATCHES...")
        print("=" * 70)
        
        predictions = []
        
        for idx, fixture in fixtures_df.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            match_date = fixture.get('date', 'TBD')
            match_time = fixture.get('time', 'TBD')
            
            prediction = self.predict_single_match(home_team, away_team, match_date, match_time)
            
            if prediction:
                predictions.append(prediction)
            
            print("\n" + "="*70 + "\n")
        
        # Save predictions to CSV
        if predictions:
            self.save_predictions(predictions)
        
        # Show summary
        self.show_predictions_summary(predictions)
        
        return predictions
    
    def save_predictions(self, predictions, filename="../outputs/upcoming_predictions.csv"):
        """Save predictions to CSV file."""
        
        print(f"💾 Saving predictions to {filename}")
        
        df = pd.DataFrame(predictions)
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Saved {len(df)} predictions")
    
    def show_predictions_summary(self, predictions):
        """Show summary of all predictions."""
        
        if not predictions:
            print("❌ No predictions to show")
            return
        
        print("🏆 UPCOMING MATCHES PREDICTIONS SUMMARY")
        print("=" * 70)
        
        for pred in predictions:
            confidence_emoji = "🔥" if pred['confidence'] > 0.45 else "📊"
            
            print(f"{confidence_emoji} {pred['home_team']} vs {pred['away_team']}")
            print(f"    📅 {pred['date']} {pred['time']}")
            print(f"    🎯 Prediction: {pred['prediction']} ({pred['confidence']:.1%})")
            print(f"    📊 H: {pred['home_win_prob']:.1%} | D: {pred['draw_prob']:.1%} | A: {pred['away_win_prob']:.1%}")
            print()
        
        # Statistics
        home_wins = sum(1 for p in predictions if p['prediction'] == 'Home Win')
        draws = sum(1 for p in predictions if p['prediction'] == 'Draw')
        away_wins = sum(1 for p in predictions if p['prediction'] == 'Away Win')
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        
        print("📈 PREDICTION STATISTICS:")
        print(f"   🏠 Home Wins: {home_wins}/{len(predictions)} ({home_wins/len(predictions):.1%})")
        print(f"   🤝 Draws: {draws}/{len(predictions)} ({draws/len(predictions):.1%})")
        print(f"   ✈️  Away Wins: {away_wins}/{len(predictions)} ({away_wins/len(predictions):.1%})")
        print(f"   🎲 Avg Confidence: {avg_confidence:.1%}")
        
        print(f"\n✨ ALL PREDICTIONS COMPLETE!")
        print("🏆 Made with our 53.7% champion model!")
        print("🔥 Powered by logarithmic ratio features!")
        print("💬 Each prediction includes detailed explanations!")


def predict_all_upcoming():
    """Main function to predict all upcoming matches."""
    
    predictor = UpcomingMatchesPredictor()
    predictions = predictor.predict_all_upcoming_matches()
    
    return predictions

if __name__ == "__main__":
    predict_all_upcoming() 
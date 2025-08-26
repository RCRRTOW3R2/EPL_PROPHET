#!/usr/bin/env python3
"""
PREDICT ALL FUTURE EPL MATCHES
==============================

Predict ALL upcoming EPL matches for the entire season!
- 100 upcoming fixtures across 10 gameweeks
- Choose how many matches to predict
- Get predictions for specific gameweeks
- Full season predictions available!
"""

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

class AllFuturePredictor:
    """Predict all future EPL matches."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.team_form = None
        
    def load_system(self):
        """Load champion system."""
        
        print("🏆 Loading Champion System...")
        
        try:
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion model (53.7% accuracy)")
            
            df = pd.read_csv("../outputs/champion_features.csv")
            
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features")
            
            self.calculate_team_form(df)
            print(f"   ✅ Form for {len(self.team_form)} teams")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
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
    
    def predict_match_quick(self, home_team, away_team):
        """Quick prediction without detailed output."""
        
        features = self.create_features(home_team, away_team)
        
        if features is None:
            return None
        
        features_array = features.reshape(1, -1)
        proba = self.model.predict_proba(features_array)[0]
        pred_class = self.model.predict(features_array)[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[pred_class]
        confidence = max(proba)
        
        return {
            'prediction': predicted_result,
            'confidence': confidence,
            'home_win_prob': proba[2],
            'draw_prob': proba[1],
            'away_win_prob': proba[0]
        }
    
    def predict_all_fixtures(self, max_matches=None, gameweek=None):
        """Predict all or selected future fixtures."""
        
        print("🚀 EPL PROPHET - ALL FUTURE MATCHES PREDICTOR")
        print("=" * 70)
        
        # Load all fixtures
        try:
            fixtures_df = pd.read_csv("../outputs/all_upcoming_fixtures.csv")
            total_fixtures = len(fixtures_df)
            print(f"📁 Loaded {total_fixtures} upcoming fixtures")
        except Exception as e:
            print(f"❌ Error loading fixtures: {e}")
            return None
        
        # Filter by gameweek if specified
        if gameweek:
            fixtures_df = fixtures_df[fixtures_df['gameweek'] == gameweek]
            print(f"🎯 Filtering to Gameweek {gameweek}: {len(fixtures_df)} matches")
        
        # Limit number of matches if specified
        if max_matches:
            fixtures_df = fixtures_df.head(max_matches)
            print(f"🎯 Limiting to first {max_matches} matches")
        
        if len(fixtures_df) == 0:
            print("❌ No fixtures to predict")
            return None
        
        # Load system
        if not self.load_system():
            return None
        
        print(f"\n🔮 PREDICTING {len(fixtures_df)} MATCHES...")
        print("=" * 70)
        
        predictions = []
        successful_predictions = 0
        
        for idx, fixture in fixtures_df.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            match_date = fixture.get('date', 'TBD')
            match_time = fixture.get('time', 'TBD')
            gameweek = fixture.get('gameweek', 'TBD')
            
            prediction = self.predict_match_quick(home_team, away_team)
            
            if prediction:
                prediction.update({
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': match_date,
                    'time': match_time,
                    'gameweek': gameweek
                })
                predictions.append(prediction)
                successful_predictions += 1
                
                # Progress indicator
                if successful_predictions % 10 == 0:
                    print(f"   ✅ Predicted {successful_predictions}/{len(fixtures_df)} matches...")
        
        print(f"\n✅ COMPLETED: {successful_predictions}/{len(fixtures_df)} predictions")
        
        # Save predictions
        if predictions:
            df_pred = pd.DataFrame(predictions)
            filename = f"../outputs/all_future_predictions.csv"
            df_pred.to_csv(filename, index=False)
            print(f"💾 Saved to: {filename}")
        
        # Show summary
        self.show_summary(predictions)
        
        return predictions
    
    def show_summary(self, predictions):
        """Show predictions summary."""
        
        if not predictions:
            return
        
        print(f"\n🏆 ALL FUTURE MATCHES SUMMARY")
        print("=" * 70)
        
        # Group by gameweek
        df = pd.DataFrame(predictions)
        
        if 'gameweek' in df.columns:
            for gameweek in sorted(df['gameweek'].unique()):
                gw_predictions = df[df['gameweek'] == gameweek]
                print(f"\n📅 GAMEWEEK {gameweek} ({len(gw_predictions)} matches):")
                print("-" * 50)
                
                for _, pred in gw_predictions.head(5).iterrows():  # Show first 5 per gameweek
                    emoji = "🔥" if pred['confidence'] > 0.45 else "📊"
                    print(f"{emoji} {pred['home_team']} vs {pred['away_team']}")
                    print(f"    🎯 {pred['prediction']} ({pred['confidence']:.1%})")
                
                if len(gw_predictions) > 5:
                    print(f"    ... and {len(gw_predictions)-5} more matches")
        
        # Overall statistics
        home_wins = sum(1 for p in predictions if p['prediction'] == 'Home Win')
        draws = sum(1 for p in predictions if p['prediction'] == 'Draw')
        away_wins = sum(1 for p in predictions if p['prediction'] == 'Away Win')
        avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
        
        print(f"\n📈 OVERALL STATISTICS ({len(predictions)} matches):")
        print(f"   🏠 Home Wins: {home_wins} ({home_wins/len(predictions):.1%})")
        print(f"   🤝 Draws: {draws} ({draws/len(predictions):.1%})")
        print(f"   ✈️  Away Wins: {away_wins} ({away_wins/len(predictions):.1%})")
        print(f"   🎲 Avg Confidence: {avg_conf:.1%}")
        
        # High confidence predictions
        high_conf = [p for p in predictions if p['confidence'] > 0.5]
        if high_conf:
            print(f"\n🔥 HIGH CONFIDENCE PREDICTIONS ({len(high_conf)} matches):")
            for pred in sorted(high_conf, key=lambda x: x['confidence'], reverse=True)[:5]:
                print(f"   🏆 {pred['home_team']} vs {pred['away_team']}: {pred['prediction']} ({pred['confidence']:.1%})")
        
        print(f"\n✨ ALL FUTURE PREDICTIONS COMPLETE!")
        print("🏆 53.7% accurate champion model")
        print("🔥 Powered by logarithmic ratios")
        print("📊 Ready for analysis!")

def main():
    """Main function with options."""
    
    predictor = AllFuturePredictor()
    
    print("🎯 CHOOSE PREDICTION SCOPE:")
    print("1. Next 10 matches")
    print("2. Next gameweek (10 matches)")
    print("3. Next 3 gameweeks (30 matches)")
    print("4. ALL upcoming matches (100 matches)")
    
    # For demo, let's predict next 20 matches
    print("\n🚀 DEMO: Predicting next 20 matches...")
    predictions = predictor.predict_all_fixtures(max_matches=20)
    
    return predictions

if __name__ == "__main__":
    main()

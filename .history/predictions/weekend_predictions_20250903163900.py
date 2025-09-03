#!/usr/bin/env python3
"""
EPL Prophet - Weekend Predictions with Confidence Analysis
Test our Frankenstein model on real upcoming matches
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
import json

class WeekendPredictor:
    """Predict weekend EPL matches with confidence analysis"""
    
    def __init__(self):
        self.load_frankenstein_model()
        self.load_current_season_data()
        
    def load_frankenstein_model(self):
        """Load our trained Frankenstein model"""
        try:
            model_data = joblib.load('models/frankenstein_ultimate.pkl')
            self.specialists = model_data['specialists']
            self.ensemble = model_data['ensemble'] 
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            print("✅ Frankenstein model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.ensemble = None
    
    def load_current_season_data(self):
        """Load current season data for recent form"""
        try:
            self.current_season = pd.read_csv('2425.csv')
            print(f"✅ Current season data loaded: {len(self.current_season)} matches")
        except Exception as e:
            print(f"⚠️ Using sample data: {e}")
            # Create sample recent data
            self.current_season = self.create_sample_current_data()
    
    def create_sample_current_data(self):
        """Create sample current season data for demonstration"""
        teams = ['Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United', 
                'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham',
                'Crystal Palace', 'Fulham', 'Wolves', 'Everton', 'Brentford',
                'Nottingham Forest', 'Bournemouth', 'Sheffield United', 'Burnley', 'Luton']
        
        # Sample recent matches for current season
        sample_matches = []
        match_id = 1
        
        for i in range(100):  # 100 recent matches
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Simulate realistic results based on team strength
            home_strength = self.get_team_strength(home_team)
            away_strength = self.get_team_strength(away_team)
            
            prob_home = 0.46 + (home_strength - away_strength) * 0.1 + 0.1  # Home advantage
            prob_draw = 0.25
            
            result_rand = np.random.random()
            if result_rand < prob_home:
                ftr = 'H'
                home_goals = np.random.poisson(1.8)
                away_goals = np.random.poisson(1.2)
            elif result_rand < prob_home + prob_draw:
                ftr = 'D'
                home_goals = np.random.poisson(1.4)
                away_goals = home_goals  # Force draw
            else:
                ftr = 'A'
                home_goals = np.random.poisson(1.2)
                away_goals = np.random.poisson(1.8)
            
            sample_matches.append({
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'FTR': ftr,
                'Date': f"2024-12-{(i % 30) + 1:02d}"
            })
        
        return pd.DataFrame(sample_matches)
    
    def get_team_strength(self, team):
        """Get relative team strength (0-1 scale)"""
        strength_map = {
            'Manchester City': 0.95, 'Arsenal': 0.90, 'Liverpool': 0.88,
            'Chelsea': 0.80, 'Manchester United': 0.75, 'Tottenham': 0.72,
            'Newcastle': 0.68, 'Brighton': 0.65, 'Aston Villa': 0.62,
            'West Ham': 0.58, 'Crystal Palace': 0.55, 'Fulham': 0.52,
            'Wolves': 0.50, 'Everton': 0.48, 'Brentford': 0.46,
            'Nottingham Forest': 0.44, 'Bournemouth': 0.42, 'Sheffield United': 0.38,
            'Burnley': 0.35, 'Luton': 0.30
        }
        return strength_map.get(team, 0.50)
    
    def get_upcoming_weekend_fixtures(self):
        """Get this weekend's EPL fixtures"""
        # Sample upcoming weekend fixtures
        weekend_fixtures = [
            {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'kickoff': '2024-12-21 17:30'},
            {'home_team': 'Liverpool', 'away_team': 'Tottenham', 'kickoff': '2024-12-22 16:30'},
            {'home_team': 'Manchester City', 'away_team': 'Manchester United', 'kickoff': '2024-12-22 16:30'},
            {'home_team': 'Brighton', 'away_team': 'Brentford', 'kickoff': '2024-12-21 15:00'},
            {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'kickoff': '2024-12-21 15:00'},
            {'home_team': 'West Ham', 'away_team': 'Wolves', 'kickoff': '2024-12-22 14:00'},
            {'home_team': 'Crystal Palace', 'away_team': 'Fulham', 'kickoff': '2024-12-22 14:00'},
            {'home_team': 'Everton', 'away_team': 'Nottingham Forest', 'kickoff': '2024-12-21 15:00'}
        ]
        
        print(f"🗓️ Found {len(weekend_fixtures)} weekend fixtures")
        return weekend_fixtures
    
    def get_recent_matches_for_team(self, team, n=6):
        """Get recent matches for a team from current season"""
        team_matches = []
        
        for _, match in self.current_season.iterrows():
            if match['HomeTeam'] == team:
                result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
                goals_for = match['FTHG']
                goals_against = match['FTAG']
            elif match['AwayTeam'] == team:
                result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
                goals_for = match['FTAG']
                goals_against = match['FTHG']
            else:
                continue
            
            team_matches.append({
                'result': result,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_margin': goals_for - goals_against
            })
        
        return team_matches[-n:] if len(team_matches) >= n else team_matches
    
    def create_match_features(self, home_team, away_team):
        """Create features for a specific match"""
        # Get recent form
        home_recent = self.get_recent_matches_for_team(home_team, 6)
        away_recent = self.get_recent_matches_for_team(away_team, 6)
        
        if len(home_recent) < 3 or len(away_recent) < 3:
            print(f"⚠️ Insufficient data for {home_team} vs {away_team}")
            return None
        
        features = {}
        
        # Basic features
        home_goals_avg = np.mean([m['goals_for'] for m in home_recent])
        away_goals_avg = np.mean([m['goals_for'] for m in away_recent])
        home_conceded_avg = np.mean([m['goals_against'] for m in home_recent])
        away_conceded_avg = np.mean([m['goals_against'] for m in away_recent])
        
        home_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in home_recent) / len(home_recent)
        away_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in away_recent) / len(away_recent)
        
        features.update({
            'home_goals_avg': home_goals_avg,
            'away_goals_avg': away_goals_avg,
            'home_conceded_avg': home_conceded_avg,
            'away_conceded_avg': away_conceded_avg,
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'form_difference': home_ppg - away_ppg
        })
        
        # Psychological features
        home_last_margin = home_recent[0]['goal_margin'] if home_recent else 0
        away_last_margin = away_recent[0]['goal_margin'] if away_recent else 0
        
        features.update({
            'home_post_big_loss': 1 if home_last_margin <= -3 else 0,
            'away_post_big_loss': 1 if away_last_margin <= -3 else 0,
            'home_bounce_back_likely': 1 if (home_last_margin <= -3 and self.is_bounce_back_team(home_recent)) else 0,
            'away_bounce_back_likely': 1 if (away_last_margin <= -3 and self.is_bounce_back_team(away_recent)) else 0,
            'home_spiral_risk': 1 if (home_last_margin <= -3 and self.is_spiral_team(home_recent)) else 0,
            'away_spiral_risk': 1 if (away_last_margin <= -3 and self.is_spiral_team(away_recent)) else 0,
            'home_post_big_win': 1 if home_last_margin >= 3 else 0,
            'away_post_big_win': 1 if away_last_margin >= 3 else 0,
            'home_momentum_likely': 1 if (home_last_margin >= 3 and self.is_momentum_team(home_recent)) else 0,
            'away_momentum_likely': 1 if (away_last_margin >= 3 and self.is_momentum_team(away_recent)) else 0,
            'home_complacency_risk': 1 if (home_last_margin >= 3 and self.is_complacency_team(home_recent)) else 0,
            'away_complacency_risk': 1 if (away_last_margin >= 3 and self.is_complacency_team(away_recent)) else 0,
            'home_psych_momentum': self.calculate_psychological_momentum(home_recent),
            'away_psych_momentum': self.calculate_psychological_momentum(away_recent)
        })
        
        # Momentum features
        home_streak = self.calculate_streak_length(home_recent)
        away_streak = self.calculate_streak_length(away_recent)
        
        features.update({
            'home_win_streak': home_streak if home_recent[0]['result'] == 'W' else 0,
            'away_win_streak': away_streak if away_recent[0]['result'] == 'W' else 0,
            'home_loss_streak': home_streak if home_recent[0]['result'] == 'L' else 0,
            'away_loss_streak': away_streak if away_recent[0]['result'] == 'L' else 0
        })
        
        # Opponent psychology
        big6_teams = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham'}
        features.update({
            'home_vs_big6': 1 if away_team in big6_teams else 0,
            'away_vs_big6': 1 if home_team in big6_teams else 0,
            'big6_clash': 1 if (home_team in big6_teams and away_team in big6_teams) else 0
        })
        
        # Logarithmic ratios
        features.update({
            'log_goals_ratio': np.log((home_goals_avg + 1) / (away_goals_avg + 1)),
            'log_conceded_ratio': np.log((away_conceded_avg + 1) / (home_conceded_avg + 1))
        })
        
        return features
    
    def is_bounce_back_team(self, recent_matches):
        """Check if team bounces back from losses"""
        return len([m for m in recent_matches[:3] if m['result'] == 'W']) >= 2
    
    def is_spiral_team(self, recent_matches):
        """Check if team spirals after losses"""
        return len([m for m in recent_matches[:3] if m['result'] == 'L']) >= 2
    
    def is_momentum_team(self, recent_matches):
        """Check if team maintains momentum"""
        win_streak = 0
        for match in recent_matches:
            if match['result'] == 'W':
                win_streak += 1
            else:
                break
        return win_streak >= 3
    
    def is_complacency_team(self, recent_matches):
        """Check if team gets complacent"""
        if len(recent_matches) < 4:
            return False
        inconsistent = 0
        for i in range(len(recent_matches) - 1):
            if recent_matches[i+1]['result'] == 'W' and recent_matches[i]['result'] == 'L':
                inconsistent += 1
        return inconsistent >= 2
    
    def calculate_psychological_momentum(self, recent_matches):
        """Calculate psychological momentum (1-10)"""
        if not recent_matches:
            return 5.0
        
        ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in recent_matches) / len(recent_matches)
        base_score = (ppg / 3.0) * 5.0 + 2.5
        
        last_margin = recent_matches[0]['goal_margin']
        if last_margin >= 3:
            base_score += 1.5
        elif last_margin <= -3:
            base_score -= 1.5
        
        return max(1.0, min(10.0, base_score))
    
    def calculate_streak_length(self, recent_matches):
        """Calculate current streak length"""
        if not recent_matches:
            return 0
        
        current_result = recent_matches[0]['result']
        streak = 1
        for match in recent_matches[1:]:
            if match['result'] == current_result:
                streak += 1
            else:
                break
        return streak
    
    def predict_match_with_confidence(self, home_team, away_team):
        """Predict match with detailed confidence analysis"""
        if not self.ensemble:
            return None
        
        print(f"\n🔮 PREDICTING: {home_team} vs {away_team}")
        
        # Create features
        features = self.create_match_features(home_team, away_team)
        if not features:
            return None
        
        # Prepare feature vector (ensure all expected features are present)
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Get predictions from each specialist
        specialist_predictions = {}
        
        # Linear specialist (needs scaled data)
        feature_scaled = self.scaler.transform(feature_array)
        if 'linear' in self.specialists:
            linear_pred = self.specialists['linear'].predict_proba(feature_scaled)[0]
            specialist_predictions['linear'] = {
                'home_win': linear_pred[2] if len(linear_pred) > 2 else 0,
                'draw': linear_pred[1] if len(linear_pred) > 1 else 0,
                'away_win': linear_pred[0]
            }
        
        # Psychology specialist (uses original data)
        if 'psychology' in self.specialists:
            psych_pred = self.specialists['psychology'].predict_proba(feature_array)[0]
            specialist_predictions['psychology'] = {
                'home_win': psych_pred[2] if len(psych_pred) > 2 else 0,
                'draw': psych_pred[1] if len(psych_pred) > 1 else 0,
                'away_win': psych_pred[0]
            }
        
        # Boundary specialist (needs scaled data)
        if 'boundary' in self.specialists:
            boundary_pred = self.specialists['boundary'].predict_proba(feature_scaled)[0]
            specialist_predictions['boundary'] = {
                'home_win': boundary_pred[2] if len(boundary_pred) > 2 else 0,
                'draw': boundary_pred[1] if len(boundary_pred) > 1 else 0,
                'away_win': boundary_pred[0]
            }
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(feature_array)[0]
        
        # Calculate confidence
        confidence = self.calculate_prediction_confidence(
            specialist_predictions, ensemble_pred, features
        )
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'ensemble_prediction': {
                'home_win': round(ensemble_pred[2] * 100, 1) if len(ensemble_pred) > 2 else 0,
                'draw': round(ensemble_pred[1] * 100, 1) if len(ensemble_pred) > 1 else 0,
                'away_win': round(ensemble_pred[0] * 100, 1)
            },
            'specialist_predictions': specialist_predictions,
            'confidence': confidence,
            'key_factors': self.get_key_factors(features),
            'psychological_insights': self.get_psychological_insights(features, home_team, away_team)
        }
    
    def calculate_prediction_confidence(self, specialist_preds, ensemble_pred, features):
        """Calculate overall prediction confidence"""
        
        # Agreement between specialists
        if specialist_preds:
            pred_values = []
            for specialist, pred in specialist_preds.items():
                pred_values.append([pred['away_win'], pred['draw'], pred['home_win']])
            
            if len(pred_values) >= 2:
                agreement = 1.0 - np.std(pred_values, axis=0).mean()
            else:
                agreement = 0.7
        else:
            agreement = 0.5
        
        # Prediction certainty (how far from 33.3% each outcome is)
        if len(ensemble_pred) >= 3:
            certainty = max(ensemble_pred) - (1.0 / len(ensemble_pred))
        else:
            certainty = 0.5
        
        # Feature coverage (how many psychological features are active)
        psych_features = ['home_post_big_loss', 'away_post_big_loss', 'home_post_big_win', 
                         'away_post_big_win', 'home_bounce_back_likely', 'away_bounce_back_likely']
        active_features = sum(1 for f in psych_features if features.get(f, 0) > 0)
        coverage = min(1.0, active_features / 3.0)
        
        # Composite confidence
        composite_confidence = (agreement * 0.4) + (certainty * 0.4) + (coverage * 0.2)
        
        confidence_level = "Very High" if composite_confidence > 0.8 else \
                          "High" if composite_confidence > 0.6 else \
                          "Medium" if composite_confidence > 0.4 else "Low"
        
        return {
            'level': confidence_level,
            'score': round(composite_confidence * 100, 1),
            'agreement': round(agreement * 100, 1),
            'certainty': round(certainty * 100, 1),
            'coverage': round(coverage * 100, 1)
        }
    
    def get_key_factors(self, features):
        """Get key factors influencing prediction"""
        factors = []
        
        if features.get('form_difference', 0) > 0.5:
            factors.append(f"Home team form advantage ({features['form_difference']:.2f} PPG)")
        elif features.get('form_difference', 0) < -0.5:
            factors.append(f"Away team form advantage ({abs(features['form_difference']):.2f} PPG)")
        
        if features.get('big6_clash', 0):
            factors.append("Big 6 clash - high intensity expected")
        
        if features.get('home_win_streak', 0) >= 3:
            factors.append(f"Home team on {features['home_win_streak']}-game win streak")
        
        if features.get('away_win_streak', 0) >= 3:
            factors.append(f"Away team on {features['away_win_streak']}-game win streak")
        
        return factors[:3]  # Top 3 factors
    
    def get_psychological_insights(self, features, home_team, away_team):
        """Get psychological insights"""
        insights = []
        
        if features.get('home_bounce_back_likely', 0):
            insights.append(f"{home_team} likely to bounce back from recent big loss")
        
        if features.get('away_spiral_risk', 0):
            insights.append(f"{away_team} at risk of spiraling after recent defeat")
        
        if features.get('home_momentum_likely', 0):
            insights.append(f"{home_team} riding momentum from recent big win")
        
        if features.get('away_complacency_risk', 0):
            insights.append(f"{away_team} at risk of complacency after good result")
        
        home_momentum = features.get('home_psych_momentum', 5)
        away_momentum = features.get('away_psych_momentum', 5)
        
        if abs(home_momentum - away_momentum) > 2:
            if home_momentum > away_momentum:
                insights.append(f"{home_team} has significant psychological advantage ({home_momentum:.1f} vs {away_momentum:.1f})")
            else:
                insights.append(f"{away_team} has significant psychological advantage ({away_momentum:.1f} vs {home_momentum:.1f})")
        
        return insights
    
    def predict_weekend_fixtures(self):
        """Predict all weekend fixtures"""
        print("🏆 EPL PROPHET - WEEKEND PREDICTIONS")
        print("="*50)
        
        fixtures = self.get_upcoming_weekend_fixtures()
        predictions = []
        
        for fixture in fixtures:
            prediction = self.predict_match_with_confidence(
                fixture['home_team'], 
                fixture['away_team']
            )
            
            if prediction:
                predictions.append({**fixture, **prediction})
                
                # Display prediction
                print(f"\n🎯 {prediction['home_team']} vs {prediction['away_team']}")
                print(f"   📊 Prediction: H {prediction['ensemble_prediction']['home_win']}% | "
                      f"D {prediction['ensemble_prediction']['draw']}% | "
                      f"A {prediction['ensemble_prediction']['away_win']}%")
                print(f"   🎯 Confidence: {prediction['confidence']['level']} ({prediction['confidence']['score']}%)")
                
                if prediction['key_factors']:
                    print(f"   🔑 Key factors:")
                    for factor in prediction['key_factors']:
                        print(f"      • {factor}")
                
                if prediction['psychological_insights']:
                    print(f"   🧠 Psychology:")
                    for insight in prediction['psychological_insights']:
                        print(f"      • {insight}")
        
        return predictions

def main():
    """Main prediction function"""
    predictor = WeekendPredictor()
    predictions = predictor.predict_weekend_fixtures()
    
    print(f"\n🏆 WEEKEND PREDICTION SUMMARY:")
    print(f"   🎯 {len(predictions)} matches predicted")
    
    high_confidence = [p for p in predictions if p['confidence']['level'] in ['High', 'Very High']]
    print(f"   📈 {len(high_confidence)} high-confidence predictions")
    
    big_games = [p for p in predictions if p.get('big6_clash', 0)]
    print(f"   ⚡ {len(big_games)} Big 6 clashes")
    
    print(f"\n🧟‍♂️ Frankenstein ensemble active with psychological insights!")

if __name__ == "__main__":
    main() 
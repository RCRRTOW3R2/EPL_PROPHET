#!/usr/bin/env python3
"""
EPL Prophet - Score & Spread Prediction Model
Extends the 53.7% win/loss model to predict actual scores and goal spreads
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson
import joblib
import json

class ScorePredictionModel:
    """Predict exact scores and goal spreads using EPL Prophet features."""
    
    def __init__(self):
        self.poisson_model_home = None
        self.poisson_model_away = None
        self.neural_net_model = None
        self.spread_model = None
        self.feature_columns = None
        
    def load_data_and_features(self):
        """Load the champion features dataset."""
        try:
            # Load the champion features that achieved 53.7% accuracy
            df = pd.read_csv('ANALYSIS1.0/outputs/champion_features.csv')
            print(f"✅ Loaded {len(df)} matches with champion features")
            
            # Load team data for normalization
            team_data = self.load_team_data()
            
            return df, team_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def load_team_data(self):
        """Load current team Elo and xG data."""
        team_data = {
            'Liverpool': {'elo': 1626, 'xg': 2.1}, 'Manchester City': {'elo': 1623, 'xg': 2.3},
            'Arsenal': {'elo': 1621, 'xg': 2.0}, 'Chelsea': {'elo': 1577, 'xg': 1.8},
            'Aston Villa': {'elo': 1571, 'xg': 1.7}, 'Newcastle': {'elo': 1564, 'xg': 1.6},
            'Crystal Palace': {'elo': 1556, 'xg': 1.4}, 'Brighton': {'elo': 1551, 'xg': 1.5},
            'Nottingham Forest': {'elo': 1544, 'xg': 1.3}, 'Brentford': {'elo': 1524, 'xg': 1.4},
            'Everton': {'elo': 1523, 'xg': 1.2}, 'Fulham': {'elo': 1520, 'xg': 1.5},
            'Bournemouth': {'elo': 1518, 'xg': 1.3}, 'West Ham': {'elo': 1490, 'xg': 1.2},
            'Wolves': {'elo': 1478, 'xg': 1.1}, 'Tottenham': {'elo': 1470, 'xg': 1.6},
            'Manchester United': {'elo': 1465, 'xg': 1.4}, 'Ipswich': {'elo': 1450, 'xg': 1.0},
            'Leicester': {'elo': 1445, 'xg': 1.1}, 'Southampton': {'elo': 1440, 'xg': 0.9}
        }
        return team_data
    
    def prepare_score_features(self, df):
        """Prepare features specifically for score prediction."""
        print("🔧 Engineering score prediction features...")
        
        # Select key features for score prediction
        score_features = [
            # Team strength
            'home_elo_rating', 'away_elo_rating', 'elo_difference',
            # Attacking metrics
            'home_xg_last_5', 'away_xg_last_5', 'xg_difference',
            'home_goals_scored_last_5', 'away_goals_scored_last_5',
            'home_shots_last_5', 'away_shots_last_5',
            # Defensive metrics  
            'home_goals_against_last_5', 'away_goals_against_last_5',
            'home_clean_sheets_last_5', 'away_clean_sheets_last_5',
            # Form and momentum
            'home_form_points', 'away_form_points', 'form_difference',
            'home_win_streak', 'away_win_streak',
            # Home advantage
            'home_advantage', 'home_goals_avg', 'away_goals_avg',
            # Head to head
            'h2h_home_wins', 'h2h_away_wins', 'h2h_avg_goals',
            # Context
            'rest_days_difference', 'fixture_congestion_home', 'fixture_congestion_away'
        ]
        
        # Filter available features
        available_features = [col for col in score_features if col in df.columns]
        
        if len(available_features) < 10:
            print("⚠️  Using all numerical features as score features are limited")
            # Use all numerical features if specific ones aren't available
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target variables
            available_features = [col for col in available_features if not col.startswith(('FTH', 'FTA', 'result'))]
        
        print(f"📊 Using {len(available_features)} features for score prediction")
        self.feature_columns = available_features
        
        return df[available_features].fillna(0)
    
    def train_poisson_models(self, X, y_home, y_away):
        """Train Poisson-based models for goal prediction."""
        print("🎯 Training Poisson-based score models...")
        
        # Train separate models for home and away goals
        self.poisson_model_home = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )
        self.poisson_model_away = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )
        
        # Fit models
        self.poisson_model_home.fit(X, y_home)
        self.poisson_model_away.fit(X, y_away)
        
        # Predict on training data for evaluation
        pred_home = self.poisson_model_home.predict(X)
        pred_away = self.poisson_model_away.predict(X)
        
        # Calculate metrics
        mae_home = mean_absolute_error(y_home, pred_home)
        mae_away = mean_absolute_error(y_away, pred_away)
        
        print(f"✅ Home goals MAE: {mae_home:.3f}")
        print(f"✅ Away goals MAE: {mae_away:.3f}")
        
        return pred_home, pred_away
    
    def train_neural_network(self, X, y_combined):
        """Train neural network for multi-output goal prediction."""
        print("🧠 Training neural network score model...")
        
        self.neural_net_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        # Fit model
        self.neural_net_model.fit(X, y_combined)
        
        # Predict and evaluate
        pred_combined = self.neural_net_model.predict(X)
        
        mae_home = mean_absolute_error(y_combined[:, 0], pred_combined[:, 0])
        mae_away = mean_absolute_error(y_combined[:, 1], pred_combined[:, 1])
        
        print(f"✅ Neural Net Home goals MAE: {mae_home:.3f}")
        print(f"✅ Neural Net Away goals MAE: {mae_away:.3f}")
        
        return pred_combined
    
    def train_spread_model(self, X, y_spread):
        """Train model for goal spread prediction."""
        print("📈 Training spread prediction model...")
        
        self.spread_model = RandomForestRegressor(
            n_estimators=150, max_depth=8, random_state=42
        )
        
        self.spread_model.fit(X, y_spread)
        
        pred_spread = self.spread_model.predict(X)
        mae_spread = mean_absolute_error(y_spread, pred_spread)
        
        print(f"✅ Spread MAE: {mae_spread:.3f}")
        
        return pred_spread
    
    def predict_match_score(self, match_features):
        """Predict exact score for a single match."""
        if not all([self.poisson_model_home, self.poisson_model_away, self.neural_net_model]):
            raise ValueError("Models not trained yet!")
        
        # Ensure features are in correct format
        if isinstance(match_features, dict):
            # Convert dict to DataFrame with correct feature order
            features_df = pd.DataFrame([match_features])
            # Reorder columns to match training
            features_df = features_df.reindex(columns=self.feature_columns, fill_value=0)
            match_features = features_df.values
        
        # Poisson model predictions
        home_goals_pred = max(0, self.poisson_model_home.predict(match_features)[0])
        away_goals_pred = max(0, self.poisson_model_away.predict(match_features)[0])
        
        # Neural network prediction
        nn_pred = self.neural_net_model.predict(match_features)[0]
        nn_home_goals = max(0, nn_pred[0])
        nn_away_goals = max(0, nn_pred[1])
        
        # Spread prediction
        spread_pred = self.spread_model.predict(match_features)[0]
        
        # Generate most likely scorelines using Poisson
        def generate_scorelines(lambda_home, lambda_away, max_goals=5):
            scorelines = []
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                    scorelines.append({
                        'score': f"{h}-{a}",
                        'home_goals': h,
                        'away_goals': a,
                        'probability': prob
                    })
            return sorted(scorelines, key=lambda x: x['probability'], reverse=True)
        
        # Average the two approaches
        avg_home = (home_goals_pred + nn_home_goals) / 2
        avg_away = (away_goals_pred + nn_away_goals) / 2
        
        most_likely_scores = generate_scorelines(avg_home, avg_away)
        
        return {
            'poisson_prediction': {
                'home_goals': round(home_goals_pred, 1),
                'away_goals': round(away_goals_pred, 1),
                'most_likely_score': f"{round(home_goals_pred)}-{round(away_goals_pred)}"
            },
            'neural_net_prediction': {
                'home_goals': round(nn_home_goals, 1),
                'away_goals': round(nn_away_goals, 1),
                'most_likely_score': f"{round(nn_home_goals)}-{round(nn_away_goals)}"
            },
            'ensemble_prediction': {
                'home_goals': round(avg_home, 1),
                'away_goals': round(avg_away, 1),
                'most_likely_score': f"{round(avg_home)}-{round(avg_away)}",
                'spread': round(spread_pred, 1)
            },
            'top_scorelines': most_likely_scores[:5]
        }
    
    def train_all_models(self):
        """Train all score prediction models."""
        print("🚀 EPL PROPHET - SCORE PREDICTION TRAINING")
        print("=" * 50)
        
        # Load data
        df, team_data = self.load_data_and_features()
        if df is None:
            return False
        
        # Check for goal columns
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
            print("❌ Goal data (FTHG, FTAG) not found in dataset")
            return False
        
        # Prepare features
        X = self.prepare_score_features(df)
        
        # Prepare targets
        y_home = df['FTHG'].values
        y_away = df['FTAG'].values
        y_combined = np.column_stack([y_home, y_away])
        y_spread = y_home - y_away  # Goal difference
        
        print(f"📊 Training on {len(X)} matches")
        print(f"📊 Average goals: Home {y_home.mean():.2f}, Away {y_away.mean():.2f}")
        
        # Train models
        self.train_poisson_models(X, y_home, y_away)
        self.train_neural_network(X, y_combined)
        self.train_spread_model(X, y_spread)
        
        # Save models
        joblib.dump(self.poisson_model_home, 'score_model_home_goals.joblib')
        joblib.dump(self.poisson_model_away, 'score_model_away_goals.joblib')
        joblib.dump(self.neural_net_model, 'score_model_neural_net.joblib')
        joblib.dump(self.spread_model, 'score_model_spread.joblib')
        
        # Save feature columns
        with open('score_model_features.json', 'w') as f:
            json.dump(self.feature_columns, f)
        
        print("\n💾 All score prediction models saved!")
        print("📈 Ready to predict exact scores and spreads!")
        
        return True

def main():
    """Main training function."""
    model = ScorePredictionModel()
    success = model.train_all_models()
    
    if success:
        # Test prediction
        print("\n🔮 Testing score prediction...")
        sample_features = {col: 0.5 for col in model.feature_columns}
        
        try:
            prediction = model.predict_match_score(sample_features)
            print("✅ Sample prediction successful:")
            print(f"   Ensemble: {prediction['ensemble_prediction']['most_likely_score']}")
            print(f"   Spread: {prediction['ensemble_prediction']['spread']}")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")

if __name__ == "__main__":
    main() 
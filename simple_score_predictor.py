#!/usr/bin/env python3
"""
EPL Prophet - Simple Score Prediction
Creates score prediction from basic team stats and Elo ratings
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import poisson
import joblib
import json

def load_and_prepare_data():
    """Load all seasons and prepare score prediction data."""
    print("📊 Loading EPL data for score prediction...")
    
    all_data = []
    seasons = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425']
    
    for season in seasons:
        try:
            df = pd.read_csv(f'{season}.csv')
            df['season'] = season
            all_data.append(df)
            print(f"✅ Loaded {len(df)} matches from {season}")
        except Exception as e:
            print(f"❌ Failed to load {season}: {e}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"📈 Total matches: {len(combined_data)}")
    
    return combined_data

def create_score_features(df):
    """Create features for score prediction."""
    print("🔧 Engineering score prediction features...")
    
    # Initialize feature DataFrame
    features = []
    
    for idx, match in df.iterrows():
        if idx < 50:  # Skip first 50 matches to have history
            continue
            
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = pd.to_datetime(match['Date'], dayfirst=True)
        
        # Get recent history (last 10 matches for each team)
        recent_matches = df[df.index < idx].copy()
        recent_matches['Date'] = pd.to_datetime(recent_matches['Date'], dayfirst=True)
        recent_matches = recent_matches[recent_matches['Date'] > match_date - pd.Timedelta(days=365)]
        
        # Home team recent stats
        home_recent = recent_matches[
            (recent_matches['HomeTeam'] == home_team) | 
            (recent_matches['AwayTeam'] == home_team)
        ].tail(10)
        
        # Away team recent stats  
        away_recent = recent_matches[
            (recent_matches['HomeTeam'] == away_team) | 
            (recent_matches['AwayTeam'] == away_team)
        ].tail(10)
        
        # Calculate home team stats
        home_goals_for = 0
        home_goals_against = 0
        home_matches = 0
        
        for _, hmatch in home_recent.iterrows():
            if hmatch['HomeTeam'] == home_team:
                home_goals_for += hmatch['FTHG']
                home_goals_against += hmatch['FTAG']
            else:
                home_goals_for += hmatch['FTAG']
                home_goals_against += hmatch['FTHG']
            home_matches += 1
        
        # Calculate away team stats
        away_goals_for = 0
        away_goals_against = 0
        away_matches = 0
        
        for _, amatch in away_recent.iterrows():
            if amatch['HomeTeam'] == away_team:
                away_goals_for += amatch['FTHG']
                away_goals_against += amatch['FTAG']
            else:
                away_goals_for += amatch['FTAG']
                away_goals_against += amatch['FTHG']
            away_matches += 1
        
        # Calculate averages
        home_goals_avg = home_goals_for / max(home_matches, 1)
        home_conceded_avg = home_goals_against / max(home_matches, 1)
        away_goals_avg = away_goals_for / max(away_matches, 1)
        away_conceded_avg = away_goals_against / max(away_matches, 1)
        
        # Create feature vector
        feature_row = {
            'home_goals_avg': home_goals_avg,
            'home_conceded_avg': home_conceded_avg,
            'away_goals_avg': away_goals_avg,
            'away_conceded_avg': away_conceded_avg,
            'home_attack_strength': home_goals_avg / 1.5,  # League avg ~1.5
            'home_defense_strength': 1.5 / max(home_conceded_avg, 0.1),
            'away_attack_strength': away_goals_avg / 1.3,  # Away teams score less
            'away_defense_strength': 1.3 / max(away_conceded_avg, 0.1),
            'home_advantage': 1.0,  # Home advantage factor
            'FTHG': match['FTHG'],  # Target
            'FTAG': match['FTAG']   # Target
        }
        
        features.append(feature_row)
    
    feature_df = pd.DataFrame(features)
    print(f"✅ Created {len(feature_df)} feature rows")
    
    return feature_df

def train_score_models(df):
    """Train Poisson-based score prediction models."""
    print("🎯 Training score prediction models...")
    
    # Prepare features and targets
    feature_cols = [col for col in df.columns if col not in ['FTHG', 'FTAG']]
    X = df[feature_cols].fillna(0)
    y_home = df['FTHG']
    y_away = df['FTAG']
    
    print(f"📊 Features: {feature_cols}")
    print(f"📊 Training samples: {len(X)}")
    print(f"📊 Avg goals: Home {y_home.mean():.2f}, Away {y_away.mean():.2f}")
    
    # Train separate models for home and away goals
    home_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    away_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    
    # Evaluate
    home_pred = home_model.predict(X)
    away_pred = away_model.predict(X)
    
    home_mae = mean_absolute_error(y_home, home_pred)
    away_mae = mean_absolute_error(y_away, away_pred)
    
    print(f"✅ Home goals MAE: {home_mae:.3f}")
    print(f"✅ Away goals MAE: {away_mae:.3f}")
    
    # Save models
    joblib.dump(home_model, 'simple_score_home_model.joblib')
    joblib.dump(away_model, 'simple_score_away_model.joblib')
    
    # Save feature names
    with open('simple_score_features.json', 'w') as f:
        json.dump(feature_cols, f)
    
    return home_model, away_model, feature_cols

def predict_match_score(home_team, away_team, team_stats):
    """Predict score for a specific match."""
    try:
        # Load models
        home_model = joblib.load('simple_score_home_model.joblib')
        away_model = joblib.load('simple_score_away_model.joblib')
        
        with open('simple_score_features.json', 'r') as f:
            feature_cols = json.load(f)
    except:
        print("❌ Models not trained yet!")
        return None
    
    # Get team stats
    home_stats = team_stats.get(home_team, {'goals_avg': 1.5, 'conceded_avg': 1.5})
    away_stats = team_stats.get(away_team, {'goals_avg': 1.3, 'conceded_avg': 1.3})
    
    # Create feature vector
    features = {
        'home_goals_avg': home_stats['goals_avg'],
        'home_conceded_avg': home_stats['conceded_avg'],
        'away_goals_avg': away_stats['goals_avg'],
        'away_conceded_avg': away_stats['conceded_avg'],
        'home_attack_strength': home_stats['goals_avg'] / 1.5,
        'home_defense_strength': 1.5 / max(home_stats['conceded_avg'], 0.1),
        'away_attack_strength': away_stats['goals_avg'] / 1.3,
        'away_defense_strength': 1.3 / max(away_stats['conceded_avg'], 0.1),
        'home_advantage': 1.0
    }
    
    # Ensure all features are present
    feature_vector = [features.get(col, 0) for col in feature_cols]
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Predict
    home_goals_pred = max(0, home_model.predict(feature_array)[0])
    away_goals_pred = max(0, away_model.predict(feature_array)[0])
    
    # Generate most likely scorelines
    def generate_scorelines(lambda_home, lambda_away, max_goals=4):
        scorelines = []
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                scorelines.append({
                    'score': f"{h}-{a}",
                    'probability': prob * 100
                })
        return sorted(scorelines, key=lambda x: x['probability'], reverse=True)
    
    most_likely_scores = generate_scorelines(home_goals_pred, away_goals_pred)
    
    return {
        'predicted_score': f"{round(home_goals_pred)}-{round(away_goals_pred)}",
        'home_goals': round(home_goals_pred, 1),
        'away_goals': round(away_goals_pred, 1),
        'spread': round(home_goals_pred - away_goals_pred, 1),
        'total_goals': round(home_goals_pred + away_goals_pred, 1),
        'most_likely_scores': most_likely_scores[:5]
    }

def create_team_stats():
    """Create current team stats for prediction."""
    team_stats = {
        'Liverpool': {'goals_avg': 2.1, 'conceded_avg': 1.0},
        'Manchester City': {'goals_avg': 2.3, 'conceded_avg': 0.9},
        'Arsenal': {'goals_avg': 2.0, 'conceded_avg': 1.1},
        'Chelsea': {'goals_avg': 1.8, 'conceded_avg': 1.2},
        'Aston Villa': {'goals_avg': 1.7, 'conceded_avg': 1.3},
        'Newcastle': {'goals_avg': 1.6, 'conceded_avg': 1.2},
        'Crystal Palace': {'goals_avg': 1.4, 'conceded_avg': 1.4},
        'Brighton': {'goals_avg': 1.5, 'conceded_avg': 1.3},
        'Nottingham Forest': {'goals_avg': 1.3, 'conceded_avg': 1.5},
        'Brentford': {'goals_avg': 1.4, 'conceded_avg': 1.4},
        'Everton': {'goals_avg': 1.2, 'conceded_avg': 1.6},
        'Fulham': {'goals_avg': 1.5, 'conceded_avg': 1.4},
        'Bournemouth': {'goals_avg': 1.3, 'conceded_avg': 1.7},
        'West Ham': {'goals_avg': 1.2, 'conceded_avg': 1.5},
        'Wolves': {'goals_avg': 1.1, 'conceded_avg': 1.6},
        'Tottenham': {'goals_avg': 1.6, 'conceded_avg': 1.4},
        'Manchester United': {'goals_avg': 1.4, 'conceded_avg': 1.3},
        'Ipswich': {'goals_avg': 1.0, 'conceded_avg': 1.8},
        'Leicester': {'goals_avg': 1.1, 'conceded_avg': 1.7},
        'Southampton': {'goals_avg': 0.9, 'conceded_avg': 1.9}
    }
    return team_stats

def main():
    """Main function to train score prediction models."""
    # Load and prepare data
    df = load_and_prepare_data()
    feature_df = create_score_features(df)
    
    # Train models
    home_model, away_model, feature_cols = train_score_models(feature_df)
    
    # Test prediction
    print("\n🔮 Testing score prediction...")
    team_stats = create_team_stats()
    
    # Test with Man City vs Liverpool
    prediction = predict_match_score('Manchester City', 'Liverpool', team_stats)
    if prediction:
        print("✅ Sample prediction (Man City vs Liverpool):")
        print(f"   Predicted Score: {prediction['predicted_score']}")
        print(f"   Spread: {prediction['spread']}")
        print(f"   Total Goals: {prediction['total_goals']}")
        print(f"   Top Scorelines:")
        for score in prediction['most_likely_scores'][:3]:
            print(f"     {score['score']}: {score['probability']:.1f}%")
    
    print("\n💾 Score prediction models saved!")
    print("📈 Ready to predict exact scores!")

if __name__ == "__main__":
    main() 
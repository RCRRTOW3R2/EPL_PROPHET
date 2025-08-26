#!/usr/bin/env python3
"""
EPL PROPHET - Elo Analysis & Feature Engineering
==============================================

Analysis of Elo rating performance and creation of enhanced features
for the forecasting system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add utils to path
sys.path.append('../utils')
from epl_utils import load_epl_data, validate_data_quality

# Add models to path  
sys.path.append('../models')
from elo_rating_system import EloRatingSystem, EloAnalyzer, process_epl_data

def analyze_elo_performance():
    """Comprehensive analysis of Elo rating system performance."""
    
    print("🔍 EPL PROPHET - Elo Analysis")
    print("=" * 50)
    
    # Load data and build Elo system
    data_path = "../data/epl_master_dataset.csv"
    df = load_epl_data(data_path)
    
    print(f"📊 Dataset: {len(df)} matches loaded")
    
    # Build Elo system (suppress output)
    elo_system = process_epl_data(data_path)
    
    # Load outputs
    rankings = pd.read_csv("../outputs/elo_current_rankings.csv")
    history = pd.read_csv("../outputs/elo_rating_history.csv")
    
    print(f"\n🏆 Current Top 10 Teams by Elo Rating:")
    print("-" * 40)
    for idx, row in rankings.head(10).iterrows():
        print(f"{idx+1:2d}. {row['team']:<20} {row['rating']:.0f}")
    
    # Analyze rating distribution
    print(f"\n📈 Rating Distribution:")
    print(f"   Highest Rating: {rankings['rating'].max():.0f} ({rankings.iloc[0]['team']})")
    print(f"   Lowest Rating:  {rankings['rating'].min():.0f} ({rankings.iloc[-1]['team']})")
    print(f"   Average Rating: {rankings['rating'].mean():.0f}")
    print(f"   Rating Spread:  {rankings['rating'].max() - rankings['rating'].min():.0f} points")
    
    # Analyze Big Six performance
    big_six = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham']
    big_six_rankings = rankings[rankings['team'].isin(big_six)].copy()
    big_six_rankings['rank'] = range(1, len(big_six_rankings) + 1)
    
    print(f"\n🔥 Big Six Rankings:")
    print("-" * 30)
    for _, row in big_six_rankings.iterrows():
        overall_rank = rankings[rankings['team'] == row['team']].index[0] + 1
        print(f"{overall_rank:2d}. {row['team']:<18} {row['rating']:.0f}")
    
    # Rating volatility analysis
    history['date'] = pd.to_datetime(history['date'])
    
    print(f"\n📊 System Statistics:")
    print(f"   Total Rating Updates: {len(history):,}")
    print(f"   Average Rating Change: {abs(history['home_change']).mean():.1f} points")
    print(f"   Biggest Rating Gain: {history['home_change'].max():.1f} points")
    print(f"   Biggest Rating Loss: {history['home_change'].min():.1f} points")
    
    return elo_system, rankings, history, df


def create_enhanced_features(elo_system, df):
    """Create enhanced feature dataset with Elo ratings."""
    
    print(f"\n🔧 Creating Enhanced Feature Dataset...")
    
    enhanced_features = []
    
    for idx, match in df.iterrows():
        if pd.isna(match['FTR']) or pd.isna(match['HomeTeam']) or pd.isna(match['AwayTeam']):
            continue
            
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed'].strftime('%Y-%m-%d')
        
        # Get Elo ratings at time of match
        if idx > 0:  # Not the first match
            ratings_at_date = elo_system.get_team_ratings_at_date(match_date)
            home_elo = ratings_at_date.get(home_team, 1500)
            away_elo = ratings_at_date.get(away_team, 1500)
        else:
            home_elo = away_elo = 1500
        
        # Calculate Elo-based features
        elo_diff = home_elo - away_elo
        elo_home_advantage = 100  # Standard home advantage
        
        # Get Elo probabilities
        elo_probs = elo_system.get_match_probabilities(home_team, away_team)
        
        # Create feature record
        features = {
            'match_id': match['match_id'],
            'date': match_date,
            'season': match['season'],
            'home_team': home_team,
            'away_team': away_team,
            
            # Elo features
            'home_elo_rating': home_elo,
            'away_elo_rating': away_elo,
            'elo_rating_diff': elo_diff,
            'elo_prob_home': elo_probs['home_win'],
            'elo_prob_draw': elo_probs['draw'],
            'elo_prob_away': elo_probs['away_win'],
            
            # Market comparison
            'market_prob_home': match.get('market_avg_prob_home', np.nan),
            'market_prob_draw': match.get('market_avg_prob_draw', np.nan),
            'market_prob_away': match.get('market_avg_prob_away', np.nan),
            
            # Probability differences (Elo vs Market)
            'prob_diff_home': elo_probs['home_win'] - match.get('market_avg_prob_home', elo_probs['home_win']),
            'prob_diff_draw': elo_probs['draw'] - match.get('market_avg_prob_draw', elo_probs['draw']),
            'prob_diff_away': elo_probs['away_win'] - match.get('market_avg_prob_away', elo_probs['away_win']),
            
            # Match stats
            'home_goals': match['FTHG'],
            'away_goals': match['FTAG'],
            'total_goals': match['total_goals'],
            'result': match['FTR'],
            
            # Basic match features
            'home_shots': match.get('HS', np.nan),
            'away_shots': match.get('AS', np.nan),
            'home_shots_target': match.get('HST', np.nan),
            'away_shots_target': match.get('AST', np.nan),
        }
        
        enhanced_features.append(features)
        
        if len(enhanced_features) % 1000 == 0:
            print(f"   Processed {len(enhanced_features)} matches...")
    
    features_df = pd.DataFrame(enhanced_features)
    
    print(f"✅ Enhanced features created: {len(features_df)} matches")
    
    return features_df


def evaluate_elo_predictions(features_df):
    """Evaluate Elo prediction accuracy vs market."""
    
    print(f"\n🎯 Elo Prediction Evaluation:")
    print("-" * 35)
    
    # Filter matches with both Elo and market probabilities
    valid_matches = features_df.dropna(subset=['market_prob_home', 'elo_prob_home'])
    
    print(f"Matches with both Elo & Market data: {len(valid_matches)}")
    
    if len(valid_matches) == 0:
        print("No matches with both Elo and market data available")
        return
    
    # Calculate prediction accuracy
    elo_correct = 0
    market_correct = 0
    both_correct = 0
    
    for _, match in valid_matches.iterrows():
        # Determine actual outcome
        actual = match['result']
        
        # Get predicted outcomes (highest probability)
        elo_pred = 'H' if match['elo_prob_home'] > max(match['elo_prob_draw'], match['elo_prob_away']) else (
                  'D' if match['elo_prob_draw'] > match['elo_prob_away'] else 'A')
        
        market_pred = 'H' if match['market_prob_home'] > max(match['market_prob_draw'], match['market_prob_away']) else (
                     'D' if match['market_prob_draw'] > match['market_prob_away'] else 'A')
        
        # Count correct predictions
        if elo_pred == actual:
            elo_correct += 1
        if market_pred == actual:
            market_correct += 1
        if elo_pred == actual and market_pred == actual:
            both_correct += 1
    
    total_matches = len(valid_matches)
    
    print(f"Elo Accuracy:    {elo_correct/total_matches:.1%} ({elo_correct}/{total_matches})")
    print(f"Market Accuracy: {market_correct/total_matches:.1%} ({market_correct}/{total_matches})")
    print(f"Both Correct:    {both_correct/total_matches:.1%} ({both_correct}/{total_matches})")
    
    # Probability calibration
    home_wins = valid_matches[valid_matches['result'] == 'H']
    draws = valid_matches[valid_matches['result'] == 'D']
    away_wins = valid_matches[valid_matches['result'] == 'A']
    
    print(f"\n📊 Probability Calibration:")
    print("-" * 30)
    
    if len(home_wins) > 0:
        elo_home_avg = home_wins['elo_prob_home'].mean()
        market_home_avg = home_wins['market_prob_home'].mean()
        print(f"Home Wins - Elo: {elo_home_avg:.2f}, Market: {market_home_avg:.2f}")
    
    if len(draws) > 0:
        elo_draw_avg = draws['elo_prob_draw'].mean()
        market_draw_avg = draws['market_prob_draw'].mean()
        print(f"Draws     - Elo: {elo_draw_avg:.2f}, Market: {market_draw_avg:.2f}")
    
    if len(away_wins) > 0:
        elo_away_avg = away_wins['elo_prob_away'].mean()
        market_away_avg = away_wins['market_prob_away'].mean()
        print(f"Away Wins - Elo: {elo_away_avg:.2f}, Market: {market_away_avg:.2f}")


def create_team_strength_summary(rankings, df):
    """Create comprehensive team strength summary."""
    
    print(f"\n⚡ Team Strength Analysis:")
    print("-" * 30)
    
    # Calculate additional strength metrics
    strength_summary = []
    
    for _, row in rankings.iterrows():
        team = row['team']
        elo_rating = row['rating']
        
        # Get team's historical performance
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        
        if len(team_matches) == 0:
            continue
        
        # Calculate goals for/against
        goals_for = 0
        goals_against = 0
        wins = draws = losses = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                gf, ga = match['FTHG'], match['FTAG']
                result = match['FTR']
                if result == 'H': wins += 1
                elif result == 'D': draws += 1
                else: losses += 1
            else:
                gf, ga = match['FTAG'], match['FTHG']
                result = match['FTR']
                if result == 'A': wins += 1
                elif result == 'D': draws += 1
                else: losses += 1
            
            goals_for += gf
            goals_against += ga
        
        matches_played = len(team_matches)
        
        strength_summary.append({
            'team': team,
            'elo_rating': elo_rating,
            'matches_played': matches_played,
            'win_rate': wins / matches_played,
            'goals_per_game': goals_for / matches_played,
            'goals_conceded_per_game': goals_against / matches_played,
            'goal_difference_per_game': (goals_for - goals_against) / matches_played,
            'points_per_game': (wins * 3 + draws) / matches_played
        })
    
    strength_df = pd.DataFrame(strength_summary)
    
    # Show correlation between Elo and performance metrics
    correlations = {
        'Win Rate': strength_df['elo_rating'].corr(strength_df['win_rate']),
        'Goals/Game': strength_df['elo_rating'].corr(strength_df['goals_per_game']),
        'Goal Diff/Game': strength_df['elo_rating'].corr(strength_df['goal_difference_per_game']),
        'Points/Game': strength_df['elo_rating'].corr(strength_df['points_per_game'])
    }
    
    print("Elo Rating Correlations:")
    for metric, corr in correlations.items():
        print(f"  {metric:<15}: {corr:.3f}")
    
    return strength_df


def main():
    """Main analysis execution."""
    
    # Run comprehensive Elo analysis
    elo_system, rankings, history, df = analyze_elo_performance()
    
    # Create enhanced features
    features_df = create_enhanced_features(elo_system, df)
    
    # Evaluate predictions
    evaluate_elo_predictions(features_df)
    
    # Team strength analysis
    strength_df = create_team_strength_summary(rankings, df)
    
    # Save enhanced datasets
    features_df.to_csv("../outputs/enhanced_match_features.csv", index=False)
    strength_df.to_csv("../outputs/team_strength_analysis.csv", index=False)
    
    print(f"\n💾 Enhanced datasets saved:")
    print(f"   📊 enhanced_match_features.csv ({len(features_df)} matches)")
    print(f"   💪 team_strength_analysis.csv ({len(strength_df)} teams)")
    
    print(f"\n🎉 Elo Analysis Complete!")
    print(f"   Ready for rolling xG analysis and forecasting models")


if __name__ == "__main__":
    main() 
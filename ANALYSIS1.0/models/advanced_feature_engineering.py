#!/usr/bin/env python3
"""
EPL PROPHET - Advanced Feature Engineering System
===============================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced contextual feature engineering for EPL matches."""
    
    def __init__(self):
        # Feature parameters
        self.h2h_short_window = 5   # Recent head-to-head matches
        self.h2h_long_window = 10   # Extended head-to-head history
        self.position_window = 10   # Matches for league position calculation
        self.congestion_window = 14 # Days to analyze fixture congestion
        
        # Rivalry classifications (based on traditional rivalries)
        self.major_rivalries = {
            'Arsenal': ['Tottenham', 'Chelsea', 'Manchester United'],
            'Tottenham': ['Arsenal', 'Chelsea', 'West Ham'],
            'Chelsea': ['Arsenal', 'Tottenham', 'Liverpool'],
            'Liverpool': ['Manchester United', 'Everton', 'Chelsea', 'Manchester City'],
            'Manchester United': ['Liverpool', 'Manchester City', 'Arsenal', 'Leeds'],
            'Manchester City': ['Manchester United', 'Liverpool'],
            'Everton': ['Liverpool'],
            'West Ham': ['Tottenham'],
            'Leeds': ['Manchester United']
        }
    
    def calculate_rest_days(self, df, match_idx):
        """Calculate rest days for both teams since their last match."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        def get_team_rest_days(team):
            # Get team's previous match
            team_matches = df[
                ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
                (df['date_parsed'] < match_date)
            ].sort_values('date_parsed')
            
            if len(team_matches) == 0:
                return 14  # Default for first match of season
            
            last_match_date = team_matches.iloc[-1]['date_parsed']
            rest_days = (match_date - last_match_date).days
            return min(rest_days, 21)  # Cap at 3 weeks
        
        return {
            'home_rest_days': get_team_rest_days(home_team),
            'away_rest_days': get_team_rest_days(away_team),
            'rest_days_advantage': get_team_rest_days(home_team) - get_team_rest_days(away_team)
        }
    
    def calculate_h2h_features(self, df, match_idx):
        """Calculate head-to-head historical features."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        # Get all historical H2H matches
        h2h_matches = df[
            (((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
             ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))) &
            (df['date_parsed'] < match_date)
        ].sort_values('date_parsed')
        
        if len(h2h_matches) == 0:
            return {
                'h2h_total_matches': 0,
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_home_win_rate': 0.33,
                'h2h_is_rivalry': 0,
                'h2h_avg_total_goals': 3.0
            }
        
        # Overall H2H stats
        total_matches = len(h2h_matches)
        home_wins = draws = away_wins = 0
        total_home_goals = total_away_goals = 0
        
        for _, h2h_match in h2h_matches.iterrows():
            if h2h_match['HomeTeam'] == home_team:
                # Current home team was home in this H2H match
                if h2h_match['FTR'] == 'H':
                    home_wins += 1
                elif h2h_match['FTR'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
                total_home_goals += h2h_match['FTHG']
                total_away_goals += h2h_match['FTAG']
            else:
                # Current home team was away in this H2H match
                if h2h_match['FTR'] == 'A':
                    home_wins += 1
                elif h2h_match['FTR'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
                total_home_goals += h2h_match['FTAG']
                total_away_goals += h2h_match['FTHG']
        
        # Check if this is a rivalry match
        is_rivalry = 0
        if home_team in self.major_rivalries:
            if away_team in self.major_rivalries[home_team]:
                is_rivalry = 1
        
        return {
            'h2h_total_matches': total_matches,
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_home_win_rate': round(home_wins / total_matches, 3),
            'h2h_is_rivalry': is_rivalry,
            'h2h_avg_total_goals': round((total_home_goals + total_away_goals) / total_matches, 2)
        }
    
    def calculate_league_position_features(self, df, match_idx):
        """Calculate league position and momentum features."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        # Get matches before this date in the same season
        season_matches = df[
            (df['season'] == match['season']) &
            (df['date_parsed'] < match_date) &
            (df['FTR'].notna())
        ]
        
        if len(season_matches) == 0:
            return {
                'home_league_position': 10,
                'away_league_position': 10,
                'position_difference': 0,
                'home_points': 0,
                'away_points': 0,
                'points_difference': 0
            }
        
        # Calculate league table at this point
        teams = sorted(set(season_matches['HomeTeam'].unique()) | set(season_matches['AwayTeam'].unique()))
        team_stats = {}
        
        for team in teams:
            points = wins = draws = losses = gf = ga = 0
            
            team_matches = season_matches[
                (season_matches['HomeTeam'] == team) | (season_matches['AwayTeam'] == team)
            ]
            
            for _, team_match in team_matches.iterrows():
                is_home = team_match['HomeTeam'] == team
                
                if is_home:
                    goals_for = team_match['FTHG']
                    goals_against = team_match['FTAG']
                    result = team_match['FTR']
                else:
                    goals_for = team_match['FTAG']
                    goals_against = team_match['FTHG']
                    result = 'H' if team_match['FTR'] == 'A' else ('A' if team_match['FTR'] == 'H' else 'D')
                
                gf += goals_for
                ga += goals_against
                
                if result == 'H' if is_home else result == 'A':
                    wins += 1
                    points += 3
                elif result == 'D':
                    draws += 1
                    points += 1
                else:
                    losses += 1
            
            team_stats[team] = {
                'points': points,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'gf': gf,
                'ga': ga,
                'gd': gf - ga,
                'matches': wins + draws + losses
            }
        
        # Sort teams by points, then goal difference
        league_table = sorted(
            team_stats.items(),
            key=lambda x: (-x[1]['points'], -x[1]['gd'], -x[1]['gf']),
            reverse=False
        )
        
        # Get positions
        position_map = {team: idx + 1 for idx, (team, _) in enumerate(league_table)}
        
        home_position = position_map.get(home_team, 10)
        away_position = position_map.get(away_team, 10)
        home_points = team_stats.get(home_team, {}).get('points', 0)
        away_points = team_stats.get(away_team, {}).get('points', 0)
        
        return {
            'home_league_position': home_position,
            'away_league_position': away_position,
            'position_difference': away_position - home_position,
            'home_points': home_points,
            'away_points': away_points,
            'points_difference': home_points - away_points
        }
    
    def calculate_match_context_features(self, df, match_idx):
        """Calculate match context features (time, season stage, etc.)."""
        
        match = df.iloc[match_idx]
        match_date = match['date_parsed']
        
        # Day of week (0=Monday, 6=Sunday)
        weekday = match_date.weekday()
        is_weekend = 1 if weekday in [5, 6] else 0  # Saturday, Sunday
        is_midweek = 1 if weekday in [1, 2, 3] else 0  # Tuesday, Wednesday, Thursday
        
        # Season stage
        season_matches = df[df['season'] == match['season']]
        total_season_matches = len(season_matches)
        matches_before = len(season_matches[season_matches['date_parsed'] < match_date])
        
        season_progress = matches_before / max(total_season_matches, 1)
        
        # Season stage categories
        is_early_season = 1 if season_progress <= 0.25 else 0
        is_mid_season = 1 if 0.25 < season_progress <= 0.75 else 0
        is_late_season = 1 if season_progress > 0.75 else 0
        
        # Month effects
        month = match_date.month
        is_winter = 1 if month in [12, 1, 2] else 0
        is_spring = 1 if month in [3, 4, 5] else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        is_autumn = 1 if month in [9, 10, 11] else 0
        
        return {
            'match_weekday': weekday,
            'is_weekend': is_weekend,
            'is_midweek': is_midweek,
            'season_progress': round(season_progress, 3),
            'is_early_season': is_early_season,
            'is_mid_season': is_mid_season,
            'is_late_season': is_late_season,
            'is_winter': is_winter,
            'is_spring': is_spring,
            'is_summer': is_summer,
            'is_autumn': is_autumn
        }
    
    def create_advanced_features(self, df, match_idx):
        """Create all advanced features for a single match."""
        
        features = {}
        
        # Add each feature category
        features.update(self.calculate_rest_days(df, match_idx))
        features.update(self.calculate_h2h_features(df, match_idx))
        features.update(self.calculate_league_position_features(df, match_idx))
        features.update(self.calculate_match_context_features(df, match_idx))
        
        return features


def process_advanced_features(data_path):
    """Process advanced features for all EPL matches."""
    
    print("🔄 Building Advanced Feature Engineering System...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Process each match
    advanced_features = []
    processed_matches = 0
    
    for idx, match in df.iterrows():
        if pd.isna(match['FTR']) or pd.isna(match['HomeTeam']) or pd.isna(match['AwayTeam']):
            continue
        
        # Create advanced features for this match
        match_features = {
            'match_id': match['match_id'],
            'date': match['date_parsed'].strftime('%Y-%m-%d'),
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam']
        }
        
        # Add advanced features
        advanced_feat = feature_engineer.create_advanced_features(df, idx)
        match_features.update(advanced_feat)
        
        advanced_features.append(match_features)
        
        processed_matches += 1
        
        if processed_matches % 1000 == 0:
            print(f"   Processed {processed_matches} matches...")
    
    advanced_features_df = pd.DataFrame(advanced_features)
    
    print(f"✅ Advanced feature engineering complete!")
    print(f"   📊 {len(advanced_features_df)} matches processed")
    print(f"   🎯 {len(advanced_features_df.columns) - 4} advanced features created")
    
    # Feature summary
    print(f"\n⚡ Advanced Features Created:")
    print(f"   🛌 Rest Days: 3 features")
    print(f"   🤝 Head-to-Head: 7 features")
    print(f"   📊 League Position: 6 features")
    print(f"   🕒 Match Context: 11 features")
    print(f"   📈 Total: {len(advanced_features_df.columns) - 4} features")
    
    return advanced_features_df


def main():
    """Main execution - build advanced feature engineering system."""
    
    # Build advanced features
    data_path = "../data/epl_master_dataset.csv"
    advanced_features_df = process_advanced_features(data_path)
    
    # Save results
    advanced_features_df.to_csv("../outputs/advanced_match_features.csv", index=False)
    
    print(f"\n💾 Advanced features saved:")
    print(f"   📊 advanced_match_features.csv ({len(advanced_features_df)} matches)")
    
    print(f"\n🎉 Advanced Feature Engineering Complete!")
    print(f"   Ready for ensemble model training with full feature set")
    
    return advanced_features_df


if __name__ == "__main__":
    main()

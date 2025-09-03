#!/usr/bin/env python3
"""
EPL PROPHET - Rolling xG Analysis System
=======================================

Advanced Expected Goals (xG) analysis system featuring:
- Shot-based xG models
- Rolling form calculations (5, 10, 15 match windows)
- Attack/Defense strength indices
- Team performance trends
- xG differential analysis

This provides the xG foundation for our forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

@dataclass
class XGConfig:
    """Configuration for xG analysis system."""
    
    # Rolling window sizes
    short_form_window: int = 5   # Last 5 matches
    medium_form_window: int = 10 # Last 10 matches  
    long_form_window: int = 15   # Last 15 matches
    
    # xG model parameters
    shot_base_xg: float = 0.10          # Base xG per shot
    shot_on_target_xg: float = 0.32     # xG for shots on target
    corner_xg: float = 0.035            # xG per corner
    
    # Quality adjustments
    home_xg_boost: float = 1.1          # 10% boost for home team xG
    big_chance_multiplier: float = 1.5   # Multiplier for high-quality chances
    
    # League averages (for normalization)
    league_avg_goals: float = 2.7       # Average goals per match
    league_avg_shots: float = 12.0      # Average shots per team per match


class RollingXGAnalyzer:
    """
    Rolling Expected Goals analysis system.
    
    Features:
    - Shot-based xG calculation
    - Rolling form analysis (multiple windows)
    - Attack/Defense strength tracking
    - Performance trend analysis
    - League-relative metrics
    """
    
    def __init__(self, config: XGConfig = None):
        self.config = config or XGConfig()
        self.team_xg_history = {}  # Store xG history by team
        self.team_form_cache = {}  # Cache rolling form calculations
        
    def calculate_match_xg(self, shots: int, shots_on_target: int, corners: int, 
                          is_home: bool = False) -> float:
        """Calculate expected goals for a team in a match."""
        
        if pd.isna(shots) or shots < 0:
            shots = 0
        if pd.isna(shots_on_target) or shots_on_target < 0:
            shots_on_target = 0
        if pd.isna(corners) or corners < 0:
            corners = 0
            
        # Ensure shots_on_target doesn't exceed total shots
        shots_on_target = min(shots_on_target, shots)
        shots_off_target = max(0, shots - shots_on_target)
        
        # Base xG calculation
        xg_from_shots_on_target = shots_on_target * self.config.shot_on_target_xg
        xg_from_shots_off_target = shots_off_target * self.config.shot_base_xg
        xg_from_corners = corners * self.config.corner_xg
        
        total_xg = xg_from_shots_on_target + xg_from_shots_off_target + xg_from_corners
        
        # Home advantage boost
        if is_home:
            total_xg *= self.config.home_xg_boost
            
        return round(total_xg, 3)
    
    def calculate_xg_for_match(self, match_row: pd.Series) -> Dict[str, float]:
        """Calculate xG for both teams in a match."""
        
        home_xg = self.calculate_match_xg(
            shots=match_row.get('HS', 0),
            shots_on_target=match_row.get('HST', 0),
            corners=match_row.get('HC', 0),
            is_home=True
        )
        
        away_xg = self.calculate_match_xg(
            shots=match_row.get('AS', 0),
            shots_on_target=match_row.get('AST', 0),
            corners=match_row.get('AC', 0),
            is_home=False
        )
        
        return {
            'home_xg': home_xg,
            'away_xg': away_xg,
            'total_xg': home_xg + away_xg,
            'xg_differential': home_xg - away_xg
        }
    
    def get_team_matches_before_date(self, df: pd.DataFrame, team: str, 
                                   before_date: str, num_matches: int = None) -> pd.DataFrame:
        """Get team's matches before a specific date."""
        
        team_matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['date_parsed'] < pd.to_datetime(before_date))
        ].sort_values('date_parsed')
        
        if num_matches:
            return team_matches.tail(num_matches)
        
        return team_matches
    
    def calculate_rolling_xg_form(self, df: pd.DataFrame, team: str, 
                                before_date: str, window: int) -> Dict[str, float]:
        """Calculate rolling xG form for a team."""
        
        recent_matches = self.get_team_matches_before_date(df, team, before_date, window)
        
        if len(recent_matches) == 0:
            return {
                'matches_played': 0,
                'xg_for': 0.0,
                'xg_against': 0.0,
                'xg_differential': 0.0,
                'xg_for_per_game': 0.0,
                'xg_against_per_game': 0.0,
                'goals_for': 0,
                'goals_against': 0,
                'xg_conversion_rate': 0.0,
                'xg_prevention_rate': 0.0,
                'xg_outperformance': 0.0
            }
        
        xg_for = xg_against = 0.0
        goals_for = goals_against = 0
        
        for _, match in recent_matches.iterrows():
            match_xg = self.calculate_xg_for_match(match)
            
            if match['HomeTeam'] == team:
                # Team playing at home
                xg_for += match_xg['home_xg']
                xg_against += match_xg['away_xg']
                goals_for += match['FTHG']
                goals_against += match['FTAG']
            else:
                # Team playing away
                xg_for += match_xg['away_xg']
                xg_against += match_xg['home_xg']
                goals_for += match['FTAG']
                goals_against += match['FTHG']
        
        matches_played = len(recent_matches)
        xg_for_per_game = xg_for / matches_played
        xg_against_per_game = xg_against / matches_played
        
        # Performance metrics
        xg_conversion_rate = goals_for / xg_for if xg_for > 0 else 0
        xg_prevention_rate = goals_against / xg_against if xg_against > 0 else 0
        xg_outperformance = (goals_for - xg_for) - (goals_against - xg_against)
        
        return {
            'matches_played': matches_played,
            'xg_for': round(xg_for, 3),
            'xg_against': round(xg_against, 3),
            'xg_differential': round(xg_for - xg_against, 3),
            'xg_for_per_game': round(xg_for_per_game, 3),
            'xg_against_per_game': round(xg_against_per_game, 3),
            'goals_for': goals_for,
            'goals_against': goals_against,
            'xg_conversion_rate': round(xg_conversion_rate, 3),
            'xg_prevention_rate': round(xg_prevention_rate, 3),
            'xg_outperformance': round(xg_outperformance, 3)
        }
    
    def calculate_attack_defense_indices(self, df: pd.DataFrame, team: str, 
                                       as_of_date: str = None) -> Dict[str, float]:
        """Calculate attack/defense strength indices relative to league average."""
        
        if as_of_date:
            team_matches = self.get_team_matches_before_date(df, team, as_of_date)
        else:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
        
        if len(team_matches) == 0:
            return {
                'attack_index': 1.0,
                'defense_index': 1.0,
                'xg_attack_index': 1.0,
                'xg_defense_index': 1.0,
                'overall_strength': 1.0
            }
        
        # Calculate team's actual performance
        goals_for = goals_against = 0
        xg_for = xg_against = 0.0
        
        for _, match in team_matches.iterrows():
            match_xg = self.calculate_xg_for_match(match)
            
            if match['HomeTeam'] == team:
                goals_for += match['FTHG']
                goals_against += match['FTAG']
                xg_for += match_xg['home_xg']
                xg_against += match_xg['away_xg']
            else:
                goals_for += match['FTAG']
                goals_against += match['FTHG']
                xg_for += match_xg['away_xg']
                xg_against += match_xg['home_xg']
        
        matches_played = len(team_matches)
        goals_for_per_game = goals_for / matches_played
        goals_against_per_game = goals_against / matches_played
        xg_for_per_game = xg_for / matches_played
        xg_against_per_game = xg_against / matches_played
        
        # Calculate indices relative to league average
        attack_index = goals_for_per_game / (self.config.league_avg_goals / 2)
        defense_index = (self.config.league_avg_goals / 2) / goals_against_per_game if goals_against_per_game > 0 else 2.0
        
        xg_attack_index = xg_for_per_game / (self.config.league_avg_goals / 2)
        xg_defense_index = (self.config.league_avg_goals / 2) / xg_against_per_game if xg_against_per_game > 0 else 2.0
        
        overall_strength = (attack_index * defense_index) ** 0.5
        
        return {
            'attack_index': round(attack_index, 3),
            'defense_index': round(defense_index, 3),
            'xg_attack_index': round(xg_attack_index, 3),
            'xg_defense_index': round(xg_defense_index, 3),
            'overall_strength': round(overall_strength, 3),
            'goals_for_per_game': round(goals_for_per_game, 3),
            'goals_against_per_game': round(goals_against_per_game, 3),
            'xg_for_per_game': round(xg_for_per_game, 3),
            'xg_against_per_game': round(xg_against_per_game, 3)
        }
    
    def calculate_form_trend(self, df: pd.DataFrame, team: str, before_date: str) -> Dict[str, float]:
        """Calculate team's form trend comparing different time windows."""
        
        # Get form for different windows
        short_form = self.calculate_rolling_xg_form(df, team, before_date, self.config.short_form_window)
        medium_form = self.calculate_rolling_xg_form(df, team, before_date, self.config.medium_form_window)
        long_form = self.calculate_rolling_xg_form(df, team, before_date, self.config.long_form_window)
        
        # Calculate trends
        if medium_form['matches_played'] >= self.config.medium_form_window and short_form['matches_played'] >= self.config.short_form_window:
            xg_trend = short_form['xg_for_per_game'] - medium_form['xg_for_per_game']
            defensive_trend = medium_form['xg_against_per_game'] - short_form['xg_against_per_game']  # Improvement = lower xG against
        else:
            xg_trend = defensive_trend = 0.0
        
        # Form momentum
        if short_form['matches_played'] > 0:
            momentum = short_form['xg_differential'] / self.config.short_form_window
        else:
            momentum = 0.0
        
        return {
            'xg_trend': round(xg_trend, 3),
            'defensive_trend': round(defensive_trend, 3),
            'momentum': round(momentum, 3),
            'short_form_xg_diff': short_form['xg_differential'],
            'medium_form_xg_diff': medium_form['xg_differential'],
            'long_form_xg_diff': long_form['xg_differential']
        }
    
    def create_match_xg_features(self, df: pd.DataFrame, match_idx: int) -> Dict[str, float]:
        """Create comprehensive xG features for a single match."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed'].strftime('%Y-%m-%d')
        
        features = {
            'match_id': match['match_id'],
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team
        }
        
        # Calculate xG for this match
        match_xg = self.calculate_xg_for_match(match)
        features.update({
            'actual_home_xg': match_xg['home_xg'],
            'actual_away_xg': match_xg['away_xg'],
            'actual_total_xg': match_xg['total_xg'],
            'actual_xg_differential': match_xg['xg_differential']
        })
        
        # Rolling form features for different windows
        for window_name, window_size in [
            ('short', self.config.short_form_window),
            ('medium', self.config.medium_form_window), 
            ('long', self.config.long_form_window)
        ]:
            
            home_form = self.calculate_rolling_xg_form(df, home_team, match_date, window_size)
            away_form = self.calculate_rolling_xg_form(df, away_team, match_date, window_size)
            
            features.update({
                f'home_{window_name}_xg_for': home_form['xg_for_per_game'],
                f'away_{window_name}_xg_for': away_form['xg_for_per_game'],
                f'home_{window_name}_xg_against': home_form['xg_against_per_game'],
                f'away_{window_name}_xg_against': away_form['xg_against_per_game'],
                f'home_{window_name}_xg_diff': home_form['xg_differential'],
                f'away_{window_name}_xg_diff': away_form['xg_differential'],
                f'home_{window_name}_conversion': home_form['xg_conversion_rate'],
                f'away_{window_name}_conversion': away_form['xg_conversion_rate'],
                f'home_{window_name}_prevention': home_form['xg_prevention_rate'],
                f'away_{window_name}_prevention': away_form['xg_prevention_rate'],
                f'home_{window_name}_outperformance': home_form['xg_outperformance'],
                f'away_{window_name}_outperformance': away_form['xg_outperformance']
            })
        
        # Attack/Defense indices
        home_indices = self.calculate_attack_defense_indices(df, home_team, match_date)
        away_indices = self.calculate_attack_defense_indices(df, away_team, match_date)
        
        features.update({
            'home_attack_index': home_indices['attack_index'],
            'away_attack_index': away_indices['attack_index'],
            'home_defense_index': home_indices['defense_index'],
            'away_defense_index': away_indices['defense_index'],
            'home_xg_attack_index': home_indices['xg_attack_index'],
            'away_xg_attack_index': away_indices['xg_attack_index'],
            'home_xg_defense_index': home_indices['xg_defense_index'],
            'away_xg_defense_index': away_indices['xg_defense_index'],
            'home_overall_strength': home_indices['overall_strength'],
            'away_overall_strength': away_indices['overall_strength']
        })
        
        # Form trends
        home_trend = self.calculate_form_trend(df, home_team, match_date)
        away_trend = self.calculate_form_trend(df, away_team, match_date)
        
        features.update({
            'home_xg_trend': home_trend['xg_trend'],
            'away_xg_trend': away_trend['xg_trend'],
            'home_defensive_trend': home_trend['defensive_trend'],
            'away_defensive_trend': away_trend['defensive_trend'],
            'home_momentum': home_trend['momentum'],
            'away_momentum': away_trend['momentum']
        })
        
        # xG-based predictions
        predicted_home_xg = (home_indices['xg_attack_index'] * away_indices['xg_defense_index'] * 
                           (self.config.league_avg_goals / 2) * self.config.home_xg_boost)
        predicted_away_xg = (away_indices['xg_attack_index'] * home_indices['xg_defense_index'] * 
                            (self.config.league_avg_goals / 2))
        
        features.update({
            'predicted_home_xg': round(predicted_home_xg, 3),
            'predicted_away_xg': round(predicted_away_xg, 3),
            'predicted_total_xg': round(predicted_home_xg + predicted_away_xg, 3),
            'predicted_xg_differential': round(predicted_home_xg - predicted_away_xg, 3)
        })
        
        # Actual outcomes (for training)
        if not pd.isna(match['FTR']):
            features.update({
                'actual_home_goals': match['FTHG'],
                'actual_away_goals': match['FTAG'],
                'actual_total_goals': match['total_goals'],
                'actual_result': match['FTR']
            })
        
        return features


def process_xg_analysis(data_path: str) -> Tuple[pd.DataFrame, RollingXGAnalyzer]:
    """
    Process complete xG analysis for all EPL matches.
    
    Args:
        data_path: Path to the master EPL dataset
        
    Returns:
        DataFrame with xG features and the analyzer instance
    """
    
    print("🔄 Building EPL Rolling xG Analysis System...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Initialize xG analyzer
    xg_analyzer = RollingXGAnalyzer()
    
    # Process each match
    xg_features = []
    processed_matches = 0
    
    for idx, match in df.iterrows():
        if pd.isna(match['FTR']) or pd.isna(match['HomeTeam']) or pd.isna(match['AwayTeam']):
            continue
            
        # Create xG features for this match
        match_features = xg_analyzer.create_match_xg_features(df, idx)
        xg_features.append(match_features)
        
        processed_matches += 1
        
        if processed_matches % 1000 == 0:
            print(f"   Processed {processed_matches} matches...")
    
    xg_features_df = pd.DataFrame(xg_features)
    
    print(f"✅ xG analysis complete!")
    print(f"   📊 {len(xg_features_df)} matches processed")
    print(f"   🎯 {len(xg_features_df.columns)} xG features created")
    
    return xg_features_df, xg_analyzer


def analyze_xg_performance(xg_features_df: pd.DataFrame) -> Dict:
    """Analyze xG model performance and accuracy."""
    
    print(f"\n🎯 xG Model Performance Analysis:")
    print("-" * 40)
    
    # Filter matches with actual outcomes
    complete_matches = xg_features_df.dropna(subset=['actual_home_goals', 'actual_away_goals'])
    
    if len(complete_matches) == 0:
        print("No complete matches for analysis")
        return {}
    
    # Calculate prediction accuracy
    home_xg_error = abs(complete_matches['predicted_home_xg'] - complete_matches['actual_home_xg']).mean()
    away_xg_error = abs(complete_matches['predicted_away_xg'] - complete_matches['actual_away_xg']).mean()
    total_xg_error = abs(complete_matches['predicted_total_xg'] - complete_matches['actual_total_xg']).mean()
    
    # Goal prediction accuracy
    home_goal_error = abs(complete_matches['predicted_home_xg'] - complete_matches['actual_home_goals']).mean()
    away_goal_error = abs(complete_matches['predicted_away_xg'] - complete_matches['actual_away_goals']).mean()
    
    # Correlations
    xg_correlation = complete_matches['predicted_total_xg'].corr(complete_matches['actual_total_xg'])
    goal_correlation = complete_matches['predicted_total_xg'].corr(complete_matches['actual_total_goals'])
    
    print(f"xG Prediction Accuracy:")
    print(f"   Home xG MAE: {home_xg_error:.3f}")
    print(f"   Away xG MAE: {away_xg_error:.3f}")
    print(f"   Total xG MAE: {total_xg_error:.3f}")
    
    print(f"\nGoal Prediction from xG:")
    print(f"   Home Goal MAE: {home_goal_error:.3f}")
    print(f"   Away Goal MAE: {away_goal_error:.3f}")
    
    print(f"\nCorrelations:")
    print(f"   Predicted vs Actual xG: {xg_correlation:.3f}")
    print(f"   Predicted xG vs Goals: {goal_correlation:.3f}")
    
    return {
        'home_xg_mae': home_xg_error,
        'away_xg_mae': away_xg_error,
        'total_xg_mae': total_xg_error,
        'home_goal_mae': home_goal_error,
        'away_goal_mae': away_goal_error,
        'xg_correlation': xg_correlation,
        'goal_correlation': goal_correlation
    }


def create_team_xg_summary(xg_features_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary of team xG performance."""
    
    print(f"\n⚡ Team xG Performance Summary:")
    print("-" * 35)
    
    teams = sorted(set(xg_features_df['home_team'].unique()) | set(xg_features_df['away_team'].unique()))
    
    team_summaries = []
    
    for team in teams:
        # Get team's matches
        home_matches = xg_features_df[xg_features_df['home_team'] == team]
        away_matches = xg_features_df[xg_features_df['away_team'] == team]
        
        if len(home_matches) == 0 and len(away_matches) == 0:
            continue
        
        # Calculate averages
        home_xg_for = home_matches['actual_home_xg'].mean() if len(home_matches) > 0 else 0
        home_xg_against = home_matches['actual_away_xg'].mean() if len(home_matches) > 0 else 0
        away_xg_for = away_matches['actual_away_xg'].mean() if len(away_matches) > 0 else 0
        away_xg_against = away_matches['actual_home_xg'].mean() if len(away_matches) > 0 else 0
        
        total_matches = len(home_matches) + len(away_matches)
        avg_xg_for = ((home_xg_for * len(home_matches)) + (away_xg_for * len(away_matches))) / total_matches if total_matches > 0 else 0
        avg_xg_against = ((home_xg_against * len(home_matches)) + (away_xg_against * len(away_matches))) / total_matches if total_matches > 0 else 0
        
        # Get latest indices
        latest_home = home_matches.iloc[-1] if len(home_matches) > 0 else None
        latest_away = away_matches.iloc[-1] if len(away_matches) > 0 else None
        
        if latest_home is not None:
            attack_index = latest_home['home_attack_index']
            defense_index = latest_home['home_defense_index']
            xg_attack_index = latest_home['home_xg_attack_index']
            xg_defense_index = latest_home['home_xg_defense_index']
        elif latest_away is not None:
            attack_index = latest_away['away_attack_index']
            defense_index = latest_away['away_defense_index']
            xg_attack_index = latest_away['away_xg_attack_index']
            xg_defense_index = latest_away['away_xg_defense_index']
        else:
            attack_index = defense_index = xg_attack_index = xg_defense_index = 1.0
        
        team_summaries.append({
            'team': team,
            'matches_played': total_matches,
            'avg_xg_for': round(avg_xg_for, 3),
            'avg_xg_against': round(avg_xg_against, 3),
            'xg_differential': round(avg_xg_for - avg_xg_against, 3),
            'attack_index': round(attack_index, 3),
            'defense_index': round(defense_index, 3),
            'xg_attack_index': round(xg_attack_index, 3),
            'xg_defense_index': round(xg_defense_index, 3),
            'overall_xg_strength': round((xg_attack_index * xg_defense_index) ** 0.5, 3)
        })
    
    summary_df = pd.DataFrame(team_summaries).sort_values('overall_xg_strength', ascending=False)
    
    print("Top 10 Teams by xG Strength:")
    for idx, row in summary_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['team']:<20} {row['overall_xg_strength']:.3f}")
    
    return summary_df


def main():
    """Main execution - build and analyze xG system."""
    
    # Build xG system
    data_path = "../data/epl_master_dataset.csv"
    xg_features_df, xg_analyzer = process_xg_analysis(data_path)
    
    # Analyze performance
    performance_stats = analyze_xg_performance(xg_features_df)
    
    # Create team summary
    team_xg_summary = create_team_xg_summary(xg_features_df)
    
    # Save results
    xg_features_df.to_csv("../outputs/xg_match_features.csv", index=False)
    team_xg_summary.to_csv("../outputs/team_xg_summary.csv", index=False)
    
    print(f"\n💾 xG analysis results saved:")
    print(f"   📊 xg_match_features.csv ({len(xg_features_df)} matches)")
    print(f"   💪 team_xg_summary.csv ({len(team_xg_summary)} teams)")
    
    print(f"\n🎉 Rolling xG Analysis Complete!")
    print(f"   Ready for ensemble forecasting models")
    
    return xg_features_df, xg_analyzer


if __name__ == "__main__":
    main() 
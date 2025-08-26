#!/usr/bin/env python3
"""
EPL PROPHET - Rolling xG Analysis System
=======================================

Advanced Expected Goals (xG) analysis system featuring:
- Shot-based xG models
- Rolling form calculations (5, 10, 15 match windows)
- Attack/Defense strength tracking
- Team performance trends

This provides the xG foundation for our forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class RollingXGAnalyzer:
    """Rolling Expected Goals analysis system."""
    
    def __init__(self):
        # xG model parameters
        self.shot_base_xg = 0.10
        self.shot_on_target_xg = 0.32
        self.corner_xg = 0.035
        self.home_xg_boost = 1.1
        
        # Rolling window sizes
        self.short_window = 5
        self.medium_window = 10
        self.long_window = 15
        
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
        xg_from_shots_on_target = shots_on_target * self.shot_on_target_xg
        xg_from_shots_off_target = shots_off_target * self.shot_base_xg
        xg_from_corners = corners * self.corner_xg
        
        total_xg = xg_from_shots_on_target + xg_from_shots_off_target + xg_from_corners
        
        # Home advantage boost
        if is_home:
            total_xg *= self.home_xg_boost
            
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
                'xg_for_per_game': 0.0,
                'xg_against_per_game': 0.0,
                'xg_differential': 0.0,
                'xg_conversion_rate': 0.0,
                'xg_outperformance': 0.0
            }
        
        xg_for = xg_against = 0.0
        goals_for = goals_against = 0
        
        for _, match in recent_matches.iterrows():
            match_xg = self.calculate_xg_for_match(match)
            
            if match['HomeTeam'] == team:
                xg_for += match_xg['home_xg']
                xg_against += match_xg['away_xg']
                goals_for += match['FTHG']
                goals_against += match['FTAG']
            else:
                xg_for += match_xg['away_xg']
                xg_against += match_xg['home_xg']
                goals_for += match['FTAG']
                goals_against += match['FTHG']
        
        matches_played = len(recent_matches)
        xg_for_per_game = xg_for / matches_played
        xg_against_per_game = xg_against / matches_played
        
        # Performance metrics
        xg_conversion_rate = goals_for / xg_for if xg_for > 0 else 0
        xg_outperformance = (goals_for - xg_for) - (goals_against - xg_against)
        
        return {
            'matches_played': matches_played,
            'xg_for_per_game': round(xg_for_per_game, 3),
            'xg_against_per_game': round(xg_against_per_game, 3),
            'xg_differential': round(xg_for - xg_against, 3),
            'xg_conversion_rate': round(xg_conversion_rate, 3),
            'xg_outperformance': round(xg_outperformance, 3)
        }
    
    def create_match_xg_features(self, df: pd.DataFrame, match_idx: int) -> Dict:
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
            ('short', self.short_window),
            ('medium', self.medium_window), 
            ('long', self.long_window)
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
                f'home_{window_name}_outperformance': home_form['xg_outperformance'],
                f'away_{window_name}_outperformance': away_form['xg_outperformance']
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
    """Process complete xG analysis for all EPL matches."""
    
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
        
        team_summaries.append({
            'team': team,
            'matches_played': total_matches,
            'avg_xg_for': round(avg_xg_for, 3),
            'avg_xg_against': round(avg_xg_against, 3),
            'xg_differential': round(avg_xg_for - avg_xg_against, 3)
        })
    
    summary_df = pd.DataFrame(team_summaries).sort_values('xg_differential', ascending=False)
    
    print("Top 10 Teams by xG Differential:")
    for idx, row in summary_df.head(10).iterrows():
        print(f"{idx+1:2d}. {row['team']:<20} {row['xg_differential']:.3f}")
    
    return summary_df


def main():
    """Main execution - build and analyze xG system."""
    
    # Build xG system
    data_path = "../data/epl_master_dataset.csv"
    xg_features_df, xg_analyzer = process_xg_analysis(data_path)
    
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
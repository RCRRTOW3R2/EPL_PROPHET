#!/usr/bin/env python3
"""
EPL PROPHET - Enhanced Feature Engineering & Monte Carlo Preparation
================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """Enhanced feature engineering for ML models and Monte Carlo simulations."""
    
    def __init__(self):
        self.efficiency_window = 10
        self.momentum_window = 5
        self.min_matches_for_stats = 5
        
    def calculate_efficiency_metrics(self, df, match_idx):
        """Calculate shot efficiency and conversion metrics."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        def get_team_efficiency(team):
            team_matches = df[
                ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
                (df['date_parsed'] < match_date)
            ].sort_values('date_parsed').tail(self.efficiency_window)
            
            if len(team_matches) == 0:
                return {
                    'shots_per_goal': 10.0,
                    'shot_accuracy': 0.35,
                    'shots_per_match': 12.0,
                    'goals_per_shot': 0.1,
                    'big_chance_conversion': 0.5
                }
            
            total_shots = total_shots_on_target = total_goals = 0
            
            for _, team_match in team_matches.iterrows():
                is_home = team_match['HomeTeam'] == team
                
                if is_home:
                    shots = team_match.get('HS', 0)
                    shots_on_target = team_match.get('HST', 0)
                    goals = team_match.get('FTHG', 0)
                else:
                    shots = team_match.get('AS', 0)
                    shots_on_target = team_match.get('AST', 0)
                    goals = team_match.get('FTAG', 0)
                
                total_shots += shots if not pd.isna(shots) else 0
                total_shots_on_target += shots_on_target if not pd.isna(shots_on_target) else 0
                total_goals += goals if not pd.isna(goals) else 0
            
            shots_per_goal = total_shots / max(total_goals, 1)
            shot_accuracy = total_shots_on_target / max(total_shots, 1)
            goals_per_shot = total_goals / max(total_shots, 1)
            shots_per_match = total_shots / max(len(team_matches), 1)
            big_chance_conversion = total_goals / max(total_shots_on_target, 1)
            
            return {
                'shots_per_goal': round(shots_per_goal, 2),
                'shot_accuracy': round(shot_accuracy, 3),
                'shots_per_match': round(shots_per_match, 1),
                'goals_per_shot': round(goals_per_shot, 3),
                'big_chance_conversion': round(big_chance_conversion, 3)
            }
        
        home_efficiency = get_team_efficiency(home_team)
        away_efficiency = get_team_efficiency(away_team)
        
        return {
            'home_shots_per_goal': home_efficiency['shots_per_goal'],
            'away_shots_per_goal': away_efficiency['shots_per_goal'],
            'home_shot_accuracy': home_efficiency['shot_accuracy'],
            'away_shot_accuracy': away_efficiency['shot_accuracy'],
            'home_big_chance_conversion': home_efficiency['big_chance_conversion'],
            'away_big_chance_conversion': away_efficiency['big_chance_conversion'],
            'home_shots_per_match': home_efficiency['shots_per_match'],
            'away_shots_per_match': away_efficiency['shots_per_match'],
            'efficiency_advantage': home_efficiency['goals_per_shot'] - away_efficiency['goals_per_shot']
        }
    
    def calculate_defensive_metrics(self, df, match_idx):
        """Calculate defensive strength metrics."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        def get_defensive_stats(team):
            team_matches = df[
                ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
                (df['date_parsed'] < match_date)
            ].sort_values('date_parsed').tail(self.efficiency_window)
            
            if len(team_matches) == 0:
                return {
                    'clean_sheet_rate': 0.3,
                    'shots_conceded_per_match': 12.0,
                    'fouls_per_match': 12.0,
                    'cards_per_match': 2.0
                }
            
            clean_sheets = shots_conceded = fouls_committed = cards_received = 0
            
            for _, team_match in team_matches.iterrows():
                is_home = team_match['HomeTeam'] == team
                
                if is_home:
                    goals_conceded = team_match.get('FTAG', 0)
                    shots_against = team_match.get('AS', 0)
                    fouls = team_match.get('HF', 0)
                    cards = team_match.get('HY', 0) + team_match.get('HR', 0)
                else:
                    goals_conceded = team_match.get('FTHG', 0)
                    shots_against = team_match.get('HS', 0)
                    fouls = team_match.get('AF', 0)
                    cards = team_match.get('AY', 0) + team_match.get('AR', 0)
                
                if goals_conceded == 0:
                    clean_sheets += 1
                
                shots_conceded += shots_against if not pd.isna(shots_against) else 0
                fouls_committed += fouls if not pd.isna(fouls) else 0
                cards_received += cards if not pd.isna(cards) else 0
            
            matches_played = len(team_matches)
            
            return {
                'clean_sheet_rate': round(clean_sheets / matches_played, 3),
                'shots_conceded_per_match': round(shots_conceded / matches_played, 1),
                'fouls_per_match': round(fouls_committed / matches_played, 1),
                'cards_per_match': round(cards_received / matches_played, 1)
            }
        
        home_defense = get_defensive_stats(home_team)
        away_defense = get_defensive_stats(away_team)
        
        return {
            'home_clean_sheet_rate': home_defense['clean_sheet_rate'],
            'away_clean_sheet_rate': away_defense['clean_sheet_rate'],
            'home_shots_conceded_pm': home_defense['shots_conceded_per_match'],
            'away_shots_conceded_pm': away_defense['shots_conceded_per_match'],
            'home_fouls_per_match': home_defense['fouls_per_match'],
            'away_fouls_per_match': away_defense['fouls_per_match'],
            'defensive_strength_diff': away_defense['shots_conceded_per_match'] - home_defense['shots_conceded_per_match']
        }
    
    def calculate_momentum_indicators(self, df, match_idx):
        """Calculate momentum and form trajectory indicators."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        def get_momentum_stats(team):
            team_matches = df[
                ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
                (df['date_parsed'] < match_date)
            ].sort_values('date_parsed').tail(self.momentum_window)
            
            if len(team_matches) == 0:
                return {
                    'win_streak': 0,
                    'unbeaten_streak': 0,
                    'scoring_streak': 0,
                    'goals_trend': 0.0,
                    'form_trend': 0.0
                }
            
            win_streak = unbeaten_streak = scoring_streak = 0
            recent_results = []
            recent_goals = []
            
            for _, team_match in team_matches.iterrows():
                is_home = team_match['HomeTeam'] == team
                
                if is_home:
                    goals_for = team_match.get('FTHG', 0)
                    goals_against = team_match.get('FTAG', 0)
                    result = team_match.get('FTR', 'D')
                else:
                    goals_for = team_match.get('FTAG', 0)
                    goals_against = team_match.get('FTHG', 0)
                    result = 'A' if team_match.get('FTR') == 'H' else ('H' if team_match.get('FTR') == 'A' else 'D')
                
                if (is_home and result == 'H') or (not is_home and result == 'A'):
                    points = 3
                    win_streak += 1
                    unbeaten_streak += 1
                elif result == 'D':
                    points = 1
                    win_streak = 0
                    unbeaten_streak += 1
                else:
                    points = 0
                    win_streak = 0
                    unbeaten_streak = 0
                
                recent_results.append(points)
                recent_goals.append(goals_for)
                
                if goals_for > 0:
                    scoring_streak += 1
                else:
                    scoring_streak = 0
            
            if len(recent_results) >= 3:
                x = np.arange(len(recent_results))
                goals_trend = np.polyfit(x, recent_goals, 1)[0]
                form_trend = np.polyfit(x, recent_results, 1)[0]
            else:
                goals_trend = form_trend = 0.0
            
            return {
                'win_streak': win_streak,
                'unbeaten_streak': unbeaten_streak,
                'scoring_streak': scoring_streak,
                'goals_trend': round(goals_trend, 3),
                'form_trend': round(form_trend, 3)
            }
        
        home_momentum = get_momentum_stats(home_team)
        away_momentum = get_momentum_stats(away_team)
        
        return {
            'home_win_streak': home_momentum['win_streak'],
            'away_win_streak': away_momentum['win_streak'],
            'home_unbeaten_streak': home_momentum['unbeaten_streak'],
            'away_unbeaten_streak': away_momentum['unbeaten_streak'],
            'home_scoring_streak': home_momentum['scoring_streak'],
            'away_scoring_streak': away_momentum['scoring_streak'],
            'home_goals_trend': home_momentum['goals_trend'],
            'away_goals_trend': away_momentum['goals_trend'],
            'home_form_trend': home_momentum['form_trend'],
            'away_form_trend': away_momentum['form_trend'],
            'momentum_advantage': home_momentum['form_trend'] - away_momentum['form_trend']
        }
    
    def calculate_score_distribution_features(self, df, match_idx):
        """Calculate score distribution features for Monte Carlo modeling."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed']
        
        def get_score_distribution(team, is_home_analysis):
            if is_home_analysis:
                team_matches = df[
                    (df['HomeTeam'] == team) &
                    (df['date_parsed'] < match_date)
                ]
                goals_col = 'FTHG'
            else:
                team_matches = df[
                    (df['AwayTeam'] == team) &
                    (df['date_parsed'] < match_date)
                ]
                goals_col = 'FTAG'
            
            if len(team_matches) < self.min_matches_for_stats:
                return {
                    'goals_mean': 1.5,
                    'goals_std': 1.2,
                    'prob_0_goals': 0.2,
                    'prob_1_goal': 0.3,
                    'prob_2_goals': 0.25,
                    'prob_3plus_goals': 0.25,
                    'high_scoring_rate': 0.15
                }
            
            goals = team_matches[goals_col].dropna()
            
            goals_mean = goals.mean()
            goals_std = goals.std()
            
            prob_0_goals = (goals == 0).mean()
            prob_1_goal = (goals == 1).mean()
            prob_2_goals = (goals == 2).mean()
            prob_3plus_goals = (goals >= 3).mean()
            high_scoring_rate = (goals >= 3).mean()
            
            return {
                'goals_mean': round(goals_mean, 3),
                'goals_std': round(goals_std, 3),
                'prob_0_goals': round(prob_0_goals, 3),
                'prob_1_goal': round(prob_1_goal, 3),
                'prob_2_goals': round(prob_2_goals, 3),
                'prob_3plus_goals': round(prob_3plus_goals, 3),
                'high_scoring_rate': round(high_scoring_rate, 3)
            }
        
        home_dist = get_score_distribution(home_team, True)
        away_dist = get_score_distribution(away_team, False)
        
        total_goals_expected = home_dist['goals_mean'] + away_dist['goals_mean']
        scoring_variance = home_dist['goals_std']**2 + away_dist['goals_std']**2
        
        return {
            'home_goals_mean': home_dist['goals_mean'],
            'away_goals_mean': away_dist['goals_mean'],
            'home_goals_std': home_dist['goals_std'],
            'away_goals_std': away_dist['goals_std'],
            'home_prob_0_goals': home_dist['prob_0_goals'],
            'away_prob_0_goals': away_dist['prob_0_goals'],
            'home_high_scoring_rate': home_dist['high_scoring_rate'],
            'away_high_scoring_rate': away_dist['high_scoring_rate'],
            'total_goals_expected': round(total_goals_expected, 3),
            'scoring_variance': round(scoring_variance, 3),
            'match_volatility': round(np.sqrt(scoring_variance), 3)
        }
    
    def create_enhanced_features(self, df, match_idx):
        """Create all enhanced features for a single match."""
        
        features = {}
        
        features.update(self.calculate_efficiency_metrics(df, match_idx))
        features.update(self.calculate_defensive_metrics(df, match_idx))
        features.update(self.calculate_momentum_indicators(df, match_idx))
        features.update(self.calculate_score_distribution_features(df, match_idx))
        
        return features


def process_enhanced_features(data_path):
    """Process enhanced features for all EPL matches."""
    
    print("🔄 Building Enhanced Feature Engineering & Monte Carlo Prep...")
    
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    feature_engineer = EnhancedFeatureEngineer()
    
    enhanced_features = []
    processed_matches = 0
    
    for idx, match in df.iterrows():
        if pd.isna(match['FTR']) or pd.isna(match['HomeTeam']) or pd.isna(match['AwayTeam']):
            continue
        
        match_features = {
            'match_id': match['match_id'],
            'date': match['date_parsed'].strftime('%Y-%m-%d'),
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam']
        }
        
        enhanced_feat = feature_engineer.create_enhanced_features(df, idx)
        match_features.update(enhanced_feat)
        
        enhanced_features.append(match_features)
        
        processed_matches += 1
        
        if processed_matches % 1000 == 0:
            print(f"   Processed {processed_matches} matches...")
    
    enhanced_features_df = pd.DataFrame(enhanced_features)
    
    print(f"✅ Enhanced feature engineering complete!")
    print(f"   📊 {len(enhanced_features_df)} matches processed")
    print(f"   🎯 {len(enhanced_features_df.columns) - 4} enhanced features created")
    
    print(f"\n⚡ Enhanced Features Created:")
    print(f"   🎯 Shot Efficiency: 9 features")
    print(f"   🛡️ Defensive Metrics: 7 features")
    print(f"   📈 Momentum Indicators: 11 features")
    print(f"   🎲 Score Distributions: 11 features")
    print(f"   📈 Total: {len(enhanced_features_df.columns) - 4} features")
    
    return enhanced_features_df


def create_monte_carlo_parameters(enhanced_df):
    """Create Monte Carlo simulation parameters."""
    
    print(f"\n🎲 Creating Monte Carlo Parameters...")
    
    mc_params = []
    
    for _, match in enhanced_df.iterrows():
        params = {
            'match_id': match['match_id'],
            'date': match['date'],
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_lambda': match['home_goals_mean'],
            'away_lambda': match['away_goals_mean'],
            'home_variance_adj': match['home_goals_std'] / max(match['home_goals_mean'], 0.1),
            'away_variance_adj': match['away_goals_std'] / max(match['away_goals_mean'], 0.1),
            'home_prob_0': match['home_prob_0_goals'],
            'away_prob_0': match['away_prob_0_goals'],
            'expected_total_goals': match['total_goals_expected'],
            'match_volatility': match['match_volatility'],
            'home_efficiency': match.get('home_big_chance_conversion', 0.5),
            'away_efficiency': match.get('away_big_chance_conversion', 0.5),
            'home_momentum': match.get('home_form_trend', 0.0),
            'away_momentum': match.get('away_form_trend', 0.0)
        }
        
        mc_params.append(params)
    
    mc_params_df = pd.DataFrame(mc_params)
    
    print(f"✅ Monte Carlo parameters created for {len(mc_params_df)} matches")
    
    return mc_params_df


def main():
    """Main execution."""
    
    data_path = "../data/epl_master_dataset.csv"
    enhanced_features_df = process_enhanced_features(data_path)
    mc_params_df = create_monte_carlo_parameters(enhanced_features_df)
    
    enhanced_features_df.to_csv("../outputs/enhanced_match_features_v2.csv", index=False)
    mc_params_df.to_csv("../outputs/monte_carlo_parameters.csv", index=False)
    
    print(f"\n💾 Enhanced features and Monte Carlo prep saved:")
    print(f"   📊 enhanced_match_features_v2.csv ({len(enhanced_features_df)} matches)")
    print(f"   🎲 monte_carlo_parameters.csv ({len(mc_params_df)} matches)")
    
    print(f"\n🎉 Enhanced Feature Engineering & Monte Carlo Prep Complete!")
    print(f"   🤖 Ready for advanced ML ensemble models")
    print(f"   �� Ready for Monte Carlo match simulations")
    print(f"   📈 Total feature arsenal: 160+ features across all systems")
    
    return enhanced_features_df, mc_params_df


if __name__ == "__main__":
    main()

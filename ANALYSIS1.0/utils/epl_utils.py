#!/usr/bin/env python3
"""
EPL PROPHET - Analysis Utilities
================================

Common utility functions for EPL analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


def load_epl_data(data_path: str) -> pd.DataFrame:
    """Load and prepare EPL data for analysis."""
    
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    
    # Sort by date
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    return df


def get_team_matches(df: pd.DataFrame, team: str, as_of_date: str = None) -> pd.DataFrame:
    """Get all matches for a specific team, optionally filtered by date."""
    
    team_matches = df[
        (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
    ].copy()
    
    if as_of_date:
        cutoff_date = pd.to_datetime(as_of_date)
        team_matches = team_matches[team_matches['date_parsed'] <= cutoff_date]
    
    return team_matches.sort_values('date_parsed').reset_index(drop=True)


def get_recent_form(df: pd.DataFrame, team: str, as_of_date: str, num_matches: int = 5) -> Dict:
    """Calculate recent form statistics for a team."""
    
    team_matches = get_team_matches(df, team, as_of_date)
    
    if len(team_matches) == 0:
        return {
            'matches_played': 0,
            'points': 0,
            'goals_for': 0,
            'goals_against': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0
        }
    
    # Get last N matches
    recent_matches = team_matches.tail(num_matches)
    
    points = 0
    goals_for = 0
    goals_against = 0
    wins = draws = losses = 0
    
    for _, match in recent_matches.iterrows():
        is_home = match['HomeTeam'] == team
        
        if is_home:
            gf = match['FTHG']
            ga = match['FTAG']
            result = match['FTR']
        else:
            gf = match['FTAG'] 
            ga = match['FTHG']
            result = 'H' if match['FTR'] == 'A' else ('A' if match['FTR'] == 'H' else 'D')
        
        goals_for += gf
        goals_against += ga
        
        if result == 'H':  # Win from team's perspective
            points += 3
            wins += 1
        elif result == 'D':  # Draw
            points += 1
            draws += 1
        else:  # Loss
            losses += 1
    
    return {
        'matches_played': len(recent_matches),
        'points': points,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'goal_difference': goals_for - goals_against,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'points_per_game': points / len(recent_matches) if len(recent_matches) > 0 else 0,
        'goals_per_game': goals_for / len(recent_matches) if len(recent_matches) > 0 else 0
    }


def calculate_head_to_head(df: pd.DataFrame, team1: str, team2: str, as_of_date: str = None) -> Dict:
    """Calculate head-to-head statistics between two teams."""
    
    h2h_matches = df[
        ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) |
        ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    ]
    
    if as_of_date:
        cutoff_date = pd.to_datetime(as_of_date)
        h2h_matches = h2h_matches[h2h_matches['date_parsed'] <= cutoff_date]
    
    if len(h2h_matches) == 0:
        return {
            'total_matches': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0
        }
    
    team1_wins = team2_wins = draws = 0
    team1_goals = team2_goals = 0
    
    for _, match in h2h_matches.iterrows():
        if match['HomeTeam'] == team1:
            # Team1 at home
            team1_goals += match['FTHG']
            team2_goals += match['FTAG']
            
            if match['FTR'] == 'H':
                team1_wins += 1
            elif match['FTR'] == 'A':
                team2_wins += 1
            else:
                draws += 1
        else:
            # Team2 at home
            team1_goals += match['FTAG']
            team2_goals += match['FTHG']
            
            if match['FTR'] == 'A':
                team1_wins += 1
            elif match['FTR'] == 'H':
                team2_wins += 1
            else:
                draws += 1
    
    return {
        'total_matches': len(h2h_matches),
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'team1_goals': team1_goals,
        'team2_goals': team2_goals,
        'team1_win_rate': team1_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0,
        'avg_goals_per_match': (team1_goals + team2_goals) / len(h2h_matches) if len(h2h_matches) > 0 else 0
    }


def get_rest_days(df: pd.DataFrame, team: str, match_date: str) -> int:
    """Calculate rest days since team's last match."""
    
    team_matches = get_team_matches(df, team, match_date)
    
    if len(team_matches) == 0:
        return 7  # Default assumption
    
    last_match_date = team_matches.iloc[-1]['date_parsed']
    current_match_date = pd.to_datetime(match_date)
    
    rest_days = (current_match_date - last_match_date).days
    
    return max(0, rest_days)


def calculate_season_stats(df: pd.DataFrame, team: str, season: str) -> Dict:
    """Calculate full season statistics for a team."""
    
    season_matches = df[
        (df['season'] == season) & 
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
    ]
    
    if len(season_matches) == 0:
        return {
            'matches_played': 0,
            'points': 0,
            'goals_for': 0,
            'goals_against': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0
        }
    
    points = goals_for = goals_against = 0
    wins = draws = losses = 0
    home_matches = away_matches = 0
    
    for _, match in season_matches.iterrows():
        is_home = match['HomeTeam'] == team
        
        if is_home:
            home_matches += 1
            gf = match['FTHG']
            ga = match['FTAG']
            result = match['FTR']
        else:
            away_matches += 1
            gf = match['FTAG']
            ga = match['FTHG']
            result = 'H' if match['FTR'] == 'A' else ('A' if match['FTR'] == 'H' else 'D')
        
        goals_for += gf
        goals_against += ga
        
        if result == 'H':
            points += 3
            wins += 1
        elif result == 'D':
            points += 1
            draws += 1
        else:
            losses += 1
    
    matches_played = len(season_matches)
    
    return {
        'season': season,
        'matches_played': matches_played,
        'home_matches': home_matches,
        'away_matches': away_matches,
        'points': points,
        'goals_for': goals_for,
        'goals_against': goals_against,
        'goal_difference': goals_for - goals_against,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': wins / matches_played if matches_played > 0 else 0,
        'points_per_game': points / matches_played if matches_played > 0 else 0,
        'goals_per_game': goals_for / matches_played if matches_played > 0 else 0,
        'goals_conceded_per_game': goals_against / matches_played if matches_played > 0 else 0
    }


def get_league_averages(df: pd.DataFrame, season: str = None) -> Dict:
    """Calculate league average statistics."""
    
    if season:
        season_df = df[df['season'] == season]
    else:
        season_df = df
    
    if len(season_df) == 0:
        return {}
    
    total_goals = season_df['FTHG'].sum() + season_df['FTAG'].sum()
    total_matches = len(season_df)
    
    home_wins = (season_df['FTR'] == 'H').sum()
    draws = (season_df['FTR'] == 'D').sum()
    away_wins = (season_df['FTR'] == 'A').sum()
    
    return {
        'avg_goals_per_match': total_goals / total_matches if total_matches > 0 else 0,
        'avg_home_goals': season_df['FTHG'].mean(),
        'avg_away_goals': season_df['FTAG'].mean(),
        'home_win_rate': home_wins / total_matches if total_matches > 0 else 0,
        'draw_rate': draws / total_matches if total_matches > 0 else 0,
        'away_win_rate': away_wins / total_matches if total_matches > 0 else 0,
        'avg_corners_per_match': (season_df['HC'].mean() + season_df['AC'].mean()),
        'avg_cards_per_match': (season_df['HY'].mean() + season_df['AY'].mean() + 
                               season_df['HR'].mean() + season_df['AR'].mean())
    }


def create_match_features(df: pd.DataFrame, match_idx: int, elo_system=None) -> Dict:
    """Create comprehensive features for a single match."""
    
    match = df.iloc[match_idx]
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    match_date = match['date_parsed'].strftime('%Y-%m-%d')
    
    features = {
        'match_id': match['match_id'],
        'date': match_date,
        'home_team': home_team,
        'away_team': away_team,
        'season': match['season']
    }
    
    # Recent form (last 5 matches)
    home_form = get_recent_form(df, home_team, match_date, 5)
    away_form = get_recent_form(df, away_team, match_date, 5)
    
    features.update({
        'home_recent_points': home_form['points'],
        'away_recent_points': away_form['points'],
        'home_recent_gf': home_form['goals_for'],
        'away_recent_gf': away_form['goals_for'],
        'home_recent_ga': home_form['goals_against'],
        'away_recent_ga': away_form['goals_against'],
        'home_recent_form_strength': home_form['points_per_game'],
        'away_recent_form_strength': away_form['points_per_game']
    })
    
    # Head-to-head
    h2h = calculate_head_to_head(df, home_team, away_team, match_date)
    features.update({
        'h2h_matches': h2h['total_matches'],
        'h2h_home_advantage': h2h['team1_win_rate'] if h2h['total_matches'] > 0 else 0.5
    })
    
    # Rest days
    features.update({
        'home_rest_days': get_rest_days(df, home_team, match_date),
        'away_rest_days': get_rest_days(df, away_team, match_date)
    })
    
    # Elo ratings (if provided)
    if elo_system:
        home_rating = elo_system.ratings.get(home_team, 1500)
        away_rating = elo_system.ratings.get(away_team, 1500)
        
        features.update({
            'home_elo_rating': home_rating,
            'away_elo_rating': away_rating,
            'elo_rating_diff': home_rating - away_rating,
            'elo_home_advantage': 100  # Standard home advantage in Elo
        })
        
        # Elo-based probabilities
        probabilities = elo_system.get_match_probabilities(home_team, away_team)
        features.update({
            'elo_prob_home': probabilities['home_win'],
            'elo_prob_draw': probabilities['draw'],
            'elo_prob_away': probabilities['away_win']
        })
    
    # Market probabilities (if available)
    if not pd.isna(match['market_avg_prob_home']):
        features.update({
            'market_prob_home': match['market_avg_prob_home'],
            'market_prob_draw': match['market_avg_prob_draw'],
            'market_prob_away': match['market_avg_prob_away']
        })
    
    # Actual outcome (for training)
    if not pd.isna(match['FTR']):
        features.update({
            'actual_result': match['FTR'],
            'actual_home_goals': match['FTHG'],
            'actual_away_goals': match['FTAG'],
            'actual_total_goals': match['total_goals']
        })
    
    return features


def create_team_strength_matrix(df: pd.DataFrame, as_of_date: str = None) -> pd.DataFrame:
    """Create attack/defense strength matrix for all teams."""
    
    if as_of_date:
        filtered_df = df[df['date_parsed'] <= pd.to_datetime(as_of_date)]
    else:
        filtered_df = df
    
    # Get unique teams
    teams = sorted(set(filtered_df['HomeTeam'].unique()) | set(filtered_df['AwayTeam'].unique()))
    
    team_stats = []
    
    for team in teams:
        team_matches = filtered_df[
            (filtered_df['HomeTeam'] == team) | (filtered_df['AwayTeam'] == team)
        ]
        
        if len(team_matches) == 0:
            continue
        
        # Calculate attack and defense ratings
        goals_for = 0
        goals_against = 0
        matches_played = len(team_matches)
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                goals_for += match['FTHG']
                goals_against += match['FTAG']
            else:
                goals_for += match['FTAG']
                goals_against += match['FTHG']
        
        # League averages for normalization
        league_avg = get_league_averages(filtered_df)
        avg_goals_per_match = league_avg.get('avg_goals_per_match', 2.5)
        
        # Normalize to league average
        attack_strength = (goals_for / matches_played) / (avg_goals_per_match / 2)
        defense_strength = (goals_against / matches_played) / (avg_goals_per_match / 2)
        
        team_stats.append({
            'team': team,
            'matches_played': matches_played,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goals_for_per_game': goals_for / matches_played,
            'goals_against_per_game': goals_against / matches_played,
            'attack_strength': attack_strength,
            'defense_strength': defense_strength,  # Lower is better
            'overall_strength': attack_strength / defense_strength
        })
    
    return pd.DataFrame(team_stats).sort_values('overall_strength', ascending=False)


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """Validate data quality and return summary statistics."""
    
    total_matches = len(df)
    
    quality_report = {
        'total_matches': total_matches,
        'date_range': f"{df['date_parsed'].min()} to {df['date_parsed'].max()}",
        'seasons_covered': df['season'].nunique(),
        'unique_teams': df['HomeTeam'].nunique(),
        
        # Missing data
        'missing_results': df['FTR'].isna().sum(),
        'missing_goals': df[['FTHG', 'FTAG']].isna().sum().sum(),
        'missing_dates': df['date_parsed'].isna().sum(),
        
        # Data consistency
        'invalid_results': (~df['FTR'].isin(['H', 'D', 'A'])).sum(),
        'negative_goals': ((df['FTHG'] < 0) | (df['FTAG'] < 0)).sum(),
        
        # Coverage rates
        'market_odds_coverage': (~df['market_avg_home'].isna()).mean(),
        'match_stats_coverage': (~df['HS'].isna()).mean(),
        
        # Completeness score
        'completeness_score': (total_matches - df['FTR'].isna().sum()) / total_matches if total_matches > 0 else 0
    }
    
    return quality_report 
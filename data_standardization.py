#!/usr/bin/env python3
"""
EPL PROPHET - Data Standardization Pipeline
=========================================

This script standardizes all EPL data files (2014-15 through 2025-26) into a clean,
consistent format for the forecasting system.

Key standardizations:
1. Schema alignment across all seasons
2. Team name normalization
3. Date/time parsing and indexing
4. Core variable extraction (results, stats, odds)
5. Market odds → implied probabilities

Output: Clean master dataset ready for Elo ratings, xG analysis, and forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class EPLDataStandardizer:
    """Standardizes EPL data across all seasons for the Prophet system."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.season_files = self._discover_season_files()
        self.team_name_map = self._create_team_name_mapping()
        
    def _discover_season_files(self) -> Dict[str, Path]:
        """Discover all season CSV files."""
        files = {}
        for file in self.data_dir.glob("*.csv"):
            # Match season patterns like 2425.csv, 1415.csv
            if re.match(r"\d{4}\.csv", file.name):
                season = file.stem
                files[season] = file
        return dict(sorted(files.items()))
    
    def _create_team_name_mapping(self) -> Dict[str, str]:
        """Create standardized team name mapping."""
        return {
            # Current teams (already standardized)
            'Arsenal': 'Arsenal',
            'Aston Villa': 'Aston Villa', 
            'Bournemouth': 'Bournemouth',
            'Brentford': 'Brentford',
            'Brighton': 'Brighton',
            'Burnley': 'Burnley',
            'Chelsea': 'Chelsea',
            'Crystal Palace': 'Crystal Palace',
            'Everton': 'Everton',
            'Fulham': 'Fulham',
            'Ipswich': 'Ipswich',
            'Leicester': 'Leicester',
            'Liverpool': 'Liverpool',
            'Man City': 'Manchester City',
            'Man United': 'Manchester United',
            'Newcastle': 'Newcastle',
            'Nott\'m Forest': 'Nottingham Forest',
            'Southampton': 'Southampton',
            'Tottenham': 'Tottenham',
            'West Ham': 'West Ham',
            'Wolves': 'Wolves',
            
            # Historical teams (relegated/promoted)
            'Cardiff': 'Cardiff City',
            'Hull': 'Hull City',
            'Norwich': 'Norwich City', 
            'QPR': 'Queens Park Rangers',
            'Stoke': 'Stoke City',
            'Swansea': 'Swansea City',
            'Watford': 'Watford',
            'West Brom': 'West Bromwich Albion',
            'Huddersfield': 'Huddersfield Town',
            'Sheffield United': 'Sheffield United',
            'Sheffield Utd': 'Sheffield United',  # Variant
            'Leeds': 'Leeds United',
            'Sunderland': 'Sunderland',
            
            # Handle any variants
            'Manchester City': 'Manchester City',
            'Manchester United': 'Manchester United',
            'Nottingham Forest': 'Nottingham Forest',
        }
    
    def _standardize_columns(self, df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Standardize column names and ensure core columns exist."""
        
        # Core columns that must exist
        core_columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 
            'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
        ]
        
        # Optional columns with defaults
        optional_columns = {
            'Time': '15:00',  # Default kickoff time
            'Referee': 'Unknown',
            'HO': 0, 'AO': 0,  # Offsides
            'HBP': None, 'ABP': None,  # Booking points (calculated later)
        }
        
        # Add missing core columns with NaN
        for col in core_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Add missing optional columns with defaults
        for col, default in optional_columns.items():
            if col not in df.columns:
                df[col] = default
                
        # Ensure booking points are calculated if missing
        if df['HBP'].isna().all():
            df['HBP'] = df['HY'] * 10 + df['HR'] * 25
        if df['ABP'].isna().all():
            df['ABP'] = df['AY'] * 10 + df['AR'] * 25
            
        return df
    
    def _extract_betting_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and standardize betting odds across different column schemas."""
        
        # Market average odds (priority order)
        avg_cols_map = {
            'market_avg_home': ['AvgH', 'BbAvH', 'B365H'],
            'market_avg_draw': ['AvgD', 'BbAvD', 'B365D'], 
            'market_avg_away': ['AvgA', 'BbAvA', 'B365A']
        }
        
        # Market maximum odds
        max_cols_map = {
            'market_max_home': ['MaxH', 'BbMxH'],
            'market_max_draw': ['MaxD', 'BbMxD'],
            'market_max_away': ['MaxA', 'BbMxA']
        }
        
        # Extract best available odds
        for target_col, source_cols in avg_cols_map.items():
            df[target_col] = self._get_first_available_column(df, source_cols)
            
        for target_col, source_cols in max_cols_map.items():
            df[target_col] = self._get_first_available_column(df, source_cols)
            
        # Bet365 odds (most consistent across seasons)
        df['bet365_home'] = self._get_first_available_column(df, ['B365H'])
        df['bet365_draw'] = self._get_first_available_column(df, ['B365D'])
        df['bet365_away'] = self._get_first_available_column(df, ['B365A'])
        
        # Pinnacle odds (sharp market)
        df['pinnacle_home'] = self._get_first_available_column(df, ['PSH', 'PH'])
        df['pinnacle_draw'] = self._get_first_available_column(df, ['PSD', 'PD'])
        df['pinnacle_away'] = self._get_first_available_column(df, ['PSA', 'PA'])
        
        # Over/Under 2.5 goals
        df['over25_avg'] = self._get_first_available_column(df, ['Avg>2.5', 'BbAv>2.5', 'B365>2.5'])
        df['under25_avg'] = self._get_first_available_column(df, ['Avg<2.5', 'BbAv<2.5', 'B365<2.5'])
        
        # Asian Handicap
        df['ah_line'] = self._get_first_available_column(df, ['AHh', 'AHCh', 'BbAHh'])
        df['ah_home_odds'] = self._get_first_available_column(df, ['AvgAHH', 'BbAvAHH', 'B365AHH'])
        df['ah_away_odds'] = self._get_first_available_column(df, ['AvgAHA', 'BbAvAHA', 'B365AHA'])
        
        return df
    
    def _get_first_available_column(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Get the first available column from a list, return NaN if none exist."""
        for col in columns:
            if col in df.columns and not df[col].isna().all():
                return df[col]
        return pd.Series(np.nan, index=df.index)
    
    def _calculate_implied_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert betting odds to implied probabilities."""
        
        odds_columns = [
            ('market_avg_home', 'market_avg_draw', 'market_avg_away', 'market_avg'),
            ('bet365_home', 'bet365_draw', 'bet365_away', 'bet365'),
            ('pinnacle_home', 'pinnacle_draw', 'pinnacle_away', 'pinnacle')
        ]
        
        for home_col, draw_col, away_col, prefix in odds_columns:
            if all(col in df.columns for col in [home_col, draw_col, away_col]):
                # Calculate implied probabilities
                prob_home = 1 / df[home_col]
                prob_draw = 1 / df[draw_col] 
                prob_away = 1 / df[away_col]
                
                # Normalize to remove overround (bookmaker margin)
                total_prob = prob_home + prob_draw + prob_away
                
                df[f'{prefix}_prob_home'] = prob_home / total_prob
                df[f'{prefix}_prob_draw'] = prob_draw / total_prob
                df[f'{prefix}_prob_away'] = prob_away / total_prob
                
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and standardize date formats."""
        
        # Handle different date formats across seasons
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
                
            date_str = str(date_str).strip()
            
            # Try different formats
            formats = ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%m/%d/%Y']
            
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
                    
            # Fallback to pandas parsing
            try:
                return pd.to_datetime(date_str)
            except:
                return pd.NaT
        
        df['date_parsed'] = df['Date'].apply(parse_date)
        
        # Handle 2-digit years (assume 20xx for years < 50, 19xx for >= 50)
        mask = df['date_parsed'].dt.year < 1950
        df.loc[mask, 'date_parsed'] = df.loc[mask, 'date_parsed'] + pd.DateOffset(years=100)
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis."""
        
        # Match outcome encoding
        df['home_win'] = (df['FTR'] == 'H').astype(int)
        df['draw'] = (df['FTR'] == 'D').astype(int) 
        df['away_win'] = (df['FTR'] == 'A').astype(int)
        
        # Goals features
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['goal_difference'] = df['FTHG'] - df['FTAG']
        df['over_25_goals'] = (df['total_goals'] > 2.5).astype(int)
        
        # Shots features (basic xG proxy)
        df['home_shot_accuracy'] = df['HST'] / df['HS'].replace(0, np.nan)
        df['away_shot_accuracy'] = df['AST'] / df['AS'].replace(0, np.nan)
        
        # Simple xG approximation (shots on target weighted)
        df['home_xg_simple'] = df['HST'] * 0.3 + (df['HS'] - df['HST']) * 0.05
        df['away_xg_simple'] = df['AST'] * 0.3 + (df['AS'] - df['AST']) * 0.05
        
        # Match intensity
        df['total_cards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']
        df['total_fouls'] = df['HF'] + df['AF']
        
        return df
    
    def process_season(self, season: str) -> pd.DataFrame:
        """Process a single season file."""
        
        print(f"Processing season {season}...")
        
        file_path = self.season_files[season]
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Add season identifier
        df['season'] = season
        
        # Standardize columns
        df = self._standardize_columns(df, season)
        
        # Normalize team names
        df['HomeTeam'] = df['HomeTeam'].map(self.team_name_map).fillna(df['HomeTeam'])
        df['AwayTeam'] = df['AwayTeam'].map(self.team_name_map).fillna(df['AwayTeam'])
        
        # Extract betting odds
        df = self._extract_betting_odds(df)
        
        # Calculate implied probabilities
        df = self._calculate_implied_probabilities(df)
        
        # Standardize dates
        df = self._standardize_dates(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        print(f"  ✓ {len(df)} matches processed")
        
        return df
    
    def create_master_dataset(self) -> pd.DataFrame:
        """Create the master dataset from all seasons."""
        
        print("Creating EPL Prophet Master Dataset")
        print("=" * 50)
        
        all_seasons = []
        
        for season in self.season_files:
            df_season = self.process_season(season)
            all_seasons.append(df_season)
        
        # Combine all seasons
        master_df = pd.concat(all_seasons, ignore_index=True)
        
        # Sort by date
        master_df = master_df.sort_values('date_parsed').reset_index(drop=True)
        
        # Add match IDs
        master_df['match_id'] = range(1, len(master_df) + 1)
        
        print(f"\nMaster dataset created:")
        print(f"  📊 {len(master_df)} total matches")
        print(f"  📅 {master_df['date_parsed'].min().strftime('%Y-%m-%d')} to {master_df['date_parsed'].max().strftime('%Y-%m-%d')}")
        print(f"  ⚽ {len(master_df['HomeTeam'].unique())} unique teams")
        print(f"  🏆 {len(master_df['season'].unique())} seasons")
        
        return master_df
    
    def select_core_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order core columns for the Prophet system."""
        
        core_cols = [
            # Match metadata
            'match_id', 'season', 'date_parsed', 'Time', 'HomeTeam', 'AwayTeam', 'Referee',
            
            # Results
            'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
            'home_win', 'draw', 'away_win', 'total_goals', 'goal_difference', 'over_25_goals',
            
            # Match stats
            'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
            'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP', 'HO', 'AO',
            
            # Derived stats
            'home_shot_accuracy', 'away_shot_accuracy', 'home_xg_simple', 'away_xg_simple',
            'total_cards', 'total_fouls',
            
            # Market odds
            'market_avg_home', 'market_avg_draw', 'market_avg_away',
            'bet365_home', 'bet365_draw', 'bet365_away',
            'pinnacle_home', 'pinnacle_draw', 'pinnacle_away',
            
            # Implied probabilities  
            'market_avg_prob_home', 'market_avg_prob_draw', 'market_avg_prob_away',
            'bet365_prob_home', 'bet365_prob_draw', 'bet365_prob_away',
            'pinnacle_prob_home', 'pinnacle_prob_draw', 'pinnacle_prob_away',
            
            # Over/Under & Asian Handicap
            'over25_avg', 'under25_avg', 'ah_line', 'ah_home_odds', 'ah_away_odds'
        ]
        
        # Only include columns that exist
        available_cols = [col for col in core_cols if col in df.columns]
        
        return df[available_cols]
    
    def export_datasets(self, master_df: pd.DataFrame):
        """Export clean datasets for the Prophet system."""
        
        # Full master dataset
        core_df = self.select_core_columns(master_df)
        core_df.to_csv('epl_master_dataset.csv', index=False)
        print(f"✅ Master dataset saved: epl_master_dataset.csv ({len(core_df)} matches)")
        
        # Current season only (for live predictions)
        current_season = sorted(self.season_files.keys())[-1]
        current_df = core_df[core_df['season'] == current_season].copy()
        current_df.to_csv('epl_current_season.csv', index=False)
        print(f"✅ Current season saved: epl_current_season.csv ({len(current_df)} matches)")
        
        # Team summary
        team_summary = self._create_team_summary(core_df)
        team_summary.to_csv('epl_team_summary.csv', index=False)
        print(f"✅ Team summary saved: epl_team_summary.csv ({len(team_summary)} teams)")
        
        return core_df
    
    def _create_team_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team summary statistics."""
        
        teams = []
        
        # Filter out NaN team names and sort
        unique_teams = df['HomeTeam'].dropna().unique()
        
        for team in sorted(unique_teams):
            # Home matches
            home_matches = df[df['HomeTeam'] == team]
            # Away matches  
            away_matches = df[df['AwayTeam'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                continue
                
            # Calculate statistics
            home_wins = home_matches['home_win'].sum()
            away_wins = away_matches['away_win'].sum()
            home_draws = home_matches['draw'].sum()
            away_draws = away_matches['draw'].sum()
            
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            total_losses = total_matches - total_wins - total_draws
            
            goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
            goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
            
            teams.append({
                'team': team,
                'total_matches': total_matches,
                'wins': total_wins,
                'draws': total_draws, 
                'losses': total_losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goals_for - goals_against,
                'win_rate': total_wins / total_matches,
                'points': total_wins * 3 + total_draws,
                'avg_goals_for': goals_for / total_matches,
                'avg_goals_against': goals_against / total_matches,
                'first_match': min(home_matches['date_parsed'].min(), away_matches['date_parsed'].min()),
                'last_match': max(home_matches['date_parsed'].max(), away_matches['date_parsed'].max())
            })
        
        return pd.DataFrame(teams)
    
    def run_full_pipeline(self):
        """Run the complete data standardization pipeline."""
        
        # Create master dataset
        master_df = self.create_master_dataset()
        
        # Export datasets
        clean_df = self.export_datasets(master_df)
        
        print(f"\n🎯 EPL Prophet data standardization complete!")
        print(f"   Ready for Elo ratings, rolling xG analysis, and forecasting.")
        
        return clean_df


def main():
    """Main execution function."""
    
    standardizer = EPLDataStandardizer()
    clean_data = standardizer.run_full_pipeline()
    
    print(f"\n📋 Data Quality Summary:")
    print(f"   Missing dates: {clean_data['date_parsed'].isna().sum()}")
    print(f"   Missing results: {clean_data['FTR'].isna().sum()}")
    print(f"   Market odds coverage: {(~clean_data['market_avg_home'].isna()).mean():.1%}")
    print(f"   Match stats coverage: {(~clean_data['HS'].isna()).mean():.1%}")


if __name__ == "__main__":
    main() 
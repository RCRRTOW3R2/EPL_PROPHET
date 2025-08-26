#!/usr/bin/env python3
"""
EPL PROPHET - Elo Rating System
===============================

Advanced Elo rating system for Premier League teams with:
- Historical rating progression
- Home advantage factor
- Goal difference consideration
- Seasonal reversion
- Rating confidence intervals

This provides the foundation for team strength features in our forecasting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

@dataclass
class EloConfig:
    """Configuration for Elo rating system."""
    
    # Core Elo parameters
    initial_rating: int = 1500
    k_factor: int = 32
    home_advantage: int = 100
    
    # Goal difference scaling
    goal_diff_factor: float = 0.1
    max_goal_diff_bonus: float = 0.5
    
    # Seasonal adjustments
    season_reversion_factor: float = 0.25  # Revert 25% toward mean each season
    reversion_target: int = 1500
    
    # New team handling
    new_team_rating: int = 1300  # Start promoted teams slightly below average


class EloRatingSystem:
    """
    Advanced Elo rating system for EPL teams.
    
    Features:
    - Home advantage adjustment
    - Goal difference scaling  
    - Seasonal reversion to prevent rating inflation
    - Historical rating tracking
    - Rating-based match probability predictions
    """
    
    def __init__(self, config: EloConfig = None):
        self.config = config or EloConfig()
        self.ratings = {}  # Current ratings by team
        self.rating_history = []  # Full historical progression
        self.match_count = {}  # Matches played by team (for confidence)
        
    def _expected_score(self, rating_a: float, rating_b: float, home_advantage: float = 0) -> float:
        """Calculate expected score for team A vs team B."""
        rating_diff = rating_a + home_advantage - rating_b
        return 1 / (1 + 10 ** (-rating_diff / 400))
    
    def _calculate_k_factor(self, team: str) -> float:
        """Calculate adaptive K-factor based on team experience."""
        matches_played = self.match_count.get(team, 0)
        
        # Higher K-factor for newer teams (faster rating adjustment)
        if matches_played < 10:
            return self.config.k_factor * 1.5
        elif matches_played < 30:
            return self.config.k_factor * 1.2
        else:
            return self.config.k_factor
    
    def _goal_difference_multiplier(self, goal_diff: int) -> float:
        """Calculate goal difference multiplier for rating change."""
        if goal_diff == 0:  # Draw
            return 1.0
        
        # Logarithmic scaling for goal difference
        multiplier = 1 + (np.log(abs(goal_diff) + 1) * self.config.goal_diff_factor)
        return min(multiplier, 1 + self.config.max_goal_diff_bonus)
    
    def _get_actual_score(self, result: str, perspective: str = 'home') -> float:
        """Convert match result to actual score from team's perspective."""
        if perspective == 'home':
            return {'H': 1.0, 'D': 0.5, 'A': 0.0}[result]
        else:  # away perspective
            return {'H': 0.0, 'D': 0.5, 'A': 1.0}[result]
    
    def update_ratings(self, home_team: str, away_team: str, result: str, 
                      home_goals: int, away_goals: int, match_date: str) -> Dict:
        """
        Update Elo ratings based on match result.
        
        Returns:
            Dict with pre-match ratings, probabilities, and rating changes
        """
        
        # Initialize teams if not seen before
        if home_team not in self.ratings:
            self.ratings[home_team] = self.config.initial_rating
            self.match_count[home_team] = 0
            
        if away_team not in self.ratings:
            self.ratings[away_team] = self.config.initial_rating
            self.match_count[away_team] = 0
        
        # Store pre-match ratings
        home_rating_before = self.ratings[home_team]
        away_rating_before = self.ratings[away_team]
        
        # Calculate expected scores
        home_expected = self._expected_score(
            home_rating_before, away_rating_before, self.config.home_advantage
        )
        away_expected = 1 - home_expected
        
        # Get actual scores
        home_actual = self._get_actual_score(result, 'home')
        away_actual = self._get_actual_score(result, 'away')
        
        # Calculate goal difference multiplier
        goal_diff = home_goals - away_goals
        goal_multiplier = self._goal_difference_multiplier(abs(goal_diff))
        
        # Calculate K-factors
        home_k = self._calculate_k_factor(home_team)
        away_k = self._calculate_k_factor(away_team)
        
        # Calculate rating changes
        home_change = home_k * goal_multiplier * (home_actual - home_expected)
        away_change = away_k * goal_multiplier * (away_actual - away_expected)
        
        # Update ratings
        self.ratings[home_team] += home_change
        self.ratings[away_team] += away_change
        
        # Update match counts
        self.match_count[home_team] += 1
        self.match_count[away_team] += 1
        
        # Store historical record
        match_record = {
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'result': result,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_rating_before': home_rating_before,
            'away_rating_before': away_rating_before,
            'home_rating_after': self.ratings[home_team],
            'away_rating_after': self.ratings[away_team],
            'home_expected': home_expected,
            'away_expected': away_expected,
            'home_change': home_change,
            'away_change': away_change,
            'goal_multiplier': goal_multiplier
        }
        
        self.rating_history.append(match_record)
        
        return match_record
    
    def apply_seasonal_reversion(self, season_end_date: str = None):
        """Apply seasonal reversion to prevent rating inflation."""
        
        reversion_factor = self.config.season_reversion_factor
        target_rating = self.config.reversion_target
        
        for team in self.ratings:
            current_rating = self.ratings[team]
            # Move rating toward target by reversion factor
            self.ratings[team] = current_rating + reversion_factor * (target_rating - current_rating)
            
        print(f"Applied seasonal reversion (factor: {reversion_factor})")
    
    def get_match_probabilities(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate match outcome probabilities based on current Elo ratings."""
        
        if home_team not in self.ratings or away_team not in self.ratings:
            # Default probabilities if teams not rated
            return {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33}
        
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]
        
        # Calculate expected score with home advantage
        home_expected = self._expected_score(home_rating, away_rating, self.config.home_advantage)
        
        # Convert to match probabilities (simplified model)
        # This is a basic conversion - could be enhanced with more sophisticated modeling
        
        if home_expected > 0.7:
            home_win_prob = 0.6 + (home_expected - 0.7) * 0.8
            draw_prob = 0.25 - (home_expected - 0.7) * 0.3
        elif home_expected < 0.3:
            home_win_prob = 0.15 + home_expected * 0.5
            draw_prob = 0.25 - (0.3 - home_expected) * 0.3
        else:
            home_win_prob = 0.15 + (home_expected - 0.3) * 1.125
            draw_prob = 0.25
        
        away_win_prob = 1 - home_win_prob - draw_prob
        
        return {
            'home_win': max(0.05, min(0.9, home_win_prob)),
            'draw': max(0.05, min(0.5, draw_prob)),
            'away_win': max(0.05, min(0.9, away_win_prob))
        }
    
    def get_team_ratings_at_date(self, date: str) -> Dict[str, float]:
        """Get all team ratings as of a specific date."""
        
        # Filter history up to specified date
        history_df = pd.DataFrame(self.rating_history)
        history_df['date'] = pd.to_datetime(history_df['date'])
        target_date = pd.to_datetime(date)
        
        relevant_history = history_df[history_df['date'] <= target_date]
        
        if len(relevant_history) == 0:
            return {team: self.config.initial_rating for team in self.ratings.keys()}
        
        # Get most recent rating for each team
        ratings_at_date = {}
        
        for team in self.ratings.keys():
            team_history = relevant_history[
                (relevant_history['home_team'] == team) | 
                (relevant_history['away_team'] == team)
            ]
            
            if len(team_history) == 0:
                ratings_at_date[team] = self.config.initial_rating
            else:
                last_match = team_history.iloc[-1]
                if last_match['home_team'] == team:
                    ratings_at_date[team] = last_match['home_rating_after']
                else:
                    ratings_at_date[team] = last_match['away_rating_after']
        
        return ratings_at_date
    
    def get_rating_confidence(self, team: str) -> float:
        """Calculate confidence level in team's rating based on matches played."""
        matches_played = self.match_count.get(team, 0)
        
        # Confidence increases logarithmically with matches played
        if matches_played == 0:
            return 0.0
        elif matches_played < 5:
            return 0.3
        elif matches_played < 15:
            return 0.6
        elif matches_played < 30:
            return 0.8
        else:
            return 0.95
    
    def export_rating_history(self) -> pd.DataFrame:
        """Export complete rating history as DataFrame."""
        return pd.DataFrame(self.rating_history)
    
    def get_current_rankings(self) -> pd.DataFrame:
        """Get current team rankings with ratings and confidence."""
        
        rankings = []
        for team, rating in self.ratings.items():
            rankings.append({
                'team': team,
                'rating': rating,
                'matches_played': self.match_count.get(team, 0),
                'confidence': self.get_rating_confidence(team)
            })
        
        df = pd.DataFrame(rankings)
        return df.sort_values('rating', ascending=False).reset_index(drop=True)


class EloAnalyzer:
    """Analysis and visualization tools for Elo rating system."""
    
    def __init__(self, elo_system: EloRatingSystem):
        self.elo = elo_system
        
    def plot_rating_evolution(self, teams: List[str] = None, figsize=(15, 8)):
        """Plot rating evolution over time for specified teams."""
        
        history_df = self.elo.export_rating_history()
        
        if len(history_df) == 0:
            print("No rating history available")
            return
        
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Create rating timeline for each team
        fig, ax = plt.subplots(figsize=figsize)
        
        if teams is None:
            teams = list(self.elo.ratings.keys())[:6]  # Top 6 by current rating
        
        for team in teams:
            team_ratings = []
            dates = []
            
            team_history = history_df[
                (history_df['home_team'] == team) | (history_df['away_team'] == team)
            ].sort_values('date')
            
            for _, match in team_history.iterrows():
                dates.append(match['date'])
                if match['home_team'] == team:
                    team_ratings.append(match['home_rating_after'])
                else:
                    team_ratings.append(match['away_rating_after'])
            
            if len(dates) > 0:
                ax.plot(dates, team_ratings, label=team, linewidth=2, marker='o', markersize=2)
        
        ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.7, label='Initial Rating')
        ax.set_xlabel('Date')
        ax.set_ylabel('Elo Rating')
        ax.set_title('EPL Team Elo Rating Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_prediction_accuracy(self) -> Dict:
        """Analyze how well Elo predictions match actual results."""
        
        history_df = self.elo.export_rating_history()
        
        if len(history_df) == 0:
            return {"error": "No history available"}
        
        # Calculate prediction accuracy
        correct_predictions = 0
        total_predictions = 0
        
        prob_calibration = {'home': [], 'draw': [], 'away': []}
        
        for _, match in history_df.iterrows():
            # Get probabilities that would have been predicted
            probs = self.elo.get_match_probabilities(match['home_team'], match['away_team'])
            
            # Determine predicted outcome
            predicted_outcome = max(probs, key=probs.get)
            actual_outcome = {'H': 'home_win', 'D': 'draw', 'A': 'away_win'}[match['result']]
            
            if predicted_outcome == actual_outcome:
                correct_predictions += 1
            total_predictions += 1
            
            # Store for calibration analysis
            prob_calibration[actual_outcome.replace('_win', '')].append(probs[actual_outcome])
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'prediction_accuracy': accuracy,
            'total_matches': total_predictions,
            'correct_predictions': correct_predictions,
            'avg_home_prob_when_home_won': np.mean(prob_calibration['home']) if prob_calibration['home'] else 0,
            'avg_draw_prob_when_draw': np.mean(prob_calibration['draw']) if prob_calibration['draw'] else 0,
            'avg_away_prob_when_away_won': np.mean(prob_calibration['away']) if prob_calibration['away'] else 0
        }


def process_epl_data(data_path: str) -> EloRatingSystem:
    """
    Process EPL match data and build complete Elo rating system.
    
    Args:
        data_path: Path to the master EPL dataset
        
    Returns:
        Fully trained EloRatingSystem
    """
    
    print("🔄 Building EPL Elo Rating System...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    
    # Sort by date to ensure chronological processing
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Initialize Elo system
    elo_system = EloRatingSystem()
    
    # Process each match
    processed_matches = 0
    seasons_processed = set()
    
    for idx, match in df.iterrows():
        if pd.isna(match['FTR']) or pd.isna(match['HomeTeam']) or pd.isna(match['AwayTeam']):
            continue
            
        # Update ratings
        elo_system.update_ratings(
            home_team=match['HomeTeam'],
            away_team=match['AwayTeam'],
            result=match['FTR'],
            home_goals=int(match['FTHG']),
            away_goals=int(match['FTAG']),
            match_date=match['date_parsed'].strftime('%Y-%m-%d')
        )
        
        processed_matches += 1
        seasons_processed.add(match['season'])
        
        # Apply seasonal reversion at end of each season
        if idx < len(df) - 1:  # Not the last match
            current_season = match['season']
            next_season = df.iloc[idx + 1]['season']
            
            if current_season != next_season:
                elo_system.apply_seasonal_reversion()
                print(f"✅ Season {current_season} complete - applied reversion")
    
    print(f"✅ Elo system built successfully!")
    print(f"   📊 {processed_matches} matches processed")
    print(f"   🏆 {len(seasons_processed)} seasons covered")
    print(f"   ⚽ {len(elo_system.ratings)} teams rated")
    
    return elo_system


def main():
    """Main execution - build and analyze Elo rating system."""
    
    # Build Elo system
    data_path = "../data/epl_master_dataset.csv"
    elo_system = process_epl_data(data_path)
    
    # Display current rankings
    rankings = elo_system.get_current_rankings()
    print(f"\n🏆 Current EPL Elo Rankings:")
    print(rankings.head(10).to_string(index=False))
    
    # Analyze prediction accuracy
    analyzer = EloAnalyzer(elo_system)
    accuracy_stats = analyzer.analyze_prediction_accuracy()
    
    print(f"\n📈 Elo System Performance:")
    print(f"   Prediction Accuracy: {accuracy_stats['prediction_accuracy']:.1%}")
    print(f"   Total Matches: {accuracy_stats['total_matches']}")
    
    # Export results
    rankings.to_csv('../outputs/elo_current_rankings.csv', index=False)
    
    history_df = elo_system.export_rating_history()
    history_df.to_csv('../outputs/elo_rating_history.csv', index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   📊 elo_current_rankings.csv")
    print(f"   📈 elo_rating_history.csv")
    
    return elo_system


if __name__ == "__main__":
    main() 
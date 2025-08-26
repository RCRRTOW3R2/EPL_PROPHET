#!/usr/bin/env python3
"""
EPL PROPHET - Recency Weighted System (Teams as Stocks)
=====================================================

Treats teams like stocks with:
- Exponential Moving Averages (EMA) for all metrics
- Simple Moving Averages (SMA) for comparison  
- Momentum indicators (MACD-style for football)
- Bollinger Bands for form variance
- RSI-style momentum for team strength

This is the CRITICAL FIX for our recency weighting problem!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class TeamStockAnalyzer:
    """Analyze teams like stocks with proper recency weighting."""
    
    def __init__(self):
        # EMA parameters (like stock analysis)
        self.ema_short = 5   # Short-term EMA (like 5-day stock EMA)
        self.ema_medium = 10 # Medium-term EMA (like 10-day stock EMA)
        self.ema_long = 20   # Long-term EMA (like 20-day stock EMA)
        
        # Momentum parameters
        self.momentum_period = 10
        self.rsi_period = 14
        
        # Volatility parameters
        self.bollinger_period = 10
        self.bollinger_std = 2
        
    def calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate Exponential Moving Average (like stocks)."""
        
        if len(values) == 0:
            return 0.0
        
        if len(values) == 1:
            return values[0]
        
        # EMA calculation: EMA = (Value * (2/(Period+1))) + (Previous_EMA * (1-(2/(Period+1))))
        multiplier = 2 / (period + 1)
        ema = values[0]  # Start with first value
        
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return round(ema, 3)
    
    def calculate_sma(self, values: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        
        if len(values) == 0:
            return 0.0
        
        relevant_values = values[-period:] if len(values) >= period else values
        return round(sum(relevant_values) / len(relevant_values), 3)
    
    def calculate_momentum(self, values: List[float]) -> float:
        """Calculate momentum (current vs N periods ago)."""
        
        if len(values) < 2:
            return 0.0
        
        current = values[-1]
        past = values[0] if len(values) < self.momentum_period else values[-self.momentum_period]
        
        momentum = ((current - past) / max(abs(past), 0.1)) * 100
        return round(momentum, 2)
    
    def calculate_rsi_style_form(self, values: List[float]) -> float:
        """Calculate RSI-style form indicator (0-100)."""
        
        if len(values) < 2:
            return 50.0  # Neutral
        
        gains = []
        losses = []
        
        for i in range(1, len(values)):
            change = values[i] - values[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) == 0:
            return 50.0
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 1)
    
    def calculate_bollinger_bands(self, values: List[float]) -> Dict[str, float]:
        """Calculate Bollinger Bands for form variance."""
        
        if len(values) < self.bollinger_period:
            sma = self.calculate_sma(values, len(values))
            return {
                'upper_band': sma + 0.5,
                'middle_band': sma,
                'lower_band': sma - 0.5,
                'band_width': 1.0,
                'position': 50.0  # Neutral position
            }
        
        relevant_values = values[-self.bollinger_period:]
        sma = self.calculate_sma(relevant_values, self.bollinger_period)
        std = np.std(relevant_values)
        
        upper_band = sma + (self.bollinger_std * std)
        lower_band = sma - (self.bollinger_std * std)
        band_width = upper_band - lower_band
        
        # Current position within bands (0-100%)
        current_value = values[-1]
        if band_width > 0:
            position = ((current_value - lower_band) / band_width) * 100
        else:
            position = 50.0
        
        return {
            'upper_band': round(upper_band, 3),
            'middle_band': round(sma, 3),
            'lower_band': round(lower_band, 3),
            'band_width': round(band_width, 3),
            'position': round(max(0, min(100, position)), 1)
        }
    
    def get_team_stock_metrics(self, df: pd.DataFrame, team: str, match_date: str, metric: str = 'goals') -> Dict[str, float]:
        """Get stock-style metrics for a team."""
        
        # Get team's historical data before this match
        team_matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['date_parsed'] < pd.to_datetime(match_date))
        ].sort_values('date_parsed')
        
        if len(team_matches) == 0:
            return self._default_stock_metrics()
        
        # Extract the metric values
        values = []
        
        for _, match in team_matches.iterrows():
            is_home = match['HomeTeam'] == team
            
            if metric == 'goals':
                value = match['FTHG'] if is_home else match['FTAG']
            elif metric == 'goals_against':
                value = match['FTAG'] if is_home else match['FTHG']
            elif metric == 'points':
                result = match['FTR']
                if (is_home and result == 'H') or (not is_home and result == 'A'):
                    value = 3
                elif result == 'D':
                    value = 1
                else:
                    value = 0
            elif metric == 'shots':
                value = match.get('HS' if is_home else 'AS', 10)
            elif metric == 'shot_accuracy':
                shots = match.get('HS' if is_home else 'AS', 10)
                shots_on_target = match.get('HST' if is_home else 'AST', 4)
                value = shots_on_target / max(shots, 1)
            else:
                value = 1.0  # Default
            
            values.append(float(value) if not pd.isna(value) else 0.0)
        
        # Calculate all stock-style indicators
        return {
            'ema_short': self.calculate_ema(values, self.ema_short),
            'ema_medium': self.calculate_ema(values, self.ema_medium),
            'ema_long': self.calculate_ema(values, self.ema_long),
            'sma_short': self.calculate_sma(values, self.ema_short),
            'sma_medium': self.calculate_sma(values, self.ema_medium),
            'sma_long': self.calculate_sma(values, self.ema_long),
            'momentum': self.calculate_momentum(values),
            'rsi_form': self.calculate_rsi_style_form(values),
            'current_value': values[-1] if values else 0.0,
            'matches_analyzed': len(values)
        }
    
    def _default_stock_metrics(self) -> Dict[str, float]:
        """Default metrics for teams with no history."""
        
        return {
            'ema_short': 1.4,
            'ema_medium': 1.4,
            'ema_long': 1.4,
            'sma_short': 1.4,
            'sma_medium': 1.4,
            'sma_long': 1.4,
            'momentum': 0.0,
            'rsi_form': 50.0,
            'current_value': 1.4,
            'matches_analyzed': 0
        }
    
    def calculate_macd_style_indicators(self, df: pd.DataFrame, team: str, match_date: str) -> Dict[str, float]:
        """Calculate MACD-style indicators for team form."""
        
        # Get goals scored metrics
        goals_metrics = self.get_team_stock_metrics(df, team, match_date, 'goals')
        points_metrics = self.get_team_stock_metrics(df, team, match_date, 'points')
        
        # MACD Line = EMA_short - EMA_long
        goals_macd = goals_metrics['ema_short'] - goals_metrics['ema_long']
        points_macd = points_metrics['ema_short'] - points_metrics['ema_long']
        
        # Signal Line = EMA of MACD (simplified as medium EMA)
        goals_signal = goals_metrics['ema_medium'] - goals_metrics['ema_long']
        points_signal = points_metrics['ema_medium'] - points_metrics['ema_long']
        
        # Histogram = MACD - Signal
        goals_histogram = goals_macd - goals_signal
        points_histogram = points_macd - points_signal
        
        return {
            'goals_macd': round(goals_macd, 3),
            'goals_signal': round(goals_signal, 3),
            'goals_histogram': round(goals_histogram, 3),
            'points_macd': round(points_macd, 3),
            'points_signal': round(points_signal, 3),
            'points_histogram': round(points_histogram, 3),
            'form_crossover': 1 if goals_histogram > 0 and points_histogram > 0 else 0
        }
    
    def create_recency_weighted_features(self, df: pd.DataFrame, match_idx: int) -> Dict[str, float]:
        """Create comprehensive recency-weighted features treating teams as stocks."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed'].strftime('%Y-%m-%d')
        
        features = {}
        
        # Get stock-style metrics for different performance indicators
        metrics_to_analyze = ['goals', 'goals_against', 'points', 'shot_accuracy']
        
        for metric in metrics_to_analyze:
            # Home team metrics
            home_metrics = self.get_team_stock_metrics(df, home_team, match_date, metric)
            for key, value in home_metrics.items():
                features[f'home_{metric}_{key}'] = value
            
            # Away team metrics  
            away_metrics = self.get_team_stock_metrics(df, away_team, match_date, metric)
            for key, value in away_metrics.items():
                features[f'away_{metric}_{key}'] = value
            
            # Comparative features (EMA differences)
            features[f'{metric}_ema_short_advantage'] = home_metrics['ema_short'] - away_metrics['ema_short']
            features[f'{metric}_ema_medium_advantage'] = home_metrics['ema_medium'] - away_metrics['ema_medium']
            features[f'{metric}_momentum_advantage'] = home_metrics['momentum'] - away_metrics['momentum']
            features[f'{metric}_rsi_advantage'] = home_metrics['rsi_form'] - away_metrics['rsi_form']
        
        # MACD-style indicators
        home_macd = self.calculate_macd_style_indicators(df, home_team, match_date)
        away_macd = self.calculate_macd_style_indicators(df, away_team, match_date)
        
        for key, value in home_macd.items():
            features[f'home_{key}'] = value
        
        for key, value in away_macd.items():
            features[f'away_{key}'] = value
        
        # Overall team strength indicators
        features['home_overall_momentum'] = (
            home_macd['goals_histogram'] + home_macd['points_histogram']
        ) / 2
        
        features['away_overall_momentum'] = (
            away_macd['goals_histogram'] + away_macd['points_histogram']
        ) / 2
        
        features['momentum_advantage'] = features['home_overall_momentum'] - features['away_overall_momentum']
        
        # Form trend indicators
        home_goals = self.get_team_stock_metrics(df, home_team, match_date, 'goals')
        away_goals = self.get_team_stock_metrics(df, away_team, match_date, 'goals')
        
        features['home_trend_strength'] = 1 if home_goals['ema_short'] > home_goals['ema_long'] else 0
        features['away_trend_strength'] = 1 if away_goals['ema_short'] > away_goals['ema_long'] else 0
        features['trend_advantage'] = features['home_trend_strength'] - features['away_trend_strength']
        
        return features


def process_recency_weighted_dataset(data_path: str) -> pd.DataFrame:
    """Process entire dataset with recency weighting (teams as stocks)."""
    
    print("🔄 BUILDING RECENCY-WEIGHTED SYSTEM (TEAMS AS STOCKS)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Initialize stock analyzer
    stock_analyzer = TeamStockAnalyzer()
    
    print(f"📊 Processing {len(df)} matches with stock-style analysis...")
    
    recency_features = []
    processed_matches = 0
    
    for idx, match in df.iterrows():
        if pd.isna(match.get('FTR')) or pd.isna(match.get('HomeTeam')):
            continue
        
        # Base match info
        match_features = {
            'match_id': match.get('match_id', idx),
            'date': match['date_parsed'].strftime('%Y-%m-%d'),
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam']
        }
        
        # Add recency-weighted stock-style features
        stock_features = stock_analyzer.create_recency_weighted_features(df, idx)
        match_features.update(stock_features)
        
        # Add actual outcomes
        if not pd.isna(match.get('FTR')):
            match_features.update({
                'actual_result': match['FTR'],
                'actual_home_goals': match.get('FTHG', 0),
                'actual_away_goals': match.get('FTAG', 0),
                'actual_total_goals': match.get('FTHG', 0) + match.get('FTAG', 0)
            })
        
        recency_features.append(match_features)
        processed_matches += 1
        
        if processed_matches % 1000 == 0:
            print(f"   📈 Processed {processed_matches} matches...")
    
    recency_df = pd.DataFrame(recency_features)
    
    print(f"✅ Recency weighting complete!")
    print(f"   📊 {len(recency_df)} matches processed")
    print(f"   🎯 {len(recency_df.columns) - 4} stock-style features created")
    
    # Feature summary
    print(f"\n⚡ STOCK-STYLE FEATURES CREATED:")
    print(f"   📈 EMA Features: Short, Medium, Long term exponential averages")
    print(f"   📊 SMA Features: Simple moving averages for comparison")
    print(f"   🚀 Momentum: Rate of change indicators")
    print(f"   📈 RSI-Style: Form strength indicators (0-100)")
    print(f"   📊 MACD-Style: Trend and momentum crossovers")
    print(f"   🎯 Comparative: Head-to-head momentum advantages")
    
    return recency_df


def validate_recency_improvements(old_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict[str, float]:
    """Validate that recency weighting improves predictions."""
    
    print(f"\n🔍 VALIDATING RECENCY IMPROVEMENTS")
    print("=" * 40)
    
    validation_results = {}
    
    # Simple prediction test using EMA vs SMA
    correct_ema = 0
    correct_sma = 0
    total_predictions = 0
    
    for _, match in new_df.iterrows():
        if pd.isna(match.get('actual_result')):
            continue
        
        # EMA-based prediction
        home_ema_goals = match.get('home_goals_ema_short', 1.4)
        away_ema_goals = match.get('away_goals_ema_short', 1.4)
        
        if home_ema_goals > away_ema_goals + 0.3:
            ema_prediction = 'H'
        elif away_ema_goals > home_ema_goals + 0.3:
            ema_prediction = 'A'
        else:
            ema_prediction = 'D'
        
        # SMA-based prediction
        home_sma_goals = match.get('home_goals_sma_short', 1.4)
        away_sma_goals = match.get('away_goals_sma_short', 1.4)
        
        if home_sma_goals > away_sma_goals + 0.3:
            sma_prediction = 'H'
        elif away_sma_goals > home_sma_goals + 0.3:
            sma_prediction = 'A'
        else:
            sma_prediction = 'D'
        
        # Check accuracy
        actual_result = match['actual_result']
        if ema_prediction == actual_result:
            correct_ema += 1
        if sma_prediction == actual_result:
            correct_sma += 1
        
        total_predictions += 1
    
    ema_accuracy = correct_ema / total_predictions if total_predictions > 0 else 0
    sma_accuracy = correct_sma / total_predictions if total_predictions > 0 else 0
    
    validation_results = {
        'ema_accuracy': round(ema_accuracy, 3),
        'sma_accuracy': round(sma_accuracy, 3),
        'improvement': round(ema_accuracy - sma_accuracy, 3),
        'total_predictions': total_predictions
    }
    
    print(f"EMA (Exponential) Accuracy: {ema_accuracy:.3f}")
    print(f"SMA (Simple) Accuracy: {sma_accuracy:.3f}")
    print(f"Improvement: {validation_results['improvement']:+.3f}")
    print(f"Total predictions: {total_predictions}")
    
    if validation_results['improvement'] > 0:
        print("✅ RECENCY WEIGHTING IMPROVES PREDICTIONS!")
    else:
        print("⚠️  Need to adjust recency parameters")
    
    return validation_results


def main():
    """Main execution - implement recency weighting fix."""
    
    print("🚀 EPL PROPHET - CRITICAL RECENCY WEIGHTING FIX")
    print("=" * 55)
    print("Treating teams like stocks with EMA, momentum, and trends!")
    
    # Process with recency weighting
    data_path = "../data/epl_master_dataset.csv"
    recency_df = process_recency_weighted_dataset(data_path)
    
    # Load old dataset for comparison
    old_df = pd.read_csv(data_path)
    
    # Validate improvements
    validation = validate_recency_improvements(old_df, recency_df)
    
    # Save results
    recency_df.to_csv("../outputs/recency_weighted_features.csv", index=False)
    
    print(f"\n💾 Recency-weighted features saved:")
    print(f"   📊 recency_weighted_features.csv ({len(recency_df)} matches)")
    
    print(f"\n🎯 RECENCY WEIGHTING FIX COMPLETE!")
    print(f"   📈 EMA vs SMA improvement: {validation['improvement']:+.3f}")
    print(f"   🚀 Teams now analyzed like stocks")
    print(f"   ⚡ Recent form properly weighted")
    print(f"   🎯 Ready for enhanced ML models!")
    
    return recency_df


if __name__ == "__main__":
    main() 
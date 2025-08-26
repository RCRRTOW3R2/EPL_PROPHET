#!/usr/bin/env python3
"""
EPL PROPHET - CRITICAL RECENCY WEIGHTING FIX
===========================================

Teams as Stocks: EMA, Momentum, MACD-style analysis
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class TeamStockAnalyzer:
    """Analyze teams like stocks with proper recency weighting."""
    
    def __init__(self):
        self.ema_short = 5   # 5-match EMA (recent form)
        self.ema_medium = 10 # 10-match EMA 
        self.ema_long = 20   # 20-match EMA (longer trend)
        
    def calculate_ema(self, values, period):
        """Calculate Exponential Moving Average."""
        
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return values[0]
        
        # EMA formula: EMA = (Value * (2/(Period+1))) + (Previous_EMA * (1-(2/(Period+1))))
        multiplier = 2 / (period + 1)
        ema = values[0]
        
        for value in values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
        
        return round(ema, 3)
    
    def calculate_sma(self, values, period):
        """Calculate Simple Moving Average."""
        
        if len(values) == 0:
            return 0.0
        
        relevant_values = values[-period:] if len(values) >= period else values
        return round(sum(relevant_values) / len(relevant_values), 3)
    
    def calculate_momentum(self, values):
        """Calculate momentum (rate of change)."""
        
        if len(values) < 2:
            return 0.0
        
        current = values[-1]
        past = values[0] if len(values) < 10 else values[-10]
        
        momentum = ((current - past) / max(abs(past), 0.1)) * 100
        return round(momentum, 2)
    
    def get_team_stock_metrics(self, df, team, match_date, metric='goals'):
        """Get stock-style metrics for a team."""
        
        # Get historical matches
        team_matches = df[
            ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) &
            (df['date_parsed'] < pd.to_datetime(match_date))
        ].sort_values('date_parsed')
        
        if len(team_matches) == 0:
            return {
                'ema_short': 1.4, 'ema_medium': 1.4, 'ema_long': 1.4,
                'sma_short': 1.4, 'sma_medium': 1.4, 'sma_long': 1.4,
                'momentum': 0.0, 'current_value': 1.4
            }
        
        # Extract metric values
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
            else:
                value = 1.0
            
            values.append(float(value) if not pd.isna(value) else 0.0)
        
        # Calculate stock indicators
        return {
            'ema_short': self.calculate_ema(values, self.ema_short),
            'ema_medium': self.calculate_ema(values, self.ema_medium),
            'ema_long': self.calculate_ema(values, self.ema_long),
            'sma_short': self.calculate_sma(values, self.ema_short),
            'sma_medium': self.calculate_sma(values, self.ema_medium),
            'sma_long': self.calculate_sma(values, self.ema_long),
            'momentum': self.calculate_momentum(values),
            'current_value': values[-1] if values else 0.0
        }
    
    def create_stock_features(self, df, match_idx):
        """Create stock-style features for a match."""
        
        match = df.iloc[match_idx]
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['date_parsed'].strftime('%Y-%m-%d')
        
        features = {}
        
        # Analyze different metrics
        metrics = ['goals', 'goals_against', 'points']
        
        for metric in metrics:
            # Home team
            home_metrics = self.get_team_stock_metrics(df, home_team, match_date, metric)
            for key, value in home_metrics.items():
                features[f'home_{metric}_{key}'] = value
            
            # Away team
            away_metrics = self.get_team_stock_metrics(df, away_team, match_date, metric)
            for key, value in away_metrics.items():
                features[f'away_{metric}_{key}'] = value
            
            # Advantages (differences)
            features[f'{metric}_ema_advantage'] = home_metrics['ema_short'] - away_metrics['ema_short']
            features[f'{metric}_momentum_advantage'] = home_metrics['momentum'] - away_metrics['momentum']
        
        # Overall momentum indicators
        home_goals = self.get_team_stock_metrics(df, home_team, match_date, 'goals')
        away_goals = self.get_team_stock_metrics(df, away_team, match_date, 'goals')
        
        # Trend strength (EMA short vs long)
        features['home_trend_strength'] = 1 if home_goals['ema_short'] > home_goals['ema_long'] else 0
        features['away_trend_strength'] = 1 if away_goals['ema_short'] > away_goals['ema_long'] else 0
        
        # MACD-style indicator
        features['home_macd'] = home_goals['ema_short'] - home_goals['ema_long']
        features['away_macd'] = away_goals['ema_short'] - away_goals['ema_long']
        features['macd_advantage'] = features['home_macd'] - features['away_macd']
        
        return features


def process_recency_weighted_data():
    """Process dataset with recency weighting."""
    
    print("🚀 EPL PROPHET - CRITICAL RECENCY WEIGHTING FIX")
    print("=" * 55)
    print("Treating teams like stocks with EMA, momentum, and trends!")
    
    # Load data
    df = pd.read_csv("../data/epl_master_dataset.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    print(f"📊 Processing {len(df)} matches with stock-style analysis...")
    
    # Initialize analyzer
    stock_analyzer = TeamStockAnalyzer()
    
    recency_features = []
    processed = 0
    
    for idx, match in df.iterrows():
        if pd.isna(match.get('FTR')) or pd.isna(match.get('HomeTeam')):
            continue
        
        # Base info
        match_features = {
            'match_id': match.get('match_id', idx),
            'date': match['date_parsed'].strftime('%Y-%m-%d'),
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam']
        }
        
        # Add stock-style features
        stock_features = stock_analyzer.create_stock_features(df, idx)
        match_features.update(stock_features)
        
        # Add outcomes
        if not pd.isna(match.get('FTR')):
            match_features.update({
                'actual_result': match['FTR'],
                'actual_home_goals': match.get('FTHG', 0),
                'actual_away_goals': match.get('FTAG', 0)
            })
        
        recency_features.append(match_features)
        processed += 1
        
        if processed % 1000 == 0:
            print(f"   📈 Processed {processed} matches...")
    
    recency_df = pd.DataFrame(recency_features)
    
    print(f"✅ Recency weighting complete!")
    print(f"   📊 {len(recency_df)} matches processed")
    print(f"   🎯 {len(recency_df.columns) - 4} stock-style features created")
    
    # Validation: EMA vs SMA prediction accuracy
    print(f"\n🔍 VALIDATING EMA vs SMA IMPROVEMENTS")
    print("=" * 40)
    
    correct_ema = 0
    correct_sma = 0
    total = 0
    
    for _, match in recency_df.iterrows():
        if pd.isna(match.get('actual_result')):
            continue
        
        # EMA prediction
        home_ema = match.get('home_goals_ema_short', 1.4)
        away_ema = match.get('away_goals_ema_short', 1.4)
        
        if home_ema > away_ema + 0.3:
            ema_pred = 'H'
        elif away_ema > home_ema + 0.3:
            ema_pred = 'A'
        else:
            ema_pred = 'D'
        
        # SMA prediction
        home_sma = match.get('home_goals_sma_short', 1.4)
        away_sma = match.get('away_goals_sma_short', 1.4)
        
        if home_sma > away_sma + 0.3:
            sma_pred = 'H'
        elif away_sma > home_sma + 0.3:
            sma_pred = 'A'
        else:
            sma_pred = 'D'
        
        actual = match['actual_result']
        if ema_pred == actual:
            correct_ema += 1
        if sma_pred == actual:
            correct_sma += 1
        total += 1
    
    ema_accuracy = correct_ema / total if total > 0 else 0
    sma_accuracy = correct_sma / total if total > 0 else 0
    improvement = ema_accuracy - sma_accuracy
    
    print(f"EMA (Exponential) Accuracy: {ema_accuracy:.3f} ({correct_ema}/{total})")
    print(f"SMA (Simple) Accuracy: {sma_accuracy:.3f} ({correct_sma}/{total})")
    print(f"Improvement: {improvement:+.3f}")
    
    if improvement > 0:
        print("✅ RECENCY WEIGHTING IMPROVES PREDICTIONS!")
    else:
        print("⚠️  Need parameter adjustment")
    
    # Save results
    recency_df.to_csv("../outputs/recency_weighted_stock_features.csv", index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   📊 recency_weighted_stock_features.csv ({len(recency_df)} matches)")
    
    print(f"\n🎯 CRITICAL RECENCY FIX COMPLETE!")
    print(f"   📈 EMA improvement: {improvement:+.3f}")
    print(f"   🚀 Teams analyzed as stocks")
    print(f"   ⚡ Recent form properly weighted")
    print(f"   🎯 Ready for ML models!")
    
    # Show sample of key features
    print(f"\n📊 SAMPLE STOCK FEATURES:")
    sample = recency_df.iloc[1000]
    print(f"Match: {sample['home_team']} vs {sample['away_team']}")
    print(f"Home Goals EMA: {sample.get('home_goals_ema_short', 0):.2f}")
    print(f"Away Goals EMA: {sample.get('away_goals_ema_short', 0):.2f}")
    print(f"MACD Advantage: {sample.get('macd_advantage', 0):.2f}")
    print(f"Goals EMA Advantage: {sample.get('goals_ema_advantage', 0):.2f}")
    
    return recency_df

if __name__ == "__main__":
    process_recency_weighted_data()

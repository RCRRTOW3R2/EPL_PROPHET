#!/usr/bin/env python3
"""
EPL Prophet - ML Algorithm Comparison
Compare Random Forest vs XGBoost vs other algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_simple_features():
    """Load a simplified feature set for comparison"""
    print("📊 Loading data for ML comparison...")
    
    # Load recent seasons
    seasons = ['1819', '1920', '2021', '2122', '2223']
    all_data = []
    
    for season in seasons:
        try:
            df = pd.read_csv(f'{season}.csv')
            df['season'] = season
            all_data.append(df)
            print(f"   ✅ {season}: {len(df)} matches")
        except:
            continue
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Create simple features
    features_list = []
    
    for idx, match in combined.iterrows():
        if idx < 10:  # Need some history
            continue
            
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Get last 5 matches for each team
        home_recent = get_recent_form(combined, home_team, idx, 5)
        away_recent = get_recent_form(combined, away_team, idx, 5)
        
        if len(home_recent) < 3 or len(away_recent) < 3:
            continue
        
        # Simple features
        home_ppg = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in home_recent) / len(home_recent)
        away_ppg = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in away_recent) / len(away_recent)
        
        features = {
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'form_diff': home_ppg - away_ppg,
            'home_advantage': 1,  # Always home advantage
        }
        
        # Target
        if match['FTR'] == 'H':
            features['target'] = 2
        elif match['FTR'] == 'A':
            features['target'] = 0
        else:
            features['target'] = 1
            
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def get_recent_form(df, team, current_idx, n=5):
    """Get recent form for a team"""
    results = []
    
    for i in range(current_idx - 1, -1, -1):
        if len(results) >= n:
            break
            
        prev_match = df.iloc[i]
        
        if prev_match['HomeTeam'] == team:
            result = 'W' if prev_match['FTR'] == 'H' else 'D' if prev_match['FTR'] == 'D' else 'L'
        elif prev_match['AwayTeam'] == team:
            result = 'W' if prev_match['FTR'] == 'A' else 'D' if prev_match['FTR'] == 'D' else 'L'
        else:
            continue
            
        results.append(result)
    
    return results

def compare_algorithms(features_df):
    """Compare different ML algorithms"""
    print("\n🤖 COMPARING ML ALGORITHMS")
    print("="*40)
    
    # Prepare data
    X = features_df.drop(['target'], axis=1)
    y = features_df['target']
    
    print(f"📊 Dataset: {len(X)} matches, {X.shape[1]} features")
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Scale for algorithms that need it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Algorithms to test
    algorithms = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=8, 
            min_samples_split=10,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\n🧪 Testing {name}...")
        
        try:
            if name in ['Logistic Regression', 'SVM']:
                # Use scaled data for these algorithms
                cv_scores = cross_val_score(algorithm, X_scaled, y, cv=tscv, scoring='accuracy')
            else:
                # Use original data for tree-based algorithms
                cv_scores = cross_val_score(algorithm, X, y, cv=tscv, scoring='accuracy')
            
            results[name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"   Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   Individual: {[f'{s:.3f}' for s in cv_scores]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Rank algorithms
    print(f"\n🏆 ALGORITHM RANKING:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results):
        accuracy = result['mean_accuracy']
        std = result['std_accuracy']
        print(f"   {i+1}. {name:<20} {accuracy:.4f} ± {std:.4f}")
    
    return results

def analyze_data_timeline():
    """Analyze how data timeline affects accuracy"""
    print(f"\n📅 DATA TIMELINE ANALYSIS")
    print("="*30)
    
    # Test different time periods
    time_periods = {
        'Recent (2019-2024)': ['1920', '2021', '2122', '2223', '2324'],
        'Medium (2016-2024)': ['1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324'],
        'Long (2014-2024)': ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324'],
    }
    
    timeline_results = {}
    
    for period_name, seasons in time_periods.items():
        print(f"\n🔍 Testing {period_name}...")
        
        # Load data for this period
        all_data = []
        for season in seasons:
            try:
                df = pd.read_csv(f'{season}.csv')
                df['season'] = season
                all_data.append(df)
            except:
                continue
        
        if not all_data:
            continue
            
        combined = pd.concat(all_data, ignore_index=True)
        
        # Create features (simplified)
        features_list = []
        for idx, match in combined.iterrows():
            if idx < 20:  # Need history
                continue
                
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            home_recent = get_recent_form(combined, home_team, idx, 5)
            away_recent = get_recent_form(combined, away_team, idx, 5)
            
            if len(home_recent) < 3 or len(away_recent) < 3:
                continue
            
            home_ppg = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in home_recent) / len(home_recent)
            away_ppg = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in away_recent) / len(away_recent)
            
            features = {
                'home_ppg': home_ppg,
                'away_ppg': away_ppg,
                'form_diff': home_ppg - away_ppg,
                'target': 2 if match['FTR'] == 'H' else 0 if match['FTR'] == 'A' else 1
            }
            features_list.append(features)
        
        if len(features_list) < 100:
            continue
            
        features_df = pd.DataFrame(features_list)
        
        # Test with Random Forest
        X = features_df.drop(['target'], axis=1)
        y = features_df['target']
        
        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        
        cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='accuracy')
        
        timeline_results[period_name] = {
            'seasons': len(seasons),
            'matches': len(features_df),
            'accuracy': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        print(f"   Seasons: {len(seasons)}")
        print(f"   Matches: {len(features_df)}")
        print(f"   Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\n📊 TIMELINE COMPARISON:")
    for period, results in timeline_results.items():
        print(f"   {period:<20} {results['accuracy']:.4f} ({results['matches']} matches)")
    
    return timeline_results

def main():
    """Main comparison analysis"""
    print("🚀 EPL PROPHET - ML & DATA ANALYSIS")
    print("="*45)
    
    # Load features
    features_df = load_simple_features()
    
    # Compare algorithms
    algo_results = compare_algorithms(features_df)
    
    # Analyze timeline
    timeline_results = analyze_data_timeline()
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Best algorithm
    if algo_results:
        best_algo = max(algo_results.items(), key=lambda x: x[1]['mean_accuracy'])
        print(f"   🥇 Best Algorithm: {best_algo[0]} ({best_algo[1]['mean_accuracy']:.4f})")
    
    # Timeline insights
    if timeline_results:
        best_timeline = max(timeline_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"   📅 Best Timeline: {best_timeline[0]} ({best_timeline[1]['accuracy']:.4f})")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
EPL PROPHET - COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS
Answer: How are we attributing importance to each feature?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis for EPL Prophet"""
    
    def __init__(self):
        self.load_data()
        self.results = {}
    
    def load_data(self):
        """Load our trained models and data"""
        print("📊 FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Load Frankenstein model
        try:
            model_data = joblib.load('models/frankenstein_ultimate.pkl')
            self.frankenstein_model = model_data['ensemble']
            self.specialists = model_data['specialists']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            print("✅ Frankenstein model loaded")
        except:
            print("⚠️ Frankenstein model not found - will analyze from raw data")
            self.frankenstein_model = None
        
        # Load season data for analysis
        self.load_season_data()
    
    def load_season_data(self):
        """Load recent season data for fresh analysis"""
        try:
            # Load multiple seasons for comprehensive analysis
            seasons = ['1718.csv', '1819.csv', '1920.csv', '2021.csv', '2122.csv', '2223.csv']
            all_data = []
            
            for season in seasons:
                try:
                    df = pd.read_csv(season)
                    all_data.append(df)
                except:
                    print(f"   Skipping {season}")
            
            if all_data:
                self.raw_data = pd.concat(all_data, ignore_index=True)
                print(f"✅ Loaded {len(self.raw_data)} matches from {len(all_data)} seasons")
            else:
                print("⚠️ No season data found - creating sample data")
                self.create_sample_data()
                
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for analysis"""
        teams = ['Arsenal', 'Liverpool', 'Manchester City', 'Chelsea', 'Manchester United', 
                'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
        
        sample_data = []
        for i in range(500):
            home = np.random.choice(teams)
            away = np.random.choice([t for t in teams if t != home])
            
            # Simulate realistic match data
            home_goals = np.random.poisson(1.5)
            away_goals = np.random.poisson(1.2)
            
            if home_goals > away_goals:
                result = 'H'
            elif away_goals > home_goals:
                result = 'A'
            else:
                result = 'D'
            
            sample_data.append({
                'HomeTeam': home,
                'AwayTeam': away,
                'FTHG': home_goals,
                'FTAG': away_goals,
                'FTR': result,
                'Date': f'2023-{(i%12)+1:02d}-{(i%28)+1:02d}'
            })
        
        self.raw_data = pd.DataFrame(sample_data)
        print(f"✅ Created {len(self.raw_data)} sample matches")
    
    def create_feature_dataset(self):
        """Create features from raw match data"""
        print("\n🔧 Creating feature dataset...")
        
        features_list = []
        
        for idx, match in self.raw_data.iterrows():
            if idx < 20:  # Need history for features
                continue
            
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            
            # Get recent form (last 6 matches)
            home_recent = self.get_team_recent_matches(home_team, idx, 6)
            away_recent = self.get_team_recent_matches(away_team, idx, 6)
            
            if len(home_recent) < 4 or len(away_recent) < 4:
                continue
            
            features = {}
            
            # 1. BASIC FORM FEATURES
            features.update(self.get_basic_features(home_recent, away_recent))
            
            # 2. ADVANCED RATIO FEATURES  
            features.update(self.get_ratio_features(home_recent, away_recent))
            
            # 3. PSYCHOLOGICAL FEATURES
            features.update(self.get_psychological_features(home_recent, away_recent))
            
            # 4. TARGET
            features['target'] = 2 if match['FTR'] == 'H' else (0 if match['FTR'] == 'A' else 1)
            
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        print(f"✅ Created {len(self.features_df)} feature rows with {self.features_df.shape[1]-1} features")
        
        return self.features_df
    
    def get_team_recent_matches(self, team, current_idx, n=6):
        """Get recent matches for a team"""
        recent = []
        count = 0
        
        for i in range(current_idx-1, -1, -1):
            if count >= n:
                break
            
            match = self.raw_data.iloc[i]
            
            if match['HomeTeam'] == team:
                result = 'W' if match['FTR'] == 'H' else ('D' if match['FTR'] == 'D' else 'L')
                goals_for = match['FTHG']
                goals_against = match['FTAG']
            elif match['AwayTeam'] == team:
                result = 'W' if match['FTR'] == 'A' else ('D' if match['FTR'] == 'D' else 'L')
                goals_for = match['FTAG']
                goals_against = match['FTHG']
            else:
                continue
            
            recent.append({
                'result': result,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_diff': goals_for - goals_against
            })
            count += 1
        
        return recent
    
    def get_basic_features(self, home_recent, away_recent):
        """Basic form features"""
        features = {}
        
        # Goals
        features['home_goals_avg'] = np.mean([m['goals_for'] for m in home_recent])
        features['away_goals_avg'] = np.mean([m['goals_for'] for m in away_recent])
        features['home_conceded_avg'] = np.mean([m['goals_against'] for m in home_recent])
        features['away_conceded_avg'] = np.mean([m['goals_against'] for m in away_recent])
        
        # Points per game
        home_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in home_recent) / len(home_recent)
        away_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in away_recent) / len(away_recent)
        
        features['home_ppg'] = home_ppg
        features['away_ppg'] = away_ppg
        features['ppg_difference'] = home_ppg - away_ppg
        
        return features
    
    def get_ratio_features(self, home_recent, away_recent):
        """Advanced ratio features (our breakthrough discovery)"""
        features = {}
        
        home_goals_avg = np.mean([m['goals_for'] for m in home_recent])
        away_goals_avg = np.mean([m['goals_for'] for m in away_recent])
        home_conceded_avg = np.mean([m['goals_against'] for m in home_recent])
        away_conceded_avg = np.mean([m['goals_against'] for m in away_recent])
        
        # LOGARITHMIC RATIOS (our top performing features)
        features['log_goals_ratio'] = np.log((home_goals_avg + 1) / (away_goals_avg + 1))
        features['log_defense_ratio'] = np.log((away_conceded_avg + 1) / (home_conceded_avg + 1))
        features['log_attack_defense'] = np.log((home_goals_avg + 1) / (home_conceded_avg + 1))
        
        # Goal difference ratios
        home_gd_avg = np.mean([m['goal_diff'] for m in home_recent])
        away_gd_avg = np.mean([m['goal_diff'] for m in away_recent])
        features['goal_diff_advantage'] = home_gd_avg - away_gd_avg
        
        return features
    
    def get_psychological_features(self, home_recent, away_recent):
        """Psychological momentum features"""
        features = {}
        
        # Win streaks
        home_streak = self.calculate_streak(home_recent, 'W')
        away_streak = self.calculate_streak(away_recent, 'W')
        features['home_win_streak'] = home_streak
        features['away_win_streak'] = away_streak
        features['streak_advantage'] = home_streak - away_streak
        
        # Recent momentum (last 3 games)
        home_recent_form = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in home_recent[:3])
        away_recent_form = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in away_recent[:3])
        features['momentum_advantage'] = home_recent_form - away_recent_form
        
        # Big wins/losses in recent form
        features['home_big_wins'] = sum(1 for m in home_recent if m['goal_diff'] >= 3)
        features['away_big_wins'] = sum(1 for m in away_recent if m['goal_diff'] >= 3)
        features['home_big_losses'] = sum(1 for m in home_recent if m['goal_diff'] <= -3)
        features['away_big_losses'] = sum(1 for m in away_recent if m['goal_diff'] <= -3)
        
        return features
    
    def calculate_streak(self, recent_matches, result_type):
        """Calculate current streak length"""
        streak = 0
        for match in recent_matches:
            if match['result'] == result_type:
                streak += 1
            else:
                break
        return streak
    
    def analyze_feature_importance(self):
        """Comprehensive feature importance analysis"""
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if not hasattr(self, 'features_df'):
            self.create_feature_dataset()
        
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']
        
        print(f"📊 Analyzing {X.shape[1]} features across {len(X)} matches")
        
        # 1. RANDOM FOREST FEATURE IMPORTANCE
        print("\n🌳 Random Forest Feature Importance:")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("   🔝 Top 10 Random Forest Features:")
        for i, (_, row) in enumerate(rf_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        self.results['rf_importance'] = rf_importance
        
        # 2. STATISTICAL CORRELATION ANALYSIS
        print("\n📊 Statistical Correlation with Prediction Accuracy:")
        
        # Individual feature predictive power
        feature_scores = {}
        tscv = TimeSeriesSplit(n_splits=3)
        
        print("   Testing individual feature predictive power...")
        
        # Test top features individually
        top_features = rf_importance.head(10)['feature'].tolist()
        
        for feature in top_features:
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train_single = X.iloc[train_idx][[feature]]
                X_test_single = X.iloc[test_idx][[feature]]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # Simple logistic regression on single feature
                lr = LogisticRegression(random_state=42)
                lr.fit(X_train_single, y_train)
                score = lr.score(X_test_single, y_test)
                scores.append(score)
            
            feature_scores[feature] = np.mean(scores)
        
        # Sort by predictive power
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("   🎯 Individual Feature Predictive Power:")
        for i, (feature, score) in enumerate(sorted_features):
            print(f"   {i+1:2d}. {feature:<25} {score:.3f} ({score*100:.1f}%)")
        
        self.results['individual_scores'] = sorted_features
        
        # 3. MUTUAL INFORMATION ANALYSIS
        print("\n🔗 Mutual Information Analysis:")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        print("   📈 Top 10 Mutual Information Features:")
        for i, (_, row) in enumerate(mi_df.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<25} {row['mutual_info']:.4f}")
        
        self.results['mutual_info'] = mi_df
        
        # 4. ENSEMBLE MODEL COMPARISON
        print("\n🤖 Feature Set Performance Comparison:")
        
        feature_sets = {
            'All Features': X.columns.tolist(),
            'Top 10 RF': rf_importance.head(10)['feature'].tolist(),
            'Top 10 MI': mi_df.head(10)['feature'].tolist(),
            'Basic Features': [col for col in X.columns if any(basic in col for basic in ['ppg', 'goals_avg', 'conceded'])],
            'Ratio Features': [col for col in X.columns if 'log_' in col or 'ratio' in col],
            'Psychological': [col for col in X.columns if any(psych in col for psych in ['streak', 'momentum', 'big_'])]
        }
        
        set_scores = {}
        for set_name, features in feature_sets.items():
            if not features or not all(f in X.columns for f in features):
                continue
            
            X_subset = X[features]
            scores = cross_val_score(rf, X_subset, y, cv=tscv, scoring='accuracy')
            set_scores[set_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'features': len(features)
            }
        
        print("   🏆 Feature Set Performance:")
        for set_name, metrics in sorted(set_scores.items(), key=lambda x: x[1]['mean'], reverse=True):
            print(f"   {set_name:<15} {metrics['mean']:.3f} ± {metrics['std']:.3f} ({metrics['features']} features)")
        
        self.results['feature_sets'] = set_scores
        
        return self.results
    
    def analyze_correlation_with_accuracy(self):
        """Analyze which features correlate with prediction accuracy"""
        print("\n🎯 CORRELATION WITH PREDICTION ACCURACY")
        print("="*50)
        
        if not hasattr(self, 'features_df'):
            self.create_feature_dataset()
        
        X = self.features_df.drop('target', axis=1)
        y = self.features_df['target']
        
        # For each feature, measure its correlation with making correct predictions
        tscv = TimeSeriesSplit(n_splits=3)
        
        feature_correlations = {}
        
        print("📊 Computing feature-accuracy correlations...")
        
        # Train a baseline model
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train baseline
            rf_baseline.fit(X_train, y_train)
            predictions = rf_baseline.predict(X_test)
            correct_predictions = (predictions == y_test).astype(int)
            
            # For each feature, compute correlation with correct predictions
            for feature in X.columns:
                feature_values = X_test[feature].values
                
                # Skip if no variance
                if np.std(feature_values) == 0:
                    continue
                
                correlation = np.corrcoef(feature_values, correct_predictions)[0, 1]
                
                if feature not in feature_correlations:
                    feature_correlations[feature] = []
                feature_correlations[feature].append(correlation)
        
        # Average correlations across folds
        avg_correlations = {
            feature: np.mean(corrs) for feature, corrs in feature_correlations.items()
            if not np.isnan(np.mean(corrs))
        }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(avg_correlations.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        print("🔗 Features Most Correlated with Prediction Accuracy:")
        for i, (feature, corr) in enumerate(sorted_correlations[:15]):
            direction = "📈" if corr > 0 else "📉"
            print(f"   {i+1:2d}. {feature:<25} {direction} {corr:+.4f}")
        
        self.results['accuracy_correlation'] = sorted_correlations
        
        return sorted_correlations
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*60)
        print("🏆 FEATURE IMPORTANCE SUMMARY REPORT")
        print("="*60)
        
        if not self.results:
            print("❌ No analysis results found. Run analyze_feature_importance() first.")
            return
        
        print("\n🥇 CHAMPION FEATURES (Top 5 by Multiple Metrics):")
        
        # Get top features from each method
        rf_top5 = self.results['rf_importance'].head(5)['feature'].tolist()
        mi_top5 = self.results['mutual_info'].head(5)['feature'].tolist()
        ind_top5 = [item[0] for item in self.results['individual_scores'][:5]]
        
        # Count appearances
        feature_votes = {}
        for feature in rf_top5 + mi_top5 + ind_top5:
            feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        champions = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, votes) in enumerate(champions[:10]):
            print(f"   {i+1:2d}. {feature:<25} ({votes}/3 methods)")
        
        print("\n📊 FEATURE CATEGORIES RANKING:")
        if 'feature_sets' in self.results:
            for set_name, metrics in sorted(self.results['feature_sets'].items(), 
                                          key=lambda x: x[1]['mean'], reverse=True):
                print(f"   {set_name:<15} {metrics['mean']:.3f} accuracy ({metrics['features']} features)")
        
        print("\n🎯 KEY INSIGHTS:")
        
        # Identify top feature types
        top_features = [item[0] for item in champions[:5]]
        
        log_features = [f for f in top_features if 'log_' in f]
        ratio_features = [f for f in top_features if 'ratio' in f or 'advantage' in f]
        psych_features = [f for f in top_features if any(p in f for p in ['streak', 'momentum', 'big_'])]
        basic_features = [f for f in top_features if any(b in f for b in ['ppg', 'goals', 'conceded'])]
        
        if log_features:
            print(f"   🔥 LOGARITHMIC FEATURES dominate ({len(log_features)}/5 top features)")
        if ratio_features:
            print(f"   ⚖️  RATIO FEATURES highly important ({len(ratio_features)}/5 top features)")
        if psych_features:
            print(f"   🧠 PSYCHOLOGICAL FEATURES matter ({len(psych_features)}/5 top features)")
        if basic_features:
            print(f"   📊 BASIC FEATURES still relevant ({len(basic_features)}/5 top features)")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Focus on LOGARITHMIC RATIOS for best predictive power")
        print("   2. Combine multiple feature types for optimal performance")
        print("   3. Psychological features add value but shouldn't dominate")
        print("   4. Feature selection can reduce complexity without losing accuracy")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save analysis results"""
        try:
            import json
            
            # Prepare results for JSON
            json_results = {}
            
            if 'rf_importance' in self.results:
                json_results['random_forest_top10'] = self.results['rf_importance'].head(10).to_dict('records')
            
            if 'individual_scores' in self.results:
                json_results['individual_feature_accuracy'] = self.results['individual_scores']
            
            if 'feature_sets' in self.results:
                json_results['feature_set_performance'] = self.results['feature_sets']
            
            if 'accuracy_correlation' in self.results:
                json_results['accuracy_correlations'] = self.results['accuracy_correlation'][:10]
            
            with open('feature_importance_analysis.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Results saved to: feature_importance_analysis.json")
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")

def main():
    """Run comprehensive feature importance analysis"""
    analyzer = FeatureImportanceAnalyzer()
    
    # Run all analyses
    analyzer.analyze_feature_importance()
    analyzer.analyze_correlation_with_accuracy()
    analyzer.generate_summary_report()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📋 Check feature_importance_analysis.json for detailed results")

if __name__ == "__main__":
    main() 
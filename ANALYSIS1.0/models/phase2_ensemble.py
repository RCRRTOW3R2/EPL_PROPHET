#!/usr/bin/env python3
"""
EPL PROPHET - Phase 2: Multi-Timeframe Ensemble
Advanced ensemble with short, medium, long-term models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

class MultiTimeframeEnsemble:
    """Multi-timeframe ensemble system."""
    
    def __init__(self):
        self.timeframe_models = {
            'short_term': {
                'focus': 'ema_short',
                'description': 'Recent form & momentum (5-match focus)',
                'model': RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
            },
            'medium_term': {
                'focus': 'ema_medium', 
                'description': 'Balanced form analysis (10-match focus)',
                'model': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            },
            'long_term': {
                'focus': 'ema_long',
                'description': 'Underlying strength (20-match focus)', 
                'model': GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=10, random_state=42)
            }
        }
    
    def create_timeframe_features(self, df):
        """Create specialized feature sets for each timeframe."""
        
        print("🔄 Creating Timeframe-Specific Features...")
        
        base_features = ['match_id', 'date', 'home_team', 'away_team', 'actual_result']
        timeframe_datasets = {}
        
        # Short-term features (recent form emphasis)
        short_features = base_features.copy()
        for col in df.columns:
            if any(keyword in col for keyword in ['ema_short', 'momentum', 'macd', 'trend_']):
                short_features.append(col)
        
        timeframe_datasets['short_term'] = df[short_features].copy()
        print(f"   Short-term features: {len(short_features) - 5}")
        
        # Medium-term features (balanced approach)
        medium_features = base_features.copy()
        for col in df.columns:
            if any(keyword in col for keyword in ['ema_medium', 'ema_short', 'momentum', 'advantage']):
                medium_features.append(col)
        
        timeframe_datasets['medium_term'] = df[medium_features].copy()
        print(f"   Medium-term features: {len(medium_features) - 5}")
        
        # Long-term features (underlying strength)
        long_features = base_features.copy()
        for col in df.columns:
            if any(keyword in col for keyword in ['ema_long', 'ema_medium', 'advantage']):
                long_features.append(col)
        
        timeframe_datasets['long_term'] = df[long_features].copy()
        print(f"   Long-term features: {len(long_features) - 5}")
        
        return timeframe_datasets
    
    def add_advanced_features(self, df):
        """Add advanced EMA crossover and convergence features."""
        
        print("📈 Adding Advanced EMA Features...")
        
        df_enhanced = df.copy()
        crossover_count = 0
        ratio_count = 0
        
        # EMA crossovers and convergence
        for team_type in ['home', 'away']:
            for metric in ['goals', 'points']:
                short_col = f'{team_type}_{metric}_ema_short'
                medium_col = f'{team_type}_{metric}_ema_medium' 
                long_col = f'{team_type}_{metric}_ema_long'
                
                if all(col in df.columns for col in [short_col, medium_col, long_col]):
                    # Bullish crossover (short > medium > long)
                    bullish_signal = f'{team_type}_{metric}_bullish'
                    df_enhanced[bullish_signal] = (
                        (df[short_col] > df[medium_col]) & 
                        (df[medium_col] > df[long_col])
                    ).astype(int)
                    crossover_count += 1
                    
                    # Bearish crossover (short < medium < long)
                    bearish_signal = f'{team_type}_{metric}_bearish'
                    df_enhanced[bearish_signal] = (
                        (df[short_col] < df[medium_col]) & 
                        (df[medium_col] < df[long_col])
                    ).astype(int)
                    crossover_count += 1
                    
                    # EMA convergence
                    convergence = f'{team_type}_{metric}_convergence'
                    ema_spread = df[long_col] - df[short_col]
                    df_enhanced[convergence] = 1.0 / (1.0 + abs(ema_spread))
                    crossover_count += 1
        
        # Cross-team ratios
        for metric in ['goals', 'points']:
            for timeframe in ['short', 'medium', 'long']:
                home_col = f'home_{metric}_ema_{timeframe}'
                away_col = f'away_{metric}_ema_{timeframe}'
                
                if home_col in df.columns and away_col in df.columns:
                    ratio_col = f'{metric}_ema_{timeframe}_ratio'
                    df_enhanced[ratio_col] = df[home_col] / (df[away_col] + 0.01)
                    ratio_count += 1
        
        print(f"   Added {crossover_count} crossover features")
        print(f"   Added {ratio_count} ratio features")
        
        return df_enhanced
    
    def train_timeframe_models(self, timeframe_datasets):
        """Train specialized models for each timeframe."""
        
        print("\n🤖 Training Multi-Timeframe Models...")
        
        trained_models = {}
        
        for timeframe, dataset in timeframe_datasets.items():
            print(f"\n   Training {timeframe} model...")
            print(f"   {self.timeframe_models[timeframe]['description']}")
            
            # Prepare data
            df_clean = dataset[dataset['actual_result'].notna()].copy()
            
            feature_cols = [col for col in df_clean.columns if col not in 
                           ['match_id', 'date', 'home_team', 'away_team', 'actual_result']]
            
            X = df_clean[feature_cols].fillna(0)
            le = LabelEncoder()
            y = le.fit_transform(df_clean['actual_result'])
            
            # Chronological split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = self.timeframe_models[timeframe]['model']
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"      Features: {len(feature_cols)}")
            print(f"      Accuracy: {accuracy:.3f}")
            
            # Store model
            trained_models[timeframe] = {
                'model': model,
                'features': feature_cols,
                'label_encoder': le,
                'accuracy': accuracy,
                'test_size': len(X_test)
            }
            
            # Top features
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_))
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print(f"      Top 3 features:")
                for feat, score in sorted_features[:3]:
                    print(f"        {feat}: {score:.4f}")
        
        return trained_models
    
    def create_dynamic_ensemble(self, trained_models, X_test_dict):
        """Create dynamic ensemble with adaptive weighting."""
        
        print("\n🎯 Creating Dynamic Ensemble...")
        
        # Get predictions from each model
        timeframe_predictions = {}
        timeframe_probabilities = {}
        
        for timeframe, model_data in trained_models.items():
            model = model_data['model']
            X_test = X_test_dict[timeframe]
            
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            
            timeframe_predictions[timeframe] = pred
            timeframe_probabilities[timeframe] = prob
            
            print(f"   {timeframe}: {len(pred)} predictions")
        
        # Dynamic ensemble weighting
        ensemble_predictions = []
        confidence_scores = []
        
        for i in range(len(list(timeframe_predictions.values())[0])):
            # Get all predictions for this match
            match_probabilities = {tf: probs[i] for tf, probs in timeframe_probabilities.items()}
            
            # Calculate weights based on confidence and accuracy
            weights = {}
            total_weight = 0
            
            for timeframe, prob in match_probabilities.items():
                confidence = max(prob)  # How sure the model is
                base_accuracy = trained_models[timeframe]['accuracy']
                weight = confidence * base_accuracy
                weights[timeframe] = weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for timeframe in weights:
                    weights[timeframe] /= total_weight
            else:
                for timeframe in weights:
                    weights[timeframe] = 1.0 / len(weights)
            
            # Weighted ensemble prediction
            weighted_probs = np.zeros(3)  # H, D, A
            for timeframe, weight in weights.items():
                weighted_probs += weight * match_probabilities[timeframe]
            
            final_prediction = np.argmax(weighted_probs)
            ensemble_predictions.append(final_prediction)
            confidence_scores.append(max(weighted_probs))
        
        print(f"   Ensemble complete: {len(ensemble_predictions)} predictions")
        print(f"   Avg confidence: {np.mean(confidence_scores):.3f}")
        
        return np.array(ensemble_predictions), {
            'timeframe_predictions': timeframe_predictions,
            'confidence_scores': confidence_scores
        }
    
    def evaluate_performance(self, y_true, ensemble_pred, timeframe_preds, trained_models):
        """Evaluate ensemble vs individual performance."""
        
        print("\n📈 Performance Evaluation...")
        
        results = {}
        
        # Individual timeframe results
        for timeframe, predictions in timeframe_preds.items():
            accuracy = accuracy_score(y_true, predictions)
            results[f'{timeframe}'] = accuracy
            print(f"   {timeframe:<15}: {accuracy:.3f}")
        
        # Ensemble result
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        results['ensemble'] = ensemble_accuracy
        print(f"   {'ensemble':<15}: {ensemble_accuracy:.3f}")
        
        # Performance summary
        best_individual = max([results[tf] for tf in ['short_term', 'medium_term', 'long_term']])
        improvement = ensemble_accuracy - best_individual
        
        print(f"\n🏆 Summary:")
        print(f"   Best individual: {best_individual:.3f}")
        print(f"   Ensemble:        {ensemble_accuracy:.3f}")
        print(f"   Improvement:     {improvement:+.3f}")
        
        if improvement > 0:
            print("   ✅ Ensemble wins!")
        else:
            print("   ⚠️  Individual model better")
        
        results['improvement'] = improvement
        
        return results

def run_phase2():
    """Run Phase 2 multi-timeframe ensemble."""
    
    print("🚀 EPL PROPHET - PHASE 2: MULTI-TIMEFRAME ENSEMBLE")
    print("=" * 65)
    print("Short + Medium + Long term models with dynamic weighting!")
    
    # Load data
    print("📊 Loading Recency Data...")
    df = pd.read_csv("../outputs/recency_weighted_stock_features.csv")
    print(f"   {len(df)} matches loaded")
    
    # Initialize ensemble
    ensemble = MultiTimeframeEnsemble()
    
    # Add advanced features
    df_enhanced = ensemble.add_advanced_features(df)
    print(f"   Enhanced to {len(df_enhanced.columns)} total features")
    
    # Create timeframe datasets
    timeframe_datasets = ensemble.create_timeframe_features(df_enhanced)
    
    # Train models
    trained_models = ensemble.train_timeframe_models(timeframe_datasets)
    
    # Prepare test data
    X_test_dict = {}
    y_test = None
    
    for timeframe, dataset in timeframe_datasets.items():
        df_clean = dataset[dataset['actual_result'].notna()].copy()
        feature_cols = trained_models[timeframe]['features']
        
        X = df_clean[feature_cols].fillna(0)
        le = trained_models[timeframe]['label_encoder']
        y = le.transform(df_clean['actual_result'])
        
        split_idx = int(len(X) * 0.8)
        X_test_dict[timeframe] = X[split_idx:]
        
        if y_test is None:
            y_test = y[split_idx:]
    
    # Create ensemble
    ensemble_pred, ensemble_details = ensemble.create_dynamic_ensemble(trained_models, X_test_dict)
    
    # Evaluate
    results = ensemble.evaluate_performance(
        y_test, ensemble_pred, 
        ensemble_details['timeframe_predictions'], 
        trained_models
    )
    
    # Save models
    print("\n💾 Saving Phase 2 Models...")
    for timeframe, model_data in trained_models.items():
        joblib.dump(model_data, f"../outputs/phase2_{timeframe}.joblib")
        print(f"   Saved {timeframe}")
    
    # Save results
    results_df = pd.DataFrame([results]).T
    results_df.to_csv("../outputs/phase2_results.csv")
    
    # Save enhanced features
    df_enhanced.to_csv("../outputs/phase2_enhanced_features.csv", index=False)
    
    print(f"\n🎯 PHASE 2 COMPLETE!")
    print(f"   🏆 Ensemble: {results['ensemble']:.3f}")
    print(f"   📈 Best individual: {max([results[tf] for tf in ['short_term', 'medium_term', 'long_term']]):.3f}")
    print(f"   ⚡ Improvement: {results['improvement']:+.3f}")
    print(f"   🤖 Models: 3 timeframes")
    
    print(f"\n✨ MULTI-TIMEFRAME ACHIEVEMENTS:")
    print(f"   📈 Short-term: Momentum & recent form")
    print(f"   🎯 Medium-term: Balanced analysis") 
    print(f"   📊 Long-term: Underlying strength")
    print(f"   🧠 Dynamic weighting: Adaptive ensemble")
    
    # Check target
    target = 0.55
    if results['ensemble'] >= target:
        print(f"\n🎉 TARGET ACHIEVED: {results['ensemble']:.3f} >= {target}")
        print("🏆 EPL Prophet is now world-class!")
    else:
        print(f"\n🎯 Progress: {results['ensemble']:.3f} toward {target} target")
        print("🚀 Excellent foundation for optimization!")
    
    return results

if __name__ == "__main__":
    run_phase2()

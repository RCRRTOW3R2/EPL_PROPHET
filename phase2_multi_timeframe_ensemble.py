#!/usr/bin/env python3
"""
EPL PROPHET - Phase 2: Multi-Timeframe Ensemble Models
====================================================

Multi-timeframe ensemble approach:
- Short-term model (5-match EMA focus) - Captures immediate form
- Medium-term model (10-match EMA focus) - Balances recent vs historical  
- Long-term model (20-match EMA focus) - Captures underlying strength
- Dynamic ensemble weighting based on form stability
- Advanced EMA crossovers and momentum indicators

This builds on Phase 1's success to achieve 55%+ accuracy!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

warnings.filterwarnings('ignore')

class MultiTimeframeEnsemble:
    """Multi-timeframe ensemble with short, medium, long-term models."""
    
    def __init__(self):
        # Timeframe-specific model configurations
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
        
        # Meta-model for ensemble weighting
        self.meta_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        
    def create_timeframe_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create specialized feature sets for each timeframe."""
        
        print("🔄 Creating Timeframe-Specific Features...")
        
        # Base features for all timeframes
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
    
    def calculate_advanced_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced EMA indicators like crossovers and convergence."""
        
        print("📈 Calculating Advanced EMA Indicators...")
        
        df_enhanced = df.copy()
        
        # EMA Crossovers (bullish/bearish signals)
        ema_crossover_features = []
        
        for team_type in ['home', 'away']:
            for metric in ['goals', 'points']:
                short_col = f'{team_type}_{metric}_ema_short'
                medium_col = f'{team_type}_{metric}_ema_medium' 
                long_col = f'{team_type}_{metric}_ema_long'
                
                if all(col in df.columns for col in [short_col, medium_col, long_col]):
                    # Bullish crossover (short > medium > long)
                    bullish_signal = f'{team_type}_{metric}_bullish_crossover'
                    df_enhanced[bullish_signal] = (
                        (df[short_col] > df[medium_col]) & 
                        (df[medium_col] > df[long_col])
                    ).astype(int)
                    ema_crossover_features.append(bullish_signal)
                    
                    # Bearish crossover (short < medium < long)
                    bearish_signal = f'{team_type}_{metric}_bearish_crossover'
                    df_enhanced[bearish_signal] = (
                        (df[short_col] < df[medium_col]) & 
                        (df[medium_col] < df[long_col])
                    ).astype(int)
                    ema_crossover_features.append(bearish_signal)
                    
                    # EMA convergence (how close EMAs are - indicates stability)
                    convergence = f'{team_type}_{metric}_ema_convergence'
                    ema_spread = df[long_col] - df[short_col]
                    df_enhanced[convergence] = 1.0 / (1.0 + abs(ema_spread))
                    ema_crossover_features.append(convergence)
        
        # Cross-team EMA comparisons
        comparative_features = []
        for metric in ['goals', 'points']:
            for timeframe in ['short', 'medium', 'long']:
                home_col = f'home_{metric}_ema_{timeframe}'
                away_col = f'away_{metric}_ema_{timeframe}'
                
                if home_col in df.columns and away_col in df.columns:
                    # Relative advantage ratio
                    ratio_col = f'{metric}_ema_{timeframe}_ratio'
                    df_enhanced[ratio_col] = df[home_col] / (df[away_col] + 0.01)  # Avoid division by zero
                    comparative_features.append(ratio_col)
                    
                    # Form momentum difference
                    momentum_diff = f'{metric}_ema_{timeframe}_momentum_diff'
                    home_momentum = df.get(f'home_{metric}_momentum', 0)
                    away_momentum = df.get(f'away_{metric}_momentum', 0)
                    df_enhanced[momentum_diff] = home_momentum - away_momentum
                    comparative_features.append(momentum_diff)
        
        print(f"   Added {len(ema_crossover_features)} crossover features")
        print(f"   Added {len(comparative_features)} comparative features")
        
        return df_enhanced
    
    def calculate_form_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form volatility indicators (Bollinger Band style)."""
        
        print("📊 Calculating Form Volatility Indicators...")
        
        df_enhanced = df.copy()
        volatility_features = []
        
        for team_type in ['home', 'away']:
            for metric in ['goals', 'points']:
                ema_col = f'{team_type}_{metric}_ema_medium'
                current_col = f'{team_type}_{metric}_current_value'
                
                if ema_col in df.columns:
                    # Form volatility (how much current form deviates from EMA)
                    volatility_col = f'{team_type}_{metric}_form_volatility'
                    if current_col in df.columns:
                        df_enhanced[volatility_col] = abs(df[current_col] - df[ema_col]) / (df[ema_col] + 0.01)
                    else:
                        df_enhanced[volatility_col] = 0.1  # Default volatility
                    volatility_features.append(volatility_col)
                    
                    # Form strength (RSI-style indicator)
                    strength_col = f'{team_type}_{metric}_form_strength'
                    short_ema = df.get(f'{team_type}_{metric}_ema_short', df[ema_col])
                    long_ema = df.get(f'{team_type}_{metric}_ema_long', df[ema_col])
                    
                    # Normalize to 0-1 scale
                    if not (short_ema == long_ema).all():
                        strength = (short_ema - long_ema) / (abs(short_ema + long_ema) + 0.01)
                        df_enhanced[strength_col] = 0.5 + strength * 0.5  # Scale to 0-1
                    else:
                        df_enhanced[strength_col] = 0.5  # Neutral
                    volatility_features.append(strength_col)
        
        print(f"   Added {len(volatility_features)} volatility features")
        
        return df_enhanced
    
    def train_timeframe_models(self, timeframe_datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Train specialized models for each timeframe."""
        
        print("\n🤖 Training Multi-Timeframe Models...")
        
        trained_models = {}
        timeframe_results = {}
        
        for timeframe, dataset in timeframe_datasets.items():
            print(f"\n   Training {timeframe} model...")
            print(f"   Focus: {self.timeframe_models[timeframe]['description']}")
            
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
            
            print(f"      Features used: {len(feature_cols)}")
            print(f"      Test accuracy: {accuracy:.3f}")
            
            # Store model and results
            trained_models[timeframe] = {
                'model': model,
                'features': feature_cols,
                'label_encoder': le,
                'accuracy': accuracy
            }
            
            timeframe_results[timeframe] = accuracy
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_cols, model.feature_importances_))
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print(f"      Top 3 features:")
                for feat, score in sorted_features[:3]:
                    print(f"        {feat}: {score:.4f}")
        
        return trained_models, timeframe_results
    
    def create_dynamic_ensemble(self, trained_models: Dict, X_test_dict: Dict) -> Tuple[np.ndarray, Dict]:
        """Create dynamic ensemble with adaptive weighting."""
        
        print("\n🎯 Creating Dynamic Multi-Timeframe Ensemble...")
        
        # Get predictions from each timeframe model
        timeframe_predictions = {}
        timeframe_probabilities = {}
        
        for timeframe, model_data in trained_models.items():
            model = model_data['model']
            X_test = X_test_dict[timeframe]
            
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            
            timeframe_predictions[timeframe] = pred
            timeframe_probabilities[timeframe] = prob
            
            print(f"   {timeframe} predictions: {len(pred)} matches")
        
        # Dynamic weighting based on recent performance and form stability
        ensemble_predictions = []
        confidence_scores = []
        
        for i in range(len(list(timeframe_predictions.values())[0])):
            # Get predictions from all timeframes for this match
            match_predictions = {tf: preds[i] for tf, preds in timeframe_predictions.items()}
            match_probabilities = {tf: probs[i] for tf, probs in timeframe_probabilities.items()}
            
            # Calculate dynamic weights based on prediction confidence
            weights = {}
            total_confidence = 0
            
            for timeframe, prob in match_probabilities.items():
                # Confidence = max probability (how sure the model is)
                confidence = max(prob)
                # Weight by accuracy and confidence
                base_accuracy = trained_models[timeframe]['accuracy']
                weights[timeframe] = confidence * base_accuracy
                total_confidence += weights[timeframe]
            
            # Normalize weights
            if total_confidence > 0:
                for timeframe in weights:
                    weights[timeframe] /= total_confidence
            else:
                # Equal weights if no confidence
                for timeframe in weights:
                    weights[timeframe] = 1.0 / len(weights)
            
            # Weighted ensemble prediction
            weighted_probs = np.zeros(3)  # For H, D, A
            for timeframe, weight in weights.items():
                weighted_probs += weight * match_probabilities[timeframe]
            
            # Final prediction
            final_prediction = np.argmax(weighted_probs)
            ensemble_predictions.append(final_prediction)
            confidence_scores.append(max(weighted_probs))
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        print(f"   Dynamic ensemble complete: {len(ensemble_predictions)} predictions")
        print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
        
        return ensemble_predictions, {
            'timeframe_predictions': timeframe_predictions,
            'confidence_scores': confidence_scores,
            'probabilities': timeframe_probabilities
        }
    
    def evaluate_ensemble_performance(self, y_true: np.ndarray, ensemble_pred: np.ndarray, 
                                    timeframe_preds: Dict, trained_models: Dict) -> Dict:
        """Evaluate ensemble vs individual timeframe performance."""
        
        print("\n📈 Multi-Timeframe Ensemble Evaluation...")
        
        results = {}
        
        # Individual timeframe accuracies
        for timeframe, predictions in timeframe_preds.items():
            accuracy = accuracy_score(y_true, predictions)
            results[f'{timeframe}_accuracy'] = accuracy
            print(f"   {timeframe:<15}: {accuracy:.3f}")
        
        # Ensemble accuracy
        ensemble_accuracy = accuracy_score(y_true, ensemble_pred)
        results['ensemble_accuracy'] = ensemble_accuracy
        print(f"   {'ensemble':<15}: {ensemble_accuracy:.3f}")
        
        # Best individual vs ensemble
        best_individual = max([results[f'{tf}_accuracy'] for tf in timeframe_preds.keys()])
        improvement = ensemble_accuracy - best_individual
        
        print(f"\n🏆 Performance Summary:")
        print(f"   Best individual: {best_individual:.3f}")
        print(f"   Ensemble:        {ensemble_accuracy:.3f}")
        print(f"   Improvement:     {improvement:+.3f}")
        
        if improvement > 0:
            print("   ✅ Ensemble outperforms individual models!")
        else:
            print("   ⚠️  Individual model performs better")
        
        results['improvement'] = improvement
        
        return results


def run_phase2_multi_timeframe_ensemble():
    """Run complete Phase 2 multi-timeframe ensemble."""
    
    print("🚀 EPL PROPHET - PHASE 2: MULTI-TIMEFRAME ENSEMBLE")
    print("=" * 65)
    print("Short-term + Medium-term + Long-term models with dynamic weighting!")
    
    # Load recency-weighted data
    print("📊 Loading Enhanced Recency Data...")
    df = pd.read_csv("../outputs/recency_weighted_stock_features.csv")
    
    # Initialize ensemble system
    ensemble = MultiTimeframeEnsemble()
    
    # Add advanced EMA features
    df_enhanced = ensemble.calculate_advanced_ema_features(df)
    df_enhanced = ensemble.calculate_form_volatility(df_enhanced)
    
    # Create timeframe-specific datasets
    timeframe_datasets = ensemble.create_timeframe_features(df_enhanced)
    
    # Train timeframe models
    trained_models, timeframe_results = ensemble.train_timeframe_models(timeframe_datasets)
    
    # Prepare test data for ensemble
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
    
    # Create dynamic ensemble
    ensemble_pred, ensemble_details = ensemble.create_dynamic_ensemble(trained_models, X_test_dict)
    
    # Evaluate performance
    results = ensemble.evaluate_ensemble_performance(
        y_test, ensemble_pred, 
        ensemble_details['timeframe_predictions'], 
        trained_models
    )
    
    # Save models and results
    print("\n💾 Saving Phase 2 Models...")
    
    # Save individual timeframe models
    for timeframe, model_data in trained_models.items():
        joblib.dump(model_data, f"../outputs/phase2_{timeframe}_model.joblib")
        print(f"   Saved {timeframe} model")
    
    # Save ensemble results
    results_df = pd.DataFrame([results]).T
    results_df.to_csv("../outputs/phase2_ensemble_results.csv")
    
    # Save enhanced features for future use
    df_enhanced.to_csv("../outputs/phase2_enhanced_features.csv", index=False)
    
    print(f"\n🎯 PHASE 2 COMPLETE!")
    print(f"   🏆 Ensemble accuracy: {results['ensemble_accuracy']:.3f}")
    print(f"   📈 Best individual: {max([results[f'{tf}_accuracy'] for tf in ['short_term', 'medium_term', 'long_term']]):.3f}")
    print(f"   ⚡ Improvement: {results['improvement']:+.3f}")
    print(f"   🤖 Timeframe models: 3")
    print(f"   📊 Advanced features: EMA crossovers, volatility, convergence")
    
    print(f"\n✨ MULTI-TIMEFRAME BREAKTHROUGH:")
    print(f"   📈 Short-term captures momentum")
    print(f"   🎯 Medium-term balances recency vs stability") 
    print(f"   📊 Long-term provides foundation")
    print(f"   🧠 Dynamic weighting optimizes predictions")
    
    target_accuracy = 0.55
    if results['ensemble_accuracy'] >= target_accuracy:
        print(f"\n🎉 TARGET ACHIEVED: {results['ensemble_accuracy']:.3f} >= {target_accuracy}")
        print("🏆 EPL Prophet is now a world-class prediction system!")
    else:
        print(f"\n🎯 Progress toward 55% target: {results['ensemble_accuracy']:.3f}")
        print("🚀 Ready for further optimization!")
    
    return trained_models, results


if __name__ == "__main__":
    run_phase2_multi_timeframe_ensemble() 
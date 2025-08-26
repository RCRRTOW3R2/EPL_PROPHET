#!/usr/bin/env python3
"""
EPL PROPHET - Phase 3: Comprehensive Optimization
===============================================

The FINAL PUSH to 55%+ accuracy:

Priority 1: HYPERPARAMETER OPTIMIZATION
- Grid search for Random Forest optimization
- XGBoost fine-tuning with Bayesian optimization
- Feature selection to remove noise

Priority 2: MORE RATIO FEATURES  
- Shot ratios (shots, shots on target)
- Defensive ratios (goals against, clean sheets)
- Card ratios (discipline comparison)

Priority 3: ADVANCED ENSEMBLE
- Stacking models instead of simple weighting
- Meta-learning to predict optimal model per match
- Non-linear combinations of predictions

This is where we break through 55%!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

class Phase3Optimizer:
    """Comprehensive optimization for EPL Prophet Phase 3."""
    
    def __init__(self):
        # Hyperparameter grids
        self.rf_param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [12, 15, 18, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        self.xgb_param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
    def create_expanded_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive ratio features based on Phase 2 insights."""
        
        print("🔥 Creating Expanded Ratio Features...")
        
        df_enhanced = df.copy()
        new_features = []
        
        # 1. SHOT RATIOS (if shot data available)
        shot_metrics = []
        for col in df.columns:
            if 'shot' in col.lower() and 'ema' in col:
                shot_metrics.append(col)
        
        print(f"   Found {len(shot_metrics)} shot-related EMA features")
        
        # Create shot ratios
        shot_ratios = 0
        for timeframe in ['short', 'medium', 'long']:
            home_shots = f'home_shots_ema_{timeframe}' if f'home_shots_ema_{timeframe}' in df.columns else None
            away_shots = f'away_shots_ema_{timeframe}' if f'away_shots_ema_{timeframe}' in df.columns else None
            
            if home_shots and away_shots:
                ratio_col = f'shots_ema_{timeframe}_ratio'
                df_enhanced[ratio_col] = df[home_shots] / (df[away_shots] + 0.01)
                new_features.append(ratio_col)
                shot_ratios += 1
        
        # 2. DEFENSIVE RATIOS
        defensive_ratios = 0
        for timeframe in ['short', 'medium', 'long']:
            home_ga = f'home_goals_against_ema_{timeframe}'
            away_ga = f'away_goals_against_ema_{timeframe}'
            
            if home_ga in df.columns and away_ga in df.columns:
                # Goals against ratio (lower is better for home team)
                ga_ratio_col = f'goals_against_ema_{timeframe}_ratio'
                df_enhanced[ga_ratio_col] = df[away_ga] / (df[home_ga] + 0.01)  # Flipped for home advantage
                new_features.append(ga_ratio_col)
                defensive_ratios += 1
                
                # Defensive strength ratio
                def_strength_col = f'defensive_strength_ema_{timeframe}_ratio'
                home_def = 1.0 / (df[home_ga] + 0.01)
                away_def = 1.0 / (df[away_ga] + 0.01)
                df_enhanced[def_strength_col] = home_def / (away_def + 0.01)
                new_features.append(def_strength_col)
                defensive_ratios += 1
        
        # 3. CARD RATIOS (discipline comparison)
        card_ratios = 0
        card_types = ['yellow', 'red', 'foul']
        
        for card_type in card_types:
            for timeframe in ['short', 'medium', 'long']:
                home_card = f'home_{card_type}_ema_{timeframe}'
                away_card = f'away_{card_type}_ema_{timeframe}'
                
                if home_card in df.columns and away_card in df.columns:
                    # Card discipline ratio (lower cards = better discipline)
                    card_ratio_col = f'{card_type}_ema_{timeframe}_ratio'
                    df_enhanced[card_ratio_col] = df[away_card] / (df[home_card] + 0.01)  # Flipped for home advantage
                    new_features.append(card_ratio_col)
                    card_ratios += 1
        
        # 4. ADVANCED RATIO COMBINATIONS
        combination_ratios = 0
        
        # Goal efficiency ratios (goals per shot)
        for timeframe in ['short', 'medium', 'long']:
            home_goals = f'home_goals_ema_{timeframe}'
            away_goals = f'away_goals_ema_{timeframe}'
            home_shots = f'home_shots_ema_{timeframe}' if f'home_shots_ema_{timeframe}' in df.columns else None
            away_shots = f'away_shots_ema_{timeframe}' if f'away_shots_ema_{timeframe}' in df.columns else None
            
            if all(col in df.columns for col in [home_goals, away_goals]) and home_shots and away_shots:
                # Goal efficiency ratio
                home_efficiency = df[home_goals] / (df[home_shots] + 0.01)
                away_efficiency = df[away_goals] / (df[away_shots] + 0.01)
                
                efficiency_ratio_col = f'goal_efficiency_ema_{timeframe}_ratio'
                df_enhanced[efficiency_ratio_col] = home_efficiency / (away_efficiency + 0.01)
                new_features.append(efficiency_ratio_col)
                combination_ratios += 1
        
        # 5. MOMENTUM RATIOS
        momentum_ratios = 0
        for metric in ['goals', 'points']:
            home_momentum = f'home_{metric}_momentum'
            away_momentum = f'away_{metric}_momentum'
            
            if home_momentum in df.columns and away_momentum in df.columns:
                momentum_ratio_col = f'{metric}_momentum_ratio'
                df_enhanced[momentum_ratio_col] = df[home_momentum] / (abs(df[away_momentum]) + 0.01)
                new_features.append(momentum_ratio_col)
                momentum_ratios += 1
        
        print(f"   Created {shot_ratios} shot ratios")
        print(f"   Created {defensive_ratios} defensive ratios") 
        print(f"   Created {card_ratios} card ratios")
        print(f"   Created {combination_ratios} combination ratios")
        print(f"   Created {momentum_ratios} momentum ratios")
        print(f"   Total new ratio features: {len(new_features)}")
        
        return df_enhanced, new_features
    
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Optimize Random Forest with grid search."""
        
        print("\n🌳 Optimizing Random Forest...")
        
        # Use TimeSeriesSplit for proper validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Randomized search for efficiency
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("   Running randomized search...")
        rf_search = RandomizedSearchCV(
            rf, self.rf_param_grid, 
            n_iter=20, cv=tscv, 
            scoring='accuracy', 
            random_state=42, n_jobs=-1
        )
        
        rf_search.fit(X, y)
        
        print(f"   Best RF score: {rf_search.best_score_:.3f}")
        print(f"   Best RF params: {rf_search.best_params_}")
        
        return rf_search.best_estimator_
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
        """Optimize XGBoost with grid search."""
        
        print("\n🚀 Optimizing XGBoost...")
        
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # XGBoost classifier
        xgb_clf = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        print("   Running randomized search...")
        xgb_search = RandomizedSearchCV(
            xgb_clf, self.xgb_param_grid,
            n_iter=20, cv=tscv,
            scoring='accuracy',
            random_state=42, n_jobs=-1
        )
        
        xgb_search.fit(X, y)
        
        print(f"   Best XGB score: {xgb_search.best_score_:.3f}")
        print(f"   Best XGB params: {xgb_search.best_params_}")
        
        return xgb_search.best_estimator_
    
    def feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                         top_k: int = 50) -> Tuple[np.ndarray, List[str]]:
        """Select top K features to remove noise."""
        
        print(f"\n🎯 Feature Selection (Top {top_k})...")
        
        # Use SelectKBest with f_classif
        selector = SelectKBest(score_func=f_classif, k=top_k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        print(f"   Reduced from {len(feature_names)} to {len(selected_features)} features")
        print(f"   Top 5 selected features:")
        
        # Show top features by score
        feature_scores = selector.scores_
        selected_scores = [(feature_names[i], feature_scores[i]) for i in range(len(feature_names)) if selected_mask[i]]
        selected_scores.sort(key=lambda x: x[1], reverse=True)
        
        for feat, score in selected_scores[:5]:
            print(f"     {feat}: {score:.2f}")
        
        return X_selected, selected_features
    
    def create_stacking_ensemble(self, X: np.ndarray, y: np.ndarray, 
                                optimized_rf: RandomForestClassifier, 
                                optimized_xgb: xgb.XGBClassifier) -> StackingClassifier:
        """Create advanced stacking ensemble."""
        
        print("\n🧠 Creating Stacking Ensemble...")
        
        # Base models (level 0)
        base_models = [
            ('rf_optimized', optimized_rf),
            ('xgb_optimized', optimized_xgb),
            ('gb_baseline', GradientBoostingClassifier(n_estimators=150, random_state=42))
        ]
        
        # Meta-model (level 1) - Logistic Regression with regularization
        meta_model = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            C=1.0,
            penalty='l2'
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=3),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print(f"   Base models: {len(base_models)}")
        print(f"   Meta-model: Logistic Regression")
        print(f"   Stacking method: predict_proba")
        
        return stacking_clf
    
    def train_and_evaluate_optimized_models(self, X: np.ndarray, y: np.ndarray, 
                                          feature_names: List[str]) -> Dict:
        """Train and evaluate all optimized models."""
        
        print("\n🎯 Training Optimized Models...")
        
        # Chronological split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   Training: {len(X_train)}, Testing: {len(X_test)}")
        
        results = {}
        trained_models = {}
        
        # 1. Optimize Random Forest
        optimized_rf = self.optimize_random_forest(X_train, y_train)
        rf_pred = optimized_rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        results['optimized_rf'] = rf_accuracy
        trained_models['optimized_rf'] = optimized_rf
        print(f"   Optimized RF: {rf_accuracy:.3f}")
        
        # 2. Optimize XGBoost
        optimized_xgb = self.optimize_xgboost(X_train, y_train)
        xgb_pred = optimized_xgb.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        results['optimized_xgb'] = xgb_accuracy
        trained_models['optimized_xgb'] = optimized_xgb
        print(f"   Optimized XGB: {xgb_accuracy:.3f}")
        
        # 3. Feature Selection + Retrain
        X_train_selected, selected_features = self.feature_selection(X_train, y_train, feature_names)
        X_test_selected = X_test[:, [i for i, feat in enumerate(feature_names) if feat in selected_features]]
        
        # Train RF on selected features
        rf_selected = RandomForestClassifier(**optimized_rf.get_params())
        rf_selected.fit(X_train_selected, y_train)
        rf_selected_pred = rf_selected.predict(X_test_selected)
        rf_selected_accuracy = accuracy_score(y_test, rf_selected_pred)
        results['rf_feature_selected'] = rf_selected_accuracy
        trained_models['rf_feature_selected'] = rf_selected
        print(f"   RF + Feature Selection: {rf_selected_accuracy:.3f}")
        
        # 4. Stacking Ensemble
        print("\n   Training Stacking Ensemble...")
        stacking_clf = self.create_stacking_ensemble(X_train, y_train, optimized_rf, optimized_xgb)
        stacking_clf.fit(X_train, y_train)
        stacking_pred = stacking_clf.predict(X_test)
        stacking_accuracy = accuracy_score(y_test, stacking_pred)
        results['stacking_ensemble'] = stacking_accuracy
        trained_models['stacking_ensemble'] = stacking_clf
        print(f"   Stacking Ensemble: {stacking_accuracy:.3f}")
        
        # 5. Meta-learning: Predict best model per match
        print("\n   Creating Meta-Learning System...")
        meta_predictions = self.create_meta_learning_predictions(
            trained_models, X_test, y_test, ['optimized_rf', 'optimized_xgb', 'stacking_ensemble']
        )
        results['meta_learning'] = meta_predictions['accuracy']
        print(f"   Meta-Learning: {meta_predictions['accuracy']:.3f}")
        
        return results, trained_models, {'selected_features': selected_features}
    
    def create_meta_learning_predictions(self, trained_models: Dict, X_test: np.ndarray, 
                                       y_test: np.ndarray, model_names: List[str]) -> Dict:
        """Create meta-learning system to predict best model per match."""
        
        # Get predictions from each model
        model_predictions = {}
        model_probabilities = {}
        
        for model_name in model_names:
            model = trained_models[model_name]
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            
            model_predictions[model_name] = pred
            model_probabilities[model_name] = prob
        
        # Meta-learning: Choose model with highest confidence per match
        meta_predictions = []
        model_choices = []
        
        for i in range(len(X_test)):
            best_model = None
            best_confidence = 0
            
            for model_name in model_names:
                confidence = max(model_probabilities[model_name][i])
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_model = model_name
            
            meta_predictions.append(model_predictions[best_model][i])
            model_choices.append(best_model)
        
        meta_accuracy = accuracy_score(y_test, meta_predictions)
        
        # Model selection frequency
        from collections import Counter
        model_freq = Counter(model_choices)
        
        return {
            'accuracy': meta_accuracy,
            'predictions': meta_predictions,
            'model_choices': model_choices,
            'model_frequency': dict(model_freq)
        }


def run_phase3_optimization():
    """Run complete Phase 3 optimization pipeline."""
    
    print("🚀 EPL PROPHET - PHASE 3: COMPREHENSIVE OPTIMIZATION")
    print("=" * 70)
    print("The FINAL PUSH to 55%+ accuracy!")
    
    # Load Phase 2 enhanced data
    print("📊 Loading Phase 2 Enhanced Data...")
    df = pd.read_csv("../outputs/phase2_enhanced_features.csv")
    print(f"   Loaded {len(df)} matches with {len(df.columns)} features")
    
    # Initialize optimizer
    optimizer = Phase3Optimizer()
    
    # Create expanded ratio features
    df_final, new_ratio_features = optimizer.create_expanded_ratio_features(df)
    print(f"   Enhanced to {len(df_final.columns)} total features")
    
    # Prepare data
    df_clean = df_final[df_final['actual_result'].notna()].copy()
    
    exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                   'actual_home_goals', 'actual_away_goals']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df_clean['actual_result'])
    
    print(f"   Final feature matrix: {X.shape}")
    print(f"   Classes: {le.classes_}")
    
    # Train and evaluate all optimized models
    results, trained_models, metadata = optimizer.train_and_evaluate_optimized_models(X, y, feature_cols)
    
    # Results summary
    print(f"\n🏆 PHASE 3 FINAL RESULTS:")
    print("=" * 50)
    
    for model_name, accuracy in results.items():
        print(f"   {model_name:<25}: {accuracy:.3f}")
    
    best_model = max(results, key=results.get)
    best_accuracy = results[best_model]
    
    print(f"\n🥇 BEST MODEL: {best_model}")
    print(f"🎯 BEST ACCURACY: {best_accuracy:.3f}")
    
    # Check if we hit the target
    target = 0.55
    if best_accuracy >= target:
        print(f"\n🎉 TARGET ACHIEVED! {best_accuracy:.3f} >= {target}")
        print("🏆 EPL Prophet is now a WORLD-CLASS prediction system!")
    else:
        improvement = best_accuracy - 0.525  # From Phase 2 best
        print(f"\n📈 IMPROVEMENT from Phase 2: {improvement:+.3f}")
        print(f"🎯 Progress toward {target}: {best_accuracy:.3f}")
        print(f"🔥 Gap remaining: {target - best_accuracy:.3f}")
    
    # Save models and results
    print(f"\n💾 Saving Phase 3 Models...")
    
    for model_name, model in trained_models.items():
        if model_name != 'rf_feature_selected':  # Skip feature-selected version for simplicity
            joblib.dump(model, f"../outputs/phase3_{model_name}.joblib")
            print(f"   Saved {model_name}")
    
    # Save final results
    results_df = pd.DataFrame([results]).T
    results_df.columns = ['accuracy']
    results_df.to_csv("../outputs/phase3_final_results.csv")
    
    # Save enhanced dataset
    df_final.to_csv("../outputs/phase3_final_features.csv", index=False)
    
    print(f"\n✨ PHASE 3 OPTIMIZATION COMPLETE!")
    print(f"   🎯 Best accuracy: {best_accuracy:.3f}")
    print(f"   📊 Total features: {len(feature_cols)}")
    print(f"   🔥 New ratio features: {len(new_ratio_features)}")
    print(f"   🤖 Optimized models: {len(trained_models)}")
    print(f"   🧠 Advanced techniques: Hyperparameter tuning, feature selection, stacking, meta-learning")
    
    print(f"\n🚀 EPL PROPHET EVOLUTION COMPLETE:")
    print(f"   Phase 1: Recency weighting → 52.7%")
    print(f"   Phase 2: Multi-timeframe → 52.5%") 
    print(f"   Phase 3: Optimization → {best_accuracy:.3f}")
    
    if best_accuracy >= target:
        print(f"\n💎 MISSION ACCOMPLISHED!")
        print(f"EPL Prophet has achieved world-class performance!")
    
    return results, trained_models, new_ratio_features


if __name__ == "__main__":
    run_phase3_optimization() 
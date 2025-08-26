#!/usr/bin/env python3
"""
EPL PROPHET - Phase 3: Comprehensive Optimization
The FINAL PUSH to 55%+ accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

class Phase3Optimizer:
    """Comprehensive optimization system."""
    
    def __init__(self):
        self.rf_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [12, 15, 18, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        self.xgb_params = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    
    def create_expanded_ratios(self, df):
        """Create comprehensive ratio features."""
        
        print("🔥 Creating Expanded Ratio Features...")
        
        df_enhanced = df.copy()
        new_features = []
        
        # 1. DEFENSIVE RATIOS
        defensive_count = 0
        for timeframe in ['short', 'medium', 'long']:
            home_ga = f'home_goals_against_ema_{timeframe}'
            away_ga = f'away_goals_against_ema_{timeframe}'
            
            if home_ga in df.columns and away_ga in df.columns:
                # Goals against ratio (flipped for home advantage)
                ga_ratio = f'goals_against_ema_{timeframe}_ratio'
                df_enhanced[ga_ratio] = df[away_ga] / (df[home_ga] + 0.01)
                new_features.append(ga_ratio)
                defensive_count += 1
                
                # Defensive strength ratio
                def_strength = f'defensive_strength_ema_{timeframe}_ratio'
                home_def = 1.0 / (df[home_ga] + 0.01)
                away_def = 1.0 / (df[away_ga] + 0.01)
                df_enhanced[def_strength] = home_def / (away_def + 0.01)
                new_features.append(def_strength)
                defensive_count += 1
        
        # 2. MOMENTUM RATIOS
        momentum_count = 0
        for metric in ['goals', 'points']:
            home_momentum = f'home_{metric}_momentum'
            away_momentum = f'away_{metric}_momentum'
            
            if home_momentum in df.columns and away_momentum in df.columns:
                momentum_ratio = f'{metric}_momentum_ratio'
                df_enhanced[momentum_ratio] = df[home_momentum] / (abs(df[away_momentum]) + 0.01)
                new_features.append(momentum_ratio)
                momentum_count += 1
        
        # 3. ADVANTAGE AMPLIFIERS
        advantage_count = 0
        for timeframe in ['short', 'medium', 'long']:
            for metric in ['goals', 'points']:
                home_col = f'home_{metric}_ema_{timeframe}'
                away_col = f'away_{metric}_ema_{timeframe}'
                
                if home_col in df.columns and away_col in df.columns:
                    # Squared advantage (amplifies differences)
                    squared_adv = f'{metric}_ema_{timeframe}_squared_advantage'
                    advantage = df[home_col] - df[away_col]
                    df_enhanced[squared_adv] = np.sign(advantage) * (advantage ** 2)
                    new_features.append(squared_adv)
                    advantage_count += 1
        
        # 4. FORM STABILITY RATIOS
        stability_count = 0
        for team in ['home', 'away']:
            for metric in ['goals', 'points']:
                short_col = f'{team}_{metric}_ema_short'
                long_col = f'{team}_{metric}_ema_long'
                
                if short_col in df.columns and long_col in df.columns:
                    # Form stability (how consistent is recent vs long-term)
                    stability = f'{team}_{metric}_form_stability'
                    df_enhanced[stability] = 1.0 / (1.0 + abs(df[short_col] - df[long_col]))
                    new_features.append(stability)
                    stability_count += 1
        
        print(f"   Created {defensive_count} defensive ratios")
        print(f"   Created {momentum_count} momentum ratios")
        print(f"   Created {advantage_count} advantage amplifiers")
        print(f"   Created {stability_count} stability ratios")
        print(f"   Total new features: {len(new_features)}")
        
        return df_enhanced, new_features
    
    def optimize_random_forest(self, X, y):
        """Optimize Random Forest with randomized search."""
        
        print("\n🌳 Optimizing Random Forest...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        rf_search = RandomizedSearchCV(
            rf, self.rf_params, 
            n_iter=15, cv=tscv, 
            scoring='accuracy', 
            random_state=42, n_jobs=-1
        )
        
        rf_search.fit(X, y)
        
        print(f"   Best RF CV score: {rf_search.best_score_:.3f}")
        print(f"   Best RF params: {rf_search.best_params_}")
        
        return rf_search.best_estimator_
    
    def optimize_xgboost(self, X, y):
        """Optimize XGBoost with randomized search."""
        
        print("\n🚀 Optimizing XGBoost...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        xgb_clf = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        
        xgb_search = RandomizedSearchCV(
            xgb_clf, self.xgb_params,
            n_iter=15, cv=tscv,
            scoring='accuracy',
            random_state=42, n_jobs=-1
        )
        
        xgb_search.fit(X, y)
        
        print(f"   Best XGB CV score: {xgb_search.best_score_:.3f}")
        print(f"   Best XGB params: {xgb_search.best_params_}")
        
        return xgb_search.best_estimator_
    
    def feature_selection(self, X, y, feature_names, top_k=60):
        """Select top K features."""
        
        print(f"\n🎯 Feature Selection (Top {top_k})...")
        
        selector = SelectKBest(score_func=f_classif, k=top_k)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        print(f"   Reduced from {len(feature_names)} to {len(selected_features)} features")
        
        # Show top selected features
        feature_scores = selector.scores_
        selected_scores = [(feature_names[i], feature_scores[i]) 
                          for i in range(len(feature_names)) if selected_mask[i]]
        selected_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Top 5 selected features:")
        for feat, score in selected_scores[:5]:
            print(f"     {feat}: {score:.1f}")
        
        return X_selected, selected_features
    
    def create_stacking_ensemble(self, optimized_rf, optimized_xgb):
        """Create stacking ensemble."""
        
        print("\n🧠 Creating Stacking Ensemble...")
        
        base_models = [
            ('rf_opt', optimized_rf),
            ('xgb_opt', optimized_xgb),
            ('gb_base', GradientBoostingClassifier(n_estimators=150, random_state=42))
        ]
        
        meta_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=TimeSeriesSplit(n_splits=3),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print(f"   Base models: {len(base_models)}")
        print(f"   Meta-model: Logistic Regression")
        
        return stacking_clf
    
    def meta_learning_predictions(self, models, X_test, y_test, model_names):
        """Meta-learning: choose best model per match."""
        
        # Get predictions and probabilities
        model_preds = {}
        model_probs = {}
        
        for name in model_names:
            model = models[name]
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            
            model_preds[name] = pred
            model_probs[name] = prob
        
        # Choose model with highest confidence per match
        meta_predictions = []
        model_choices = []
        
        for i in range(len(X_test)):
            best_model = None
            best_confidence = 0
            
            for name in model_names:
                confidence = max(model_probs[name][i])
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_model = name
            
            meta_predictions.append(model_preds[best_model][i])
            model_choices.append(best_model)
        
        meta_accuracy = accuracy_score(y_test, meta_predictions)
        model_freq = Counter(model_choices)
        
        return {
            'accuracy': meta_accuracy,
            'model_frequency': dict(model_freq)
        }
    
    def train_all_models(self, X, y, feature_names):
        """Train and evaluate all optimized models."""
        
        print("\n🎯 Training All Optimized Models...")
        
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   Training: {len(X_train)}, Testing: {len(X_test)}")
        
        results = {}
        models = {}
        
        # 1. Optimized Random Forest
        opt_rf = self.optimize_random_forest(X_train, y_train)
        rf_pred = opt_rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        results['optimized_rf'] = rf_acc
        models['optimized_rf'] = opt_rf
        print(f"   Optimized RF: {rf_acc:.3f}")
        
        # 2. Optimized XGBoost
        opt_xgb = self.optimize_xgboost(X_train, y_train)
        xgb_pred = opt_xgb.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['optimized_xgb'] = xgb_acc
        models['optimized_xgb'] = opt_xgb
        print(f"   Optimized XGB: {xgb_acc:.3f}")
        
        # 3. Feature Selection + RF
        X_train_sel, sel_features = self.feature_selection(X_train, y_train, feature_names)
        X_test_sel = X_test[:, [i for i, feat in enumerate(feature_names) if feat in sel_features]]
        
        rf_sel = RandomForestClassifier(**opt_rf.get_params())
        rf_sel.fit(X_train_sel, y_train)
        rf_sel_pred = rf_sel.predict(X_test_sel)
        rf_sel_acc = accuracy_score(y_test, rf_sel_pred)
        results['rf_feature_selected'] = rf_sel_acc
        models['rf_feature_selected'] = rf_sel
        print(f"   RF + Feature Selection: {rf_sel_acc:.3f}")
        
        # 4. Stacking Ensemble
        print("\n   Training Stacking Ensemble...")
        stacking = self.create_stacking_ensemble(opt_rf, opt_xgb)
        stacking.fit(X_train, y_train)
        stack_pred = stacking.predict(X_test)
        stack_acc = accuracy_score(y_test, stack_pred)
        results['stacking_ensemble'] = stack_acc
        models['stacking_ensemble'] = stacking
        print(f"   Stacking Ensemble: {stack_acc:.3f}")
        
        # 5. Meta-Learning
        print("\n   Creating Meta-Learning System...")
        meta_result = self.meta_learning_predictions(
            models, X_test, y_test, 
            ['optimized_rf', 'optimized_xgb', 'stacking_ensemble']
        )
        results['meta_learning'] = meta_result['accuracy']
        print(f"   Meta-Learning: {meta_result['accuracy']:.3f}")
        print(f"   Model frequency: {meta_result['model_frequency']}")
        
        return results, models

def run_phase3():
    """Run Phase 3 comprehensive optimization."""
    
    print("🚀 EPL PROPHET - PHASE 3: COMPREHENSIVE OPTIMIZATION")
    print("=" * 70)
    print("The FINAL PUSH to 55%+ accuracy!")
    
    # Load Phase 2 data
    print("📊 Loading Phase 2 Data...")
    df = pd.read_csv("../outputs/phase2_enhanced_features.csv")
    print(f"   Loaded {len(df)} matches with {len(df.columns)} features")
    
    # Initialize optimizer
    optimizer = Phase3Optimizer()
    
    # Create expanded ratio features
    df_final, new_features = optimizer.create_expanded_ratios(df)
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
    
    # Train all models
    results, models = optimizer.train_all_models(X, y, feature_cols)
    
    # Results summary
    print(f"\n🏆 PHASE 3 FINAL RESULTS:")
    print("=" * 50)
    
    for model_name, accuracy in results.items():
        print(f"   {model_name:<25}: {accuracy:.3f}")
    
    best_model = max(results, key=results.get)
    best_accuracy = results[best_model]
    
    print(f"\n🥇 BEST MODEL: {best_model}")
    print(f"🎯 BEST ACCURACY: {best_accuracy:.3f}")
    
    # Check target achievement
    target = 0.55
    if best_accuracy >= target:
        print(f"\n🎉 TARGET ACHIEVED! {best_accuracy:.3f} >= {target}")
        print("🏆 EPL Prophet is now WORLD-CLASS!")
    else:
        improvement = best_accuracy - 0.525  # Phase 2 best
        print(f"\n📈 Improvement from Phase 2: {improvement:+.3f}")
        print(f"🎯 Progress toward {target}: {best_accuracy:.3f}")
        print(f"🔥 Gap remaining: {target - best_accuracy:.3f}")
    
    # Save models
    print(f"\n💾 Saving Phase 3 Models...")
    
    for name, model in models.items():
        if name != 'rf_feature_selected':  # Skip for space
            joblib.dump(model, f"../outputs/phase3_{name}.joblib")
            print(f"   Saved {name}")
    
    # Save results
    results_df = pd.DataFrame([results]).T
    results_df.columns = ['accuracy']
    results_df.to_csv("../outputs/phase3_results.csv")
    
    # Save final dataset
    df_final.to_csv("../outputs/phase3_final_features.csv", index=False)
    
    print(f"\n✨ PHASE 3 COMPLETE!")
    print(f"   �� Best accuracy: {best_accuracy:.3f}")
    print(f"   📊 Total features: {len(feature_cols)}")
    print(f"   🔥 New ratio features: {len(new_features)}")
    print(f"   🤖 Models trained: {len(models)}")
    
    print(f"\n🚀 EPL PROPHET EVOLUTION:")
    print(f"   Phase 1: Recency weighting → 52.7%")
    print(f"   Phase 2: Multi-timeframe → 52.5%")
    print(f"   Phase 3: Optimization → {best_accuracy:.3f}")
    
    if best_accuracy >= target:
        print(f"\n💎 MISSION ACCOMPLISHED!")
        print("EPL Prophet achieved world-class performance!")
    else:
        print(f"\n🚀 Excellent progress! Ready for further optimization!")
    
    return results, models, new_features

if __name__ == "__main__":
    run_phase3()

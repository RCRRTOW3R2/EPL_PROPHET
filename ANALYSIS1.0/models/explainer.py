#!/usr/bin/env python3
"""
EPL PROPHET - SHAP EXPLAINABILITY SYSTEM
Making our 53.7% champion model explainable!
"""

import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class EPLExplainer:
    """SHAP explainability for EPL Prophet."""
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.class_names = ['Away Win', 'Draw', 'Home Win']
        
    def load_champion_model(self):
        """Load our champion model."""
        
        print("🏆 Loading Champion Model...")
        
        try:
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion Random Forest loaded")
            
            # Load feature data
            df = pd.read_csv("../outputs/champion_features.csv")
            
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def setup_shap(self, X_background):
        """Setup SHAP explainer."""
        
        print("🧠 Setting up SHAP...")
        
        self.explainer = shap.TreeExplainer(self.model)
        print("   ✅ SHAP TreeExplainer ready")
        
        # Test SHAP on small sample
        sample_size = min(50, len(X_background))
        shap_values = self.explainer.shap_values(X_background[:sample_size])
        print(f"   ✅ SHAP tested on {sample_size} samples")
        
        return shap_values
    
    def explain_match(self, match_features, home_team, away_team, actual_result=None):
        """Explain a single match prediction."""
        
        print(f"\n🔍 EXPLAINING: {home_team} vs {away_team}")
        print("=" * 60)
        
        # Get prediction
        prediction_proba = self.model.predict_proba([match_features])[0]
        prediction_class = self.model.predict([match_features])[0]
        
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[prediction_class]
        
        print(f"🎯 PREDICTION: {predicted_result}")
        print(f"   Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"     {class_name}: {prediction_proba[i]:.1%}")
        
        if actual_result:
            le = LabelEncoder()
            le.fit(['A', 'D', 'H'])
            actual_readable = result_map[le.transform([actual_result])[0]]
            correct = "✅" if predicted_result == actual_readable else "❌"
            print(f"   Actual: {actual_readable} {correct}")
        
        # SHAP values
        shap_values = self.explainer.shap_values([match_features])
        class_shap_values = shap_values[prediction_class][0]
        
        # Top factors
        feature_contributions = list(zip(self.feature_names, class_shap_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🔥 TOP FACTORS FOR '{predicted_result}':")
        print("-" * 50)
        
        positive = [f for f in feature_contributions if f[1] > 0][:5]
        negative = [f for f in feature_contributions if f[1] < 0][:3]
        
        if positive:
            print("📈 SUPPORTING FACTORS:")
            for feat, value in positive:
                print(f"   + {feat}: {value:+.3f}")
        
        if negative:
            print("\n📉 OPPOSING FACTORS:")
            for feat, value in negative:
                print(f"   - {feat}: {value:+.3f}")
        
        # Human explanation
        print(f"\n💬 HUMAN EXPLANATION:")
        self.translate_to_human(positive, negative, home_team, away_team)
        
        return predicted_result, prediction_proba
    
    def translate_to_human(self, positive_factors, negative_factors, home_team, away_team):
        """Convert to human language."""
        
        explanations = []
        
        for feat, value in positive_factors[:3]:
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                explanations.append(f"✅ {explanation}")
        
        for feat, value in negative_factors[:2]:
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                explanations.append(f"⚠️ {explanation}")
        
        for explanation in explanations:
            print(f"   {explanation}")
    
    def feature_to_human(self, feature_name, shap_value, home_team, away_team):
        """Convert feature to human explanation."""
        
        # Logarithmic ratios (champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has superior long-term goal-scoring form"
            elif 'points' in feature_name and 'long' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has superior long-term points form"
            elif 'goals' in feature_name and 'medium' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has better recent goal-scoring momentum"
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has a significant goal-scoring advantage"
            elif 'points' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has a significant points form advantage"
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has overall better goal-scoring form"
            elif 'points' in feature_name:
                team = home_team if shap_value > 0 else away_team
                return f"{team} has overall better points form"
        
        # Defensive features
        if 'goals_against' in feature_name and 'ratio' in feature_name:
            team = home_team if shap_value > 0 else away_team
            return f"{team} has superior defensive solidity"
        
        # Momentum
        if 'momentum' in feature_name:
            team = home_team if shap_value > 0 else away_team
            return f"{team} has positive momentum trends"
        
        return None
    
    def global_feature_importance(self, X_sample):
        """Global feature importance via SHAP."""
        
        print("📊 GLOBAL FEATURE IMPORTANCE (SHAP)")
        print("=" * 50)
        
        shap_values = self.explainer.shap_values(X_sample)
        
        # Sum absolute SHAP values across classes
        total_importance = np.zeros(len(self.feature_names))
        for class_shap in shap_values:
            total_importance += np.mean(np.abs(class_shap), axis=0)
        
        # Rank features
        feature_importance = list(zip(self.feature_names, total_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("🏆 TOP 15 FEATURES:")
        for i, (feat, importance) in enumerate(feature_importance[:15], 1):
            print(f"   {i:2d}. {feat}: {importance:.3f}")
        
        return feature_importance

def run_explainer_demo():
    """Demo SHAP explainer."""
    
    print("🚀 EPL PROPHET - SHAP EXPLAINABILITY")
    print("=" * 60)
    print("Making our 53.7% champion model explainable!")
    
    # Initialize
    explainer = EPLExplainer()
    
    if not explainer.load_champion_model():
        return
    
    # Load data
    print("\n📊 Loading Data...")
    df = pd.read_csv("../outputs/champion_features.csv")
    df_clean = df[df['actual_result'].notna()].copy()
    
    exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                   'actual_home_goals', 'actual_away_goals']
    
    feature_cols = [col for col in df_clean.columns 
                   if col not in exclude_cols and col in explainer.feature_names]
    X = df_clean[feature_cols].fillna(0).values
    
    # Setup SHAP
    explainer.setup_shap(X)
    
    # Analyze recent matches
    print("\n🔍 ANALYZING RECENT MATCHES...")
    
    recent_matches = df_clean.tail(3)  # Last 3 matches
    
    for idx, match in recent_matches.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        actual_result = match['actual_result']
        
        match_features = match[feature_cols].fillna(0).values
        
        explainer.explain_match(match_features, home_team, away_team, actual_result)
        print("\n" + "="*60 + "\n")
    
    # Global importance
    sample_size = min(100, len(X))
    explainer.global_feature_importance(X[:sample_size])
    
    print(f"\n✨ SHAP EXPLAINABILITY COMPLETE!")
    print("🏆 Champion model is now fully explainable!")
    print("💬 You can now understand WHY predictions are made!")
    print("🔮 Perfect for upcoming match analysis!")

if __name__ == "__main__":
    run_explainer_demo()

#!/usr/bin/env python3
"""
EPL PROPHET - SHAP EXPLAINABILITY SYSTEM
========================================

SHAP explanations for our 53.7% champion model!

Now we can answer questions like:
- "Why will Manchester City beat Arsenal?"
- "What factors drive this prediction?"
- "Which team strengths matter most?"

Makes our world-class model explainable and actionable!
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

class EPLProphetExplainer:
    """SHAP explainability for EPL Prophet predictions."""
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.label_encoder = None
        self.class_names = ['Away Win', 'Draw', 'Home Win']
        
    def load_champion_model(self):
        """Load our 53.7% champion model."""
        
        print("🏆 Loading Champion Model...")
        
        try:
            # Load the champion model
            self.model = joblib.load("../outputs/champion_model.joblib")
            print("   ✅ Champion Random Forest loaded")
            
            # Load feature data to get feature names
            df = pd.read_csv("../outputs/champion_features.csv")
            
            # Get feature names
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            self.feature_names = [col for col in df.columns if col not in exclude_cols]
            print(f"   ✅ {len(self.feature_names)} features loaded")
            
            # Setup label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(['A', 'D', 'H'])
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return False
    
    def setup_shap_explainer(self, X_background):
        """Setup SHAP explainer with background data."""
        
        print("🧠 Setting up SHAP Explainer...")
        
        # Use TreeExplainer for Random Forest (most efficient)
        self.explainer = shap.TreeExplainer(self.model)
        print("   ✅ SHAP TreeExplainer initialized")
        
        # Calculate SHAP values for background data (sample for efficiency)
        sample_size = min(100, len(X_background))
        background_sample = X_background[:sample_size]
        
        print(f"   🔄 Calculating SHAP values for {sample_size} background samples...")
        shap_values = self.explainer.shap_values(background_sample)
        print("   ✅ SHAP explainer ready!")
        
        return shap_values
    
    def explain_prediction(self, match_features, home_team, away_team, actual_result=None):
        """Explain a single match prediction with SHAP."""
        
        print(f"\n🔍 EXPLAINING: {home_team} vs {away_team}")
        print("=" * 60)
        
        # Get prediction
        prediction_proba = self.model.predict_proba([match_features])[0]
        prediction_class = self.model.predict([match_features])[0]
        
        # Convert to readable result
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        predicted_result = result_map[prediction_class]
        
        print(f"🎯 PREDICTION: {predicted_result}")
        print(f"   Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"     {class_name}: {prediction_proba[i]:.1%}")
        
        if actual_result:
            actual_readable = result_map[self.label_encoder.transform([actual_result])[0]]
            correct = "✅" if predicted_result == actual_readable else "❌"
            print(f"   Actual Result: {actual_readable} {correct}")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values([match_features])
        
        # For multiclass, shap_values is a list of arrays (one per class)
        # We'll focus on the predicted class
        class_shap_values = shap_values[prediction_class][0]
        
        # Get top positive and negative contributors
        feature_contributions = list(zip(self.feature_names, class_shap_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🔥 TOP FACTORS FOR '{predicted_result}':")
        print("-" * 50)
        
        positive_factors = [f for f in feature_contributions if f[1] > 0][:5]
        negative_factors = [f for f in feature_contributions if f[1] < 0][:5]
        
        if positive_factors:
            print("📈 SUPPORTING FACTORS:")
            for feat, value in positive_factors:
                print(f"   + {feat}: {value:+.3f}")
        
        if negative_factors:
            print("\n📉 OPPOSING FACTORS:")
            for feat, value in negative_factors:
                print(f"   - {feat}: {value:+.3f}")
        
        # Translate top factors to human language
        print(f"\n💬 HUMAN EXPLANATION:")
        self.translate_to_human_language(positive_factors, negative_factors, home_team, away_team)
        
        return {
            'prediction': predicted_result,
            'probabilities': dict(zip(self.class_names, prediction_proba)),
            'shap_values': class_shap_values,
            'top_positive': positive_factors,
            'top_negative': negative_factors
        }
    
    def translate_to_human_language(self, positive_factors, negative_factors, home_team, away_team):
        """Translate SHAP values to human-readable explanations."""
        
        explanations = []
        
        for feat, value in positive_factors[:3]:  # Top 3 positive
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                explanations.append(f"✅ {explanation}")
        
        for feat, value in negative_factors[:2]:  # Top 2 negative
            explanation = self.feature_to_human(feat, value, home_team, away_team)
            if explanation:
                explanations.append(f"⚠️ {explanation}")
        
        for explanation in explanations:
            print(f"   {explanation}")
    
    def feature_to_human(self, feature_name, shap_value, home_team, away_team):
        """Convert feature names to human explanations."""
        
        # Logarithmic ratios (our champions!)
        if 'log_ratio' in feature_name:
            if 'goals' in feature_name and 'long' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has superior long-term goal-scoring form"
                else:
                    return f"{away_team} has superior long-term goal-scoring form"
            elif 'points' in feature_name and 'long' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has superior long-term points form"
                else:
                    return f"{away_team} has superior long-term points form"
            elif 'goals' in feature_name and 'medium' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has better recent goal-scoring momentum"
                else:
                    return f"{away_team} has better recent goal-scoring momentum"
        
        # Squared advantages
        if 'squared_advantage' in feature_name:
            if 'goals' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has a significant goal-scoring advantage"
                else:
                    return f"{away_team} has a significant goal-scoring advantage"
            elif 'points' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has a significant points form advantage"
                else:
                    return f"{away_team} has a significant points form advantage"
        
        # EMA advantages
        if 'ema_advantage' in feature_name:
            if 'goals' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has overall better goal-scoring form"
                else:
                    return f"{away_team} has overall better goal-scoring form"
            elif 'points' in feature_name:
                if shap_value > 0:
                    return f"{home_team} has overall better points form"
                else:
                    return f"{away_team} has overall better points form"
        
        # Defensive ratios
        if 'goals_against' in feature_name and 'ratio' in feature_name:
            if shap_value > 0:
                return f"{home_team} has superior defensive solidity"
            else:
                return f"{away_team} has superior defensive solidity"
        
        # Momentum indicators
        if 'momentum' in feature_name:
            if shap_value > 0:
                return f"{home_team} has positive momentum trends"
            else:
                return f"{away_team} has positive momentum trends"
        
        return None
    
    def analyze_upcoming_matches(self, upcoming_matches_df):
        """Analyze upcoming matches with SHAP explanations."""
        
        print("🔮 UPCOMING MATCHES ANALYSIS")
        print("=" * 70)
        
        results = []
        
        for idx, match in upcoming_matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get match features (excluding metadata columns)
            exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                           'actual_home_goals', 'actual_away_goals']
            
            feature_cols = [col for col in match.index if col not in exclude_cols and col in self.feature_names]
            match_features = match[feature_cols].fillna(0).values
            
            # Explain prediction
            explanation = self.explain_prediction(match_features, home_team, away_team)
            results.append({
                'home_team': home_team,
                'away_team': away_team,
                'prediction': explanation['prediction'],
                'confidence': max(explanation['probabilities'].values()),
                'explanation': explanation
            })
            
            print("\n" + "="*60 + "\n")
        
        return results
    
    def create_feature_importance_summary(self, X_sample):
        """Create overall feature importance summary using SHAP."""
        
        print("📊 GLOBAL FEATURE IMPORTANCE (SHAP-based)")
        print("=" * 50)
        
        # Calculate SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        # For multiclass, sum absolute SHAP values across all classes
        total_importance = np.zeros(len(self.feature_names))
        
        for class_shap in shap_values:
            total_importance += np.mean(np.abs(class_shap), axis=0)
        
        # Create feature importance ranking
        feature_importance = list(zip(self.feature_names, total_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("🏆 TOP 15 MOST IMPORTANT FEATURES:")
        for i, (feat, importance) in enumerate(feature_importance[:15], 1):
            print(f"   {i:2d}. {feat}: {importance:.3f}")
        
        return feature_importance


def demonstrate_shap_explainer():
    """Demonstrate SHAP explainer with actual matches."""
    
    print("🚀 EPL PROPHET - SHAP EXPLAINABILITY DEMO")
    print("=" * 70)
    print("Making our 53.7% champion model explainable!")
    
    # Initialize explainer
    explainer = EPLProphetExplainer()
    
    # Load champion model
    if not explainer.load_champion_model():
        print("❌ Failed to load champion model")
        return
    
    # Load data for analysis
    print("\n📊 Loading Match Data...")
    df = pd.read_csv("../outputs/champion_features.csv")
    df_clean = df[df['actual_result'].notna()].copy()
    
    # Prepare features
    exclude_cols = ['match_id', 'date', 'home_team', 'away_team', 'actual_result', 
                   'actual_home_goals', 'actual_away_goals']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols and col in explainer.feature_names]
    X = df_clean[feature_cols].fillna(0).values
    
    # Setup SHAP explainer
    shap_values_sample = explainer.setup_shap_explainer(X)
    
    # Analyze recent matches
    print("\n🔍 ANALYZING RECENT MATCHES...")
    
    # Get last 5 matches for demonstration
    recent_matches = df_clean.tail(5)
    
    for idx, match in recent_matches.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        actual_result = match['actual_result']
        
        match_features = match[feature_cols].fillna(0).values
        
        explainer.explain_prediction(match_features, home_team, away_team, actual_result)
        print("\n" + "="*60 + "\n")
    
    # Global feature importance
    sample_size = min(200, len(X))
    explainer.create_feature_importance_summary(X[:sample_size])
    
    print(f"\n✨ SHAP EXPLAINABILITY COMPLETE!")
    print("🏆 Our 53.7% champion model is now fully explainable!")
    print("💬 You can now understand WHY each prediction is made!")
    print("🔮 Perfect for analyzing upcoming matches!")


if __name__ == "__main__":
    demonstrate_shap_explainer() 
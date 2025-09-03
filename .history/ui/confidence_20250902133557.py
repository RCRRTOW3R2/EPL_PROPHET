#!/usr/bin/env python3
"""
EPL Prophet - Enhanced Confidence Calculator
Multi-dimensional confidence scoring incorporating context features
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import yaml

class ConfidenceCalculator:
    """Calculate composite confidence scores for EPL Prophet predictions"""
    
    def __init__(self, config_path="config/context_config.yaml"):
        self.config = self.load_config(config_path)
        self.weights = self.config.get('confidence_weights', [0.35, 0.25, 0.25, 0.15])
        
    def load_config(self, config_path):
        """Load confidence configuration"""
        default_config = {
            'confidence_weights': [0.35, 0.25, 0.25, 0.15],  # [calibrated, agreement, ensemble, coverage]
            'market_odds_available': False,
            'ensemble_cv_folds': 5
        }
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            pass
            
        return default_config
    
    def calculate_calibrated_confidence(self, model_probs, true_labels=None, cv_scores=None):
        """
        Calculate confidence based on model calibration
        Higher score = better calibrated predictions
        """
        if cv_scores is not None:
            # Use cross-validation Brier scores
            avg_brier = np.mean(cv_scores)
            calibrated_conf = 1 - min(avg_brier, 0.5) / 0.5  # Scale 0-1
        elif true_labels is not None:
            # Calculate Brier score for validation set
            brier = brier_score_loss(true_labels, model_probs)
            calibrated_conf = 1 - min(brier, 0.5) / 0.5
        else:
            # Fallback: Use prediction uncertainty
            max_prob = np.max(model_probs)
            calibrated_conf = (max_prob - 0.33) / 0.67  # Scale from random (33%) to certain (100%)
            
        return max(0, min(1, calibrated_conf))
    
    def calculate_market_agreement(self, model_probs, market_probs=None):
        """
        Calculate confidence based on agreement with market odds
        Higher score = closer to market consensus
        """
        if market_probs is None:
            # No market data available, return neutral
            return 0.5
        
        # Calculate absolute difference in home win probability
        model_home_prob = model_probs[0] if len(model_probs) == 3 else model_probs
        market_home_prob = market_probs[0] if len(market_probs) == 3 else market_probs
        
        diff = abs(model_home_prob - market_home_prob)
        agreement_conf = 1 - min(diff, 0.5) / 0.5  # Scale 0-1
        
        return max(0, min(1, agreement_conf))
    
    def calculate_ensemble_confidence(self, ensemble_probs):
        """
        Calculate confidence based on ensemble agreement
        Higher score = lower variance across ensemble models
        """
        if len(ensemble_probs) < 2:
            return 0.5  # Single model, neutral confidence
        
        # Calculate variance across ensemble predictions
        ensemble_array = np.array(ensemble_probs)
        if ensemble_array.ndim == 1:
            # Single probability (home win)
            variance = np.var(ensemble_array)
        else:
            # Multiple classes (home/draw/away)
            variance = np.mean(np.var(ensemble_array, axis=0))
        
        # Convert variance to confidence (lower variance = higher confidence)
        max_variance = 0.25  # Maximum expected variance
        ensemble_conf = 1 - min(variance, max_variance) / max_variance
        
        return max(0, min(1, ensemble_conf))
    
    def calculate_signal_coverage(self, context_features):
        """
        Calculate confidence based on available context signals
        Higher score = more context features available
        """
        coverage_indicators = []
        
        # Fan sentiment coverage
        fan_cov = context_features.get('fan_cov', 0)
        coverage_indicators.append(fan_cov)
        
        # Referee data coverage
        has_ref_stats = 1 if context_features.get('ref_home_bias', 0) != 0 else 0
        coverage_indicators.append(has_ref_stats)
        
        # Attendance data coverage
        has_attendance = 1 if context_features.get('att_ratio', 0) > 0 else 0
        coverage_indicators.append(has_attendance)
        
        # Travel data coverage
        has_travel = 1 if context_features.get('away_travel_km', 0) > 0 else 0
        coverage_indicators.append(has_travel)
        
        # Calculate average coverage
        signal_cov_conf = np.mean(coverage_indicators)
        
        return max(0, min(1, signal_cov_conf))
    
    def calculate_context_boost(self, context_features):
        """
        Calculate additional confidence boost from strong context signals
        Returns: boost factor (0.0 to 0.2 additional confidence)
        """
        boost_factors = []
        
        # Strong fan sentiment differential
        home_sentiment = context_features.get('home_fan_sent_mean', 0)
        away_sentiment = context_features.get('away_fan_sent_mean', 0)
        sentiment_diff = abs(home_sentiment - away_sentiment)
        if sentiment_diff > 0.3:  # Strong sentiment differential
            boost_factors.append(0.05)
        
        # High attendance (sellout crowd)
        att_ratio = context_features.get('att_ratio', 0)
        if att_ratio > 0.95:
            boost_factors.append(0.03)
        
        # Significant referee bias
        ref_bias = abs(context_features.get('ref_home_bias', 0))
        if ref_bias > 0.05:  # 5%+ bias
            boost_factors.append(0.04)
        
        # Large rest advantage
        rest_diff = abs(context_features.get('rest_diff', 0))
        if rest_diff >= 2:  # 2+ days advantage
            boost_factors.append(0.03)
        
        # Big game atmosphere
        big_game = context_features.get('big_game', 0)
        if big_game:
            boost_factors.append(0.05)
        
        return min(0.2, sum(boost_factors))  # Cap at 20% boost
    
    def calculate_composite_confidence(self, 
                                     model_probs,
                                     context_features,
                                     market_probs=None,
                                     ensemble_probs=None,
                                     cv_scores=None,
                                     true_labels=None):
        """
        Calculate comprehensive confidence score
        Returns: dict with overall confidence and component scores
        """
        
        # Calculate component confidences
        calibrated_conf = self.calculate_calibrated_confidence(
            model_probs, true_labels, cv_scores
        )
        
        market_conf = self.calculate_market_agreement(model_probs, market_probs)
        
        ensemble_conf = self.calculate_ensemble_confidence(
            ensemble_probs or [model_probs]
        )
        
        coverage_conf = self.calculate_signal_coverage(context_features)
        
        # Weighted combination
        component_scores = [calibrated_conf, market_conf, ensemble_conf, coverage_conf]
        weights = self.weights
        
        base_confidence = np.average(component_scores, weights=weights)
        
        # Context boost for strong signals
        context_boost = self.calculate_context_boost(context_features)
        
        # Final confidence (0-100%)
        final_confidence = min(100, (base_confidence + context_boost) * 100)
        
        return {
            'overall_confidence': round(final_confidence, 1),
            'components': {
                'calibrated_confidence': round(calibrated_conf * 100, 1),
                'market_agreement': round(market_conf * 100, 1),
                'ensemble_agreement': round(ensemble_conf * 100, 1),
                'signal_coverage': round(coverage_conf * 100, 1),
                'context_boost': round(context_boost * 100, 1)
            },
            'confidence_factors': self.generate_confidence_factors(
                context_features, component_scores
            )
        }
    
    def generate_confidence_factors(self, context_features, component_scores):
        """Generate human-readable confidence factors"""
        factors = []
        
        # High confidence factors
        if component_scores[0] > 0.7:  # Calibrated confidence
            factors.append("✅ Model well-calibrated on similar matches")
        
        if context_features.get('fan_cov', 0) == 1:
            sentiment_diff = abs(
                context_features.get('home_fan_sent_mean', 0) - 
                context_features.get('away_fan_sent_mean', 0)
            )
            if sentiment_diff > 0.2:
                factors.append("✅ Strong fan sentiment differential detected")
        
        if context_features.get('att_ratio', 0) > 0.95:
            factors.append("✅ Sellout crowd expected (high atmosphere)")
        
        ref_bias = context_features.get('ref_home_bias', 0)
        if abs(ref_bias) > 0.05:
            direction = "home" if ref_bias > 0 else "away"
            factors.append(f"✅ Referee shows {direction} bias ({abs(ref_bias)*100:.1f}%)")
        
        rest_diff = context_features.get('rest_diff', 0)
        if abs(rest_diff) >= 2:
            advantage_team = "home" if rest_diff > 0 else "away"
            factors.append(f"✅ {advantage_team.title()} team has {abs(rest_diff)} day rest advantage")
        
        if context_features.get('big_game', 0):
            factors.append("✅ Big game atmosphere (derby/top6/high stakes)")
        
        # Uncertainty factors
        if context_features.get('fan_cov', 0) == 0:
            factors.append("⚠️ Limited fan sentiment data available")
        
        if component_scores[1] < 0.3:  # Market disagreement
            factors.append("⚠️ Model disagrees significantly with market odds")
        
        travel_km = context_features.get('away_travel_km', 0)
        if travel_km > 300:
            factors.append(f"⚠️ Long away travel ({travel_km:.0f}km) may impact performance")
        
        return factors[:6]  # Limit to top 6 factors
    
    def get_confidence_breakdown(self, confidence_result):
        """
        Generate detailed confidence breakdown for UI
        Returns: formatted breakdown for display
        """
        overall = confidence_result['overall_confidence']
        components = confidence_result['components']
        factors = confidence_result['confidence_factors']
        
        # Confidence level description
        if overall >= 80:
            level = "Very High"
            color = "green"
        elif overall >= 65:
            level = "High" 
            color = "blue"
        elif overall >= 50:
            level = "Medium"
            color = "yellow"
        else:
            level = "Low"
            color = "red"
        
        breakdown = {
            'level': level,
            'color': color,
            'percentage': overall,
            'components': components,
            'factors': factors,
            'interpretation': self.interpret_confidence(overall)
        }
        
        return breakdown
    
    def interpret_confidence(self, confidence):
        """Provide interpretation of confidence level"""
        if confidence >= 80:
            return "Strong prediction with multiple supporting factors"
        elif confidence >= 65:
            return "Good prediction backed by solid evidence"
        elif confidence >= 50:
            return "Moderate confidence with some uncertainty"
        else:
            return "Low confidence due to limited data or conflicting signals"

# Example usage and testing
if __name__ == "__main__":
    # Sample context features
    sample_context = {
        'fan_cov': 1,
        'home_fan_sent_mean': 0.4,
        'away_fan_sent_mean': -0.2,
        'att_ratio': 0.98,
        'ref_home_bias': 0.08,
        'rest_diff': 3,
        'big_game': 1,
        'away_travel_km': 200
    }
    
    # Sample model predictions
    sample_probs = [0.55, 0.25, 0.20]  # [home, draw, away]
    
    calculator = ConfidenceCalculator()
    
    result = calculator.calculate_composite_confidence(
        model_probs=sample_probs,
        context_features=sample_context,
        ensemble_probs=[
            [0.55, 0.25, 0.20],
            [0.52, 0.28, 0.20],
            [0.58, 0.22, 0.20]
        ]
    )
    
    breakdown = calculator.get_confidence_breakdown(result)
    
    print("🎯 Confidence Analysis:")
    print(f"   Overall: {breakdown['percentage']}% ({breakdown['level']})")
    print(f"   Interpretation: {breakdown['interpretation']}")
    print("\n📊 Component Breakdown:")
    for component, score in breakdown['components'].items():
        print(f"   {component.replace('_', ' ').title()}: {score}%")
    print("\n🔍 Key Factors:")
    for factor in breakdown['factors']:
        print(f"   {factor}") 
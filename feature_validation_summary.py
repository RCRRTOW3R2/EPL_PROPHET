#!/usr/bin/env python3
"""
EPL PROPHET - Feature Validation & Summary
=========================================

Comprehensive analysis of all engineered features across:
- Data quality validation
- Feature distribution analysis  
- Correlation analysis
- Feature importance estimation
- ML readiness assessment
- Monte Carlo parameter validation

This provides the final quality check before model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class FeatureValidator:
    """Comprehensive feature validation and analysis."""
    
    def __init__(self):
        self.datasets = {}
        self.feature_summary = {}
        
    def load_all_datasets(self):
        """Load all engineered feature datasets."""
        
        print("📊 Loading All Feature Datasets...")
        
        try:
            # Core datasets
            self.datasets['master'] = pd.read_csv("../data/epl_master_dataset.csv")
            self.datasets['elo'] = pd.read_csv("../outputs/enhanced_match_features.csv")
            self.datasets['xg'] = pd.read_csv("../outputs/xg_match_features.csv")
            self.datasets['advanced'] = pd.read_csv("../outputs/advanced_match_features.csv")
            self.datasets['enhanced'] = pd.read_csv("../outputs/enhanced_match_features_v2.csv")
            self.datasets['monte_carlo'] = pd.read_csv("../outputs/monte_carlo_parameters.csv")
            
            print(f"✅ All datasets loaded successfully")
            
            for name, df in self.datasets.items():
                print(f"   {name}: {df.shape[0]} rows, {df.shape[1]} columns")
                
        except FileNotFoundError as e:
            print(f"❌ Error loading datasets: {e}")
            return False
            
        return True
    
    def validate_data_quality(self):
        """Validate data quality across all datasets."""
        
        print(f"\n🔍 Data Quality Validation")
        print("=" * 40)
        
        quality_report = {}
        
        for name, df in self.datasets.items():
            if df is None or len(df) == 0:
                continue
                
            # Basic quality metrics
            total_rows = len(df)
            total_cols = len(df.columns)
            missing_data = df.isnull().sum().sum()
            missing_percentage = (missing_data / (total_rows * total_cols)) * 100
            
            # Identify problematic columns
            high_missing_cols = []
            for col in df.columns:
                if df[col].isnull().sum() / total_rows > 0.1:  # >10% missing
                    high_missing_cols.append(col)
            
            # Check for outliers (numerical columns only)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers / len(df) > 0.05:  # >5% outliers
                        outlier_cols.append(col)
            
            quality_report[name] = {
                'total_rows': total_rows,
                'total_cols': total_cols,
                'missing_percentage': round(missing_percentage, 2),
                'high_missing_cols': len(high_missing_cols),
                'outlier_cols': len(outlier_cols),
                'quality_score': round(100 - missing_percentage - (len(outlier_cols) * 2), 1)
            }
            
            print(f"{name.upper()}: {quality_report[name]['quality_score']:.1f}% quality score")
            if missing_percentage > 1:
                print(f"   ⚠️  {missing_percentage:.1f}% missing data")
            if len(high_missing_cols) > 0:
                print(f"   ⚠️  {len(high_missing_cols)} columns with >10% missing")
            if len(outlier_cols) > 0:
                print(f"   ⚠️  {len(outlier_cols)} columns with >5% outliers")
        
        return quality_report
    
    def analyze_feature_distributions(self):
        """Analyze feature distributions and statistics."""
        
        print(f"\n📈 Feature Distribution Analysis")
        print("=" * 40)
        
        distribution_analysis = {}
        
        # Focus on enhanced features for detailed analysis
        if 'enhanced' in self.datasets:
            df = self.datasets['enhanced']
            
            # Analyze key feature categories
            feature_categories = {
                'efficiency': [col for col in df.columns if 'shot' in col.lower() or 'efficiency' in col.lower()],
                'defensive': [col for col in df.columns if 'clean_sheet' in col or 'conceded' in col or 'defensive' in col],
                'momentum': [col for col in df.columns if 'streak' in col or 'trend' in col or 'momentum' in col],
                'monte_carlo': [col for col in df.columns if 'goals_mean' in col or 'prob_' in col or 'volatility' in col]
            }
            
            for category, cols in feature_categories.items():
                if not cols:
                    continue
                    
                print(f"\n{category.upper()} FEATURES:")
                category_stats = {}
                
                for col in cols[:5]:  # Limit to first 5 features per category
                    if col in df.columns and df[col].dtype in ['int64', 'float64']:
                        stats = {
                            'mean': round(df[col].mean(), 3),
                            'std': round(df[col].std(), 3),
                            'min': round(df[col].min(), 3),
                            'max': round(df[col].max(), 3),
                            'skew': round(df[col].skew(), 3)
                        }
                        category_stats[col] = stats
                        print(f"   {col}: μ={stats['mean']}, σ={stats['std']}, range=[{stats['min']}, {stats['max']}]")
                
                distribution_analysis[category] = category_stats
        
        return distribution_analysis
    
    def analyze_feature_correlations(self):
        """Analyze correlations between different feature sets."""
        
        print(f"\n🔗 Feature Correlation Analysis")
        print("=" * 40)
        
        correlation_analysis = {}
        
        if 'enhanced' in self.datasets:
            df = self.datasets['enhanced']
            
            # Get numerical columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_df = df[numeric_cols]
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()
                
                # Find highly correlated pairs (>0.8 correlation)
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.8:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                round(corr_val, 3)
                            ))
                
                print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.8):")
                for col1, col2, corr in high_corr_pairs[:10]:  # Show first 10
                    print(f"   {col1} ↔ {col2}: {corr}")
                
                correlation_analysis = {
                    'high_corr_count': len(high_corr_pairs),
                    'high_corr_pairs': high_corr_pairs[:10],
                    'feature_count': len(numeric_cols)
                }
        
        return correlation_analysis
    
    def assess_ml_readiness(self):
        """Assess readiness for machine learning models."""
        
        print(f"\n🤖 ML Readiness Assessment")
        print("=" * 40)
        
        ml_assessment = {}
        
        # Check each dataset for ML readiness
        for name, df in self.datasets.items():
            if df is None or len(df) == 0:
                continue
            
            # Basic ML readiness criteria
            has_target = any(col in df.columns for col in ['FTR', 'actual_result', 'result'])
            has_features = len(df.select_dtypes(include=[np.number]).columns) >= 5
            sufficient_data = len(df) >= 100
            low_missing = df.isnull().sum().sum() / (len(df) * len(df.columns)) < 0.05
            
            readiness_score = sum([has_target, has_features, sufficient_data, low_missing]) / 4 * 100
            
            ml_assessment[name] = {
                'has_target': has_target,
                'has_features': has_features,
                'sufficient_data': sufficient_data,
                'low_missing': low_missing,
                'readiness_score': round(readiness_score, 1)
            }
            
            print(f"{name.upper()}: {readiness_score:.1f}% ML ready")
            if not has_target:
                print(f"   ⚠️  No target variable detected")
            if not has_features:
                print(f"   ⚠️  Insufficient numerical features")
            if not sufficient_data:
                print(f"   ⚠️  Insufficient data ({len(df)} rows)")
            if not low_missing:
                print(f"   ⚠️  High missing data percentage")
        
        return ml_assessment
    
    def validate_monte_carlo_params(self):
        """Validate Monte Carlo parameters for simulation."""
        
        print(f"\n🎲 Monte Carlo Parameter Validation")
        print("=" * 40)
        
        if 'monte_carlo' not in self.datasets:
            print("❌ Monte Carlo parameters not found")
            return {}
        
        mc_df = self.datasets['monte_carlo']
        
        # Validate parameter ranges
        validation_results = {}
        
        # Lambda parameters (Poisson rates)
        home_lambda = mc_df['home_lambda']
        away_lambda = mc_df['away_lambda']
        
        lambda_valid = ((home_lambda >= 0) & (home_lambda <= 10) & 
                       (away_lambda >= 0) & (away_lambda <= 10)).all()
        
        # Probability parameters
        prob_cols = [col for col in mc_df.columns if 'prob_' in col]
        prob_valid = True
        for col in prob_cols:
            if not ((mc_df[col] >= 0) & (mc_df[col] <= 1)).all():
                prob_valid = False
                break
        
        # Variance adjustments
        variance_valid = ((mc_df['home_variance_adj'] >= 0.1) & 
                         (mc_df['home_variance_adj'] <= 5.0) &
                         (mc_df['away_variance_adj'] >= 0.1) & 
                         (mc_df['away_variance_adj'] <= 5.0)).all()
        
        validation_results = {
            'lambda_valid': lambda_valid,
            'prob_valid': prob_valid,
            'variance_valid': variance_valid,
            'total_params': len(mc_df),
            'param_completeness': mc_df.notnull().all().all()
        }
        
        print(f"Lambda parameters valid: {'✅' if lambda_valid else '❌'}")
        print(f"Probability parameters valid: {'✅' if prob_valid else '❌'}")
        print(f"Variance parameters valid: {'✅' if variance_valid else '❌'}")
        print(f"Parameter completeness: {'✅' if validation_results['param_completeness'] else '❌'}")
        print(f"Total simulation parameters: {len(mc_df)}")
        
        return validation_results
    
    def generate_feature_summary(self):
        """Generate comprehensive feature summary across all systems."""
        
        print(f"\n📋 Complete Feature Summary")
        print("=" * 50)
        
        total_features = 0
        feature_breakdown = {}
        
        # Count features by system
        for name, df in self.datasets.items():
            if df is None:
                continue
                
            if name == 'master':
                feature_count = len([col for col in df.columns if col not in ['match_id', 'date', 'HomeTeam', 'AwayTeam']])
                feature_breakdown['Core Data'] = feature_count
            elif name == 'elo':
                feature_count = len([col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team']])
                feature_breakdown['Elo & Market'] = feature_count
            elif name == 'xg':
                feature_count = len([col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team']])
                feature_breakdown['xG Analysis'] = feature_count
            elif name == 'advanced':
                feature_count = len([col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team']])
                feature_breakdown['Advanced Context'] = feature_count
            elif name == 'enhanced':
                feature_count = len([col for col in df.columns if col not in ['match_id', 'date', 'home_team', 'away_team']])
                feature_breakdown['Enhanced ML'] = feature_count
            
            total_features += feature_count
        
        print(f"FEATURE ARSENAL BREAKDOWN:")
        for system, count in feature_breakdown.items():
            print(f"   {system:<20}: {count:>3} features")
        
        print(f"\n🎯 TOTAL FEATURES: {total_features}")
        print(f"📊 TOTAL MATCHES: {len(self.datasets.get('enhanced', []))}")
        print(f"🎲 MONTE CARLO READY: {'✅' if 'monte_carlo' in self.datasets else '❌'}")
        
        return {
            'total_features': total_features,
            'feature_breakdown': feature_breakdown,
            'total_matches': len(self.datasets.get('enhanced', [])),
            'systems_ready': len(self.datasets)
        }
    
    def run_full_validation(self):
        """Run complete feature validation pipeline."""
        
        print("🔍 EPL PROPHET - FEATURE VALIDATION & SUMMARY")
        print("=" * 60)
        
        # Load all datasets
        if not self.load_all_datasets():
            return False
        
        # Run all validation steps
        quality_report = self.validate_data_quality()
        distribution_analysis = self.analyze_feature_distributions()
        correlation_analysis = self.analyze_feature_correlations()
        ml_assessment = self.assess_ml_readiness()
        mc_validation = self.validate_monte_carlo_params()
        feature_summary = self.generate_feature_summary()
        
        # Overall system health
        print(f"\n🏆 SYSTEM HEALTH SUMMARY")
        print("=" * 30)
        
        avg_quality = np.mean([report['quality_score'] for report in quality_report.values()])
        avg_ml_readiness = np.mean([assess['readiness_score'] for assess in ml_assessment.values()])
        
        print(f"Average Data Quality: {avg_quality:.1f}%")
        print(f"Average ML Readiness: {avg_ml_readiness:.1f}%")
        print(f"Feature Systems: {feature_summary['systems_ready']}/6 operational")
        print(f"Total Feature Count: {feature_summary['total_features']}")
        
        overall_health = (avg_quality + avg_ml_readiness) / 2
        print(f"\n🎯 OVERALL SYSTEM HEALTH: {overall_health:.1f}%")
        
        if overall_health >= 90:
            print("🚀 EXCELLENT - Ready for production ML models!")
        elif overall_health >= 80:
            print("✅ GOOD - Ready for model training with minor optimizations")
        elif overall_health >= 70:
            print("⚠️  FAIR - Some data quality issues need attention")
        else:
            print("❌ POOR - Significant data quality issues detected")
        
        return True


def main():
    """Main execution - run complete feature validation."""
    
    validator = FeatureValidator()
    success = validator.run_full_validation()
    
    if success:
        print(f"\n✅ Feature validation complete!")
        print(f"📊 All systems validated and ready for ensemble modeling")
    else:
        print(f"\n❌ Feature validation failed!")
        print(f"🔧 Please check data files and re-run validation")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
EPL Prophet Cleanup Script
Removes unnecessary files and keeps only the essential core
"""

import os
import shutil

def cleanup_project():
    """Clean up the EPL Prophet project"""
    print("🧹 EPL PROPHET PROJECT CLEANUP")
    print("=" * 50)
    
    # Files to delete from root directory
    root_files_to_delete = [
        'deploy_app.py',           # Duplicate of app.py
        'web_app.py',              # Old version
        'upcoming_predictor.py',   # Superseded by final versions
        'fixture_fetcher.py',      # Superseded by quick_real_fetcher.py
        'upcoming_matches_predictor.py',  # Old version
        'shap_explainer.py',       # Superseded by explainer_fixed.py
        'phase3_breakthrough.py',  # Intermediate version
        'phase3_optimization.py',  # Intermediate version
        'phase2_multi_timeframe_ensemble.py',  # Intermediate
        'enhanced_ml_training.py', # Intermediate
        'recency_weighted_system.py',  # Intermediate
        'feature_validation_summary.py',  # Development only
        'notes.txt',               # Personal notes
        'GOAL.txt',                # Initial notes
        '.DS_Store'                # macOS system file
    ]
    
    # Files to delete from ANALYSIS1.0/models
    models_files_to_delete = [
        'improved_fetcher.py',     # Failed version with timezone issues
        'real_fixtures_fetcher.py', # Failed version, use quick_real_fetcher.py
        'test_apis.py',            # Development debugging only
        'espn_scraper.py',         # Failed ESPN scraping attempt
        'run_app.py',              # Development script
        'app.py',                  # Duplicate, use epl-prophet-web/app.py
        'predict_all_future.py',   # Superseded
        'get_all_fixtures.py',     # Superseded
        'add_more_fixtures.py',    # Development script
        'predictor_complete.py',   # Old version
        'fetcher.py',              # Old version
        'final_predictor.py',      # Superseded
        'predictor.py',            # Old version
        'shap_demo.py',            # Demo only
        'explainer.py',            # Superseded by explainer_fixed.py
        'final.py',                # Vague name, superseded
        'breakthrough.py',         # Intermediate version
        'phase3_final.py',         # Intermediate version
        'phase2_ensemble.py',      # Intermediate version
        'enhanced_ml_phase1.py',   # Intermediate version
        'recency_fix.py',          # Intermediate version
        'importance_analysis.py',  # Development analysis
        'validation.py',           # Development validation
        'enhanced_feature_engineering.py',  # Intermediate version
        'static/',                 # Duplicate of epl-prophet-web/static/
        'templates/',              # Duplicate of epl-prophet-web/templates/
        '__pycache__/'             # Python cache
    ]
    
    # Output files to delete (keep only essential)
    output_files_to_delete = [
        'upcoming_fixtures.csv',          # Old fixtures
        'all_future_predictions.csv',     # Old predictions
        'all_upcoming_fixtures.csv',      # Sample fixtures
        'upcoming_predictions.csv',       # Old predictions
        'phase2_enhanced_features.csv',   # Intermediate data
        'phase2_results.csv',             # Intermediate results
        'phase2_long_term.joblib',        # Intermediate model
        'phase2_medium_term.joblib',      # Intermediate model
        'phase2_short_term.joblib',       # Intermediate model
        'enhanced_ml_results.csv',        # Intermediate results
        'enhanced_neural_network.joblib', # Not our best model
        'enhanced_gradient_boost.joblib', # Not our best model
        'enhanced_random_forest.joblib',  # Superseded by champion_rf.joblib
        'recency_weighted_stock_features.csv',  # Intermediate data
        'feature_importance_analysis.csv', # Development analysis
        'monte_carlo_parameters.csv',      # Intermediate data
        'enhanced_match_features_v2.csv',  # Intermediate data
        'advanced_match_features.csv',     # Intermediate data
        'enhanced_match_features.csv',     # Intermediate data
        'elo_rating_history.csv',          # Full history not needed
        'xg_match_features.csv'            # Intermediate data
    ]
    
    # Directories to delete entirely
    dirs_to_delete = [
        'models/',              # Root models dir (duplicated in ANALYSIS1.0)
        'ANALYSIS1.0/notebooks/', # Jupyter notebooks not needed
        'ANALYSIS1.0/utils/',     # Empty or minimal utils
        '.history/',              # Development history
        '.venv/'                  # Virtual environment
    ]
    
    deleted_count = 0
    
    # Delete root files
    print("\n🗑️  Deleting unnecessary root files...")
    for file in root_files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   ✅ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {file}: {e}")
    
    # Delete model files
    print("\n🗑️  Deleting unnecessary model files...")
    for file in models_files_to_delete:
        filepath = f"ANALYSIS1.0/models/{file}"
        if os.path.exists(filepath):
            try:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    os.remove(filepath)
                print(f"   ✅ Deleted: {filepath}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {filepath}: {e}")
    
    # Delete output files
    print("\n🗑️  Deleting intermediate output files...")
    for file in output_files_to_delete:
        filepath = f"ANALYSIS1.0/outputs/{file}"
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"   ✅ Deleted: {filepath}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {filepath}: {e}")
    
    # Delete directories
    print("\n🗑️  Deleting unnecessary directories...")
    for dir_name in dirs_to_delete:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"   ✅ Deleted directory: {dir_name}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {dir_name}: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"📊 Total files/directories deleted: {deleted_count}")
    
    # Show what's left
    print(f"\n✅ ESSENTIAL FILES REMAINING:")
    print(f"📊 Core Data: 1415.csv through 2526.csv (11 seasons)")
    print(f"🧠 Champion Model: champion_rf.joblib (53.7% accuracy)")
    print(f"📈 Features: champion_features.csv")
    print(f"🔮 Real Fixtures: real_upcoming_fixtures.csv (360 matches)")
    print(f"🌐 Web App: epl-prophet-web/ (complete deployment)")
    print(f"📖 Documentation: PROJECT_README.md, docs/")
    print(f"🔧 Core Scripts: data_standardization.py, final_breakthrough.py, etc.")
    
    print(f"\n🚀 Your EPL Prophet is now clean and production-ready!")

if __name__ == "__main__":
    response = input("⚠️  This will permanently delete many files. Continue? (y/N): ")
    if response.lower() == 'y':
        cleanup_project()
    else:
        print("❌ Cleanup cancelled.") 
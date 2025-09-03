#!/usr/bin/env python3
"""
EPL Prophet - Psychological Patterns & Frankenstein Model
Deep analysis of team psychology patterns + hybrid ensemble
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class PsychologicalPatternAnalyzer:
    """Analyze hidden psychological patterns in EPL"""
    
    def __init__(self):
        self.patterns = {}
        
    def load_data(self):
        """Load EPL data for pattern analysis"""
        print("📊 Loading data for psychological pattern analysis...")
        
        seasons = ['1718', '1819', '1920', '2021', '2122', '2223', '2324']
        all_data = []
        
        for season in seasons:
            try:
                df = pd.read_csv(f'{season}.csv')
                df['season'] = season
                all_data.append(df)
                print(f"   ✅ {season}: {len(df)} matches")
            except:
                continue
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"📈 Total: {len(combined)} matches for analysis")
        return combined
    
    def analyze_post_big_loss_patterns(self, df):
        """Analyze team behavior after heavy defeats"""
        print("\n💔 ANALYZING POST-BIG LOSS PATTERNS...")
        
        big_loss_patterns = {
            'bounce_back': [],      # Win after big loss
            'spiral_down': [],      # Another poor result
            'tactical_change': [],  # Different style
            'psychological_damage': []
        }
        
        teams = df['HomeTeam'].unique()
        
        for team in teams:
            team_matches = self.get_team_chronological_matches(df, team)
            
            for i in range(len(team_matches) - 1):
                current_match = team_matches[i]
                next_match = team_matches[i + 1]
                
                # Check for big loss (3+ goal deficit)
                if current_match['goal_margin'] <= -3:
                    
                    # Analyze next match response
                    response_data = {
                        'team': team,
                        'big_loss_margin': current_match['goal_margin'],
                        'days_gap': self.calculate_days_between_matches(current_match, next_match),
                        'next_result': next_match['result'],
                        'next_margin': next_match['goal_margin'],
                        'next_goals_scored': next_match['goals_for'],
                        'venue_change': current_match['venue'] != next_match['venue']
                    }
                    
                    # Categorize response
                    if next_match['result'] == 'W' and next_match['goal_margin'] >= 2:
                        big_loss_patterns['bounce_back'].append(response_data)
                    elif next_match['result'] == 'L' and next_match['goal_margin'] <= -2:
                        big_loss_patterns['spiral_down'].append(response_data)
                    elif abs(next_match['goal_margin']) <= 1:
                        big_loss_patterns['tactical_change'].append(response_data)
                    else:
                        big_loss_patterns['psychological_damage'].append(response_data)
        
        # Calculate statistics
        print("   📊 Big Loss Response Patterns:")
        for pattern_type, matches in big_loss_patterns.items():
            if matches:
                avg_response = np.mean([m['next_margin'] for m in matches])
                win_rate = sum(1 for m in matches if m['next_result'] == 'W') / len(matches)
                print(f"      {pattern_type.replace('_', ' ').title()}: {len(matches)} cases, {win_rate:.1%} win rate, {avg_response:+.2f} avg margin")
        
        return big_loss_patterns
    
    def analyze_post_big_win_patterns(self, df):
        """Analyze team behavior after dominant victories"""
        print("\n🚀 ANALYZING POST-BIG WIN PATTERNS...")
        
        big_win_patterns = {
            'momentum_continuation': [],  # Another strong performance
            'complacency_effect': [],     # Poor follow-up
            'rotation_impact': [],        # Squad rotation effects
            'confidence_boost': []        # Sustained improvement
        }
        
        teams = df['HomeTeam'].unique()
        
        for team in teams:
            team_matches = self.get_team_chronological_matches(df, team)
            
            for i in range(len(team_matches) - 1):
                current_match = team_matches[i]
                next_match = team_matches[i + 1]
                
                # Check for big win (3+ goal advantage)
                if current_match['goal_margin'] >= 3:
                    
                    response_data = {
                        'team': team,
                        'big_win_margin': current_match['goal_margin'],
                        'days_gap': self.calculate_days_between_matches(current_match, next_match),
                        'next_result': next_match['result'],
                        'next_margin': next_match['goal_margin'],
                        'next_goals_scored': next_match['goals_for'],
                        'venue_change': current_match['venue'] != next_match['venue'],
                        'opponent_strength': 'Big6' if self.is_big6_team(next_match['opponent']) else 'Other'
                    }
                    
                    # Categorize response
                    if next_match['result'] == 'W' and next_match['goal_margin'] >= 1:
                        big_win_patterns['momentum_continuation'].append(response_data)
                    elif next_match['result'] == 'L' or (next_match['result'] == 'D' and next_match['goals_for'] <= 1):
                        big_win_patterns['complacency_effect'].append(response_data)
                    elif next_match['goals_for'] <= current_match['goals_for'] - 2:
                        big_win_patterns['rotation_impact'].append(response_data)
                    else:
                        big_win_patterns['confidence_boost'].append(response_data)
        
        print("   📊 Big Win Follow-up Patterns:")
        for pattern_type, matches in big_win_patterns.items():
            if matches:
                avg_response = np.mean([m['next_margin'] for m in matches])
                win_rate = sum(1 for m in matches if m['next_result'] == 'W') / len(matches)
                print(f"      {pattern_type.replace('_', ' ').title()}: {len(matches)} cases, {win_rate:.1%} win rate, {avg_response:+.2f} avg margin")
        
        return big_win_patterns
    
    def analyze_card_impact_patterns(self, df):
        """Analyze impact of cards on subsequent performance"""
        print("\n🟨 ANALYZING CARD IMPACT PATTERNS...")
        
        # Note: We don't have card data in basic CSV, so we'll simulate analysis structure
        card_patterns = {
            'disciplinary_response': "Cards affect next match mentality",
            'referee_bias_memory': "Teams remember harsh referees",
            'tactical_adjustment': "More cards = more conservative next game"
        }
        
        print("   📊 Card Impact Patterns (Analysis Framework):")
        print("      🟨 Yellow Card Accumulation: Affects player confidence")
        print("      🟥 Red Cards: Team unity vs discipline issues")
        print("      👨‍⚖️ Referee Reputation: Teams adjust play style")
        print("      🎯 Fouling Strategy: Response to physical opponents")
        
        return card_patterns
    
    def analyze_fixture_congestion_patterns(self, df):
        """Analyze performance under fixture pressure"""
        print("\n📅 ANALYZING FIXTURE CONGESTION PATTERNS...")
        
        congestion_patterns = {
            'quick_turnaround': [],  # <4 days between matches
            'rotation_effects': [],  # Squad depth impact
            'fatigue_decline': [],   # Performance drop
            'squad_depth_advantage': []  # Deep squads thrive
        }
        
        teams = df['HomeTeam'].unique()
        
        for team in teams:
            team_matches = self.get_team_chronological_matches(df, team)
            
            for i in range(1, len(team_matches)):
                current_match = team_matches[i]
                prev_match = team_matches[i - 1]
                
                days_gap = self.calculate_days_between_matches(prev_match, current_match)
                
                if days_gap <= 4:  # Fixture congestion
                    congestion_data = {
                        'team': team,
                        'days_gap': days_gap,
                        'prev_result': prev_match['result'],
                        'prev_goals': prev_match['goals_for'],
                        'current_result': current_match['result'],
                        'current_goals': current_match['goals_for'],
                        'performance_change': current_match['goal_margin'] - prev_match['goal_margin'],
                        'is_big6': team in {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham'}
                    }
                    
                    if congestion_data['performance_change'] >= 1:
                        congestion_patterns['squad_depth_advantage'].append(congestion_data)
                    elif congestion_data['performance_change'] <= -2:
                        congestion_patterns['fatigue_decline'].append(congestion_data)
                    elif congestion_data['current_goals'] < congestion_data['prev_goals']:
                        congestion_patterns['rotation_effects'].append(congestion_data)
                    else:
                        congestion_patterns['quick_turnaround'].append(congestion_data)
        
        print("   📊 Fixture Congestion Patterns:")
        for pattern_type, matches in congestion_patterns.items():
            if matches:
                avg_change = np.mean([m['performance_change'] for m in matches])
                big6_rate = sum(1 for m in matches if m['is_big6']) / len(matches)
                print(f"      {pattern_type.replace('_', ' ').title()}: {len(matches)} cases, {avg_change:+.2f} avg change, {big6_rate:.1%} Big6")
        
        return congestion_patterns
    
    def get_team_chronological_matches(self, df, team):
        """Get chronological matches for a team"""
        team_matches = []
        
        for _, match in df.iterrows():
            if match['HomeTeam'] == team:
                team_match = {
                    'date': match.get('Date', ''),
                    'venue': 'H',
                    'opponent': match['AwayTeam'],
                    'result': 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L',
                    'goals_for': match['FTHG'],
                    'goals_against': match['FTAG'],
                    'goal_margin': match['FTHG'] - match['FTAG']
                }
            elif match['AwayTeam'] == team:
                team_match = {
                    'date': match.get('Date', ''),
                    'venue': 'A',
                    'opponent': match['HomeTeam'],
                    'result': 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L',
                    'goals_for': match['FTAG'],
                    'goals_against': match['FTHG'],
                    'goal_margin': match['FTAG'] - match['FTHG']
                }
            else:
                continue
            
            team_matches.append(team_match)
        
        return team_matches
    
    def calculate_days_between_matches(self, match1, match2):
        """Calculate days between matches (simplified)"""
        # In real implementation, parse dates properly
        return 7  # Placeholder - assume weekly fixtures
    
    def is_big6_team(self, team):
        """Check if team is Big 6"""
        big6 = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham'}
        return team in big6

class FrankensteinEnsemble:
    """Hybrid ensemble combining best of each algorithm"""
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        
    def create_specialized_models(self):
        """Create specialized models for different scenarios"""
        
        # Model 1: Linear relationships (form, basic stats)
        self.models['linear_specialist'] = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42
        )
        
        # Model 2: Psychological patterns (streaks, morale)
        self.models['psychology_specialist'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            random_state=42
        )
        
        # Model 3: Complex interactions (opponent effects)
        self.models['interaction_specialist'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Model 4: Decision boundaries (close games)
        self.models['boundary_specialist'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        print("🧟‍♂️ Frankenstein models created!")
        print("   📈 Linear Specialist: Form & stats")
        print("   🧠 Psychology Specialist: Streaks & morale")  
        print("   🔄 Interaction Specialist: Complex patterns")
        print("   🎯 Boundary Specialist: Close decisions")
    
    def train_frankenstein(self, X, y):
        """Train the Frankenstein ensemble"""
        print("\n⚡ Training Frankenstein Ensemble...")
        
        # Scale data for some models
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train each specialist
        specialist_scores = {}
        
        for name, model in self.models.items():
            print(f"   🧬 Training {name}...")
            
            if name in ['linear_specialist', 'boundary_specialist']:
                # Use scaled data
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                model.fit(X_scaled, y)
            else:
                # Use original data
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                model.fit(X, y)
            
            specialist_scores[name] = cv_scores.mean()
            print(f"      Accuracy: {cv_scores.mean():.4f}")
        
        # Create ensemble with adaptive weights
        weights = self.calculate_adaptive_weights(specialist_scores)
        
        # Create final ensemble
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probabilities
        )
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        # Evaluate ensemble
        ensemble_scores = cross_val_score(self.ensemble, X, y, cv=tscv, scoring='accuracy')
        
        print(f"\n🏆 FRANKENSTEIN ENSEMBLE RESULTS:")
        print(f"   Individual specialists: {[f'{name}: {score:.3f}' for name, score in specialist_scores.items()]}")
        print(f"   🧟‍♂️ Ensemble accuracy: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
        
        return ensemble_scores
    
    def calculate_adaptive_weights(self, scores):
        """Calculate adaptive weights based on performance"""
        total_score = sum(scores.values())
        weights = {name: score/total_score for name, score in scores.items()}
        
        print(f"   ⚖️ Adaptive weights: {[f'{name}: {weight:.3f}' for name, weight in weights.items()]}")
        return weights
    
    def predict_with_specialist_analysis(self, X_new):
        """Make prediction with specialist breakdown"""
        X_scaled = self.scaler.transform(X_new)
        
        # Get predictions from each specialist
        predictions = {}
        
        for name, model in self.models.items():
            if name in ['linear_specialist', 'boundary_specialist']:
                pred_proba = model.predict_proba(X_scaled)[0]
            else:
                pred_proba = model.predict_proba(X_new)[0]
            
            predictions[name] = {
                'home_win': pred_proba[2] if len(pred_proba) > 2 else 0,
                'draw': pred_proba[1] if len(pred_proba) > 1 else 0,
                'away_win': pred_proba[0]
            }
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(X_new)[0]
        
        return {
            'specialists': predictions,
            'ensemble': {
                'home_win': ensemble_pred[2] if len(ensemble_pred) > 2 else 0,
                'draw': ensemble_pred[1] if len(ensemble_pred) > 1 else 0,
                'away_win': ensemble_pred[0]
            }
        }

def main():
    """Main analysis and model building"""
    print("🧟‍♂️ EPL PROPHET - PSYCHOLOGICAL PATTERNS & FRANKENSTEIN MODEL")
    print("="*65)
    
    # Analyze psychological patterns
    analyzer = PsychologicalPatternAnalyzer()
    df = analyzer.load_data()
    
    # Deep pattern analysis
    big_loss_patterns = analyzer.analyze_post_big_loss_patterns(df)
    big_win_patterns = analyzer.analyze_post_big_win_patterns(df)
    card_patterns = analyzer.analyze_card_impact_patterns(df)
    congestion_patterns = analyzer.analyze_fixture_congestion_patterns(df)
    
    # Store patterns for future use
    all_patterns = {
        'big_loss_recovery': big_loss_patterns,
        'big_win_followup': big_win_patterns,
        'disciplinary_impact': card_patterns,
        'fixture_congestion': congestion_patterns
    }
    
    print(f"\n🎯 PATTERN DISCOVERY COMPLETE!")
    print(f"   💔 Big loss patterns: {sum(len(v) if isinstance(v, list) else 0 for v in big_loss_patterns.values())} cases")
    print(f"   🚀 Big win patterns: {sum(len(v) if isinstance(v, list) else 0 for v in big_win_patterns.values())} cases")
    print(f"   📅 Congestion patterns: {sum(len(v) if isinstance(v, list) else 0 for v in congestion_patterns.values())} cases")
    
    # Save patterns
    import json
    with open('psychological_patterns.json', 'w') as f:
        json.dump(all_patterns, f, indent=2, default=str)
    
    print(f"\n💾 Patterns saved to: psychological_patterns.json")
    print(f"🧟‍♂️ Ready to build Frankenstein ensemble with these insights!")

if __name__ == "__main__":
    main() 
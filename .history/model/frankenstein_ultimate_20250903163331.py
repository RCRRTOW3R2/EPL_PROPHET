#!/usr/bin/env python3
"""
EPL Prophet - Frankenstein Ultimate Model
Combines best of all algorithms with psychological pattern insights
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

class FrankensteinUltimate:
    """Ultimate ensemble with psychological pattern integration"""
    
    def __init__(self):
        self.specialists = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Load psychological patterns
        self.load_psychological_patterns()
        
    def load_psychological_patterns(self):
        """Load discovered psychological patterns"""
        try:
            with open('psychological_patterns.json', 'r') as f:
                self.patterns = json.load(f)
            print("✅ Loaded psychological patterns")
        except:
            print("⚠️ Using default psychological patterns")
            self.patterns = {'default': True}
    
    def load_and_prepare_data(self):
        """Load data with psychological feature engineering"""
        print("\n📊 Loading data with psychological features...")
        
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
        
        # Create enhanced features with psychological insights
        features_df = self.create_psychological_features(combined)
        
        return features_df
    
    def create_psychological_features(self, df):
        """Create features including psychological patterns"""
        print("🧠 Creating psychological features...")
        
        features_list = []
        
        for idx, match in df.iterrows():
            if idx < 20:  # Need history
                continue
                
            try:
                features = self.extract_match_features(match, df, idx)
                if features:
                    features_list.append(features)
            except:
                continue
        
        features_df = pd.DataFrame(features_list)
        print(f"✅ Created {len(features_df)} matches with {features_df.shape[1]} features")
        
        return features_df
    
    def extract_match_features(self, match, df, idx):
        """Extract enhanced features for one match"""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Get recent matches
        home_recent = self.get_recent_matches(df, home_team, idx, n=6)
        away_recent = self.get_recent_matches(df, away_team, idx, n=6)
        
        if len(home_recent) < 4 or len(away_recent) < 4:
            return None
        
        features = {}
        
        # 1. BASIC PERFORMANCE FEATURES
        features.update(self.get_basic_features(home_recent, away_recent))
        
        # 2. PSYCHOLOGICAL PATTERN FEATURES (NEW!)
        features.update(self.get_psychological_features(home_recent, away_recent))
        
        # 3. FORM & MOMENTUM  
        features.update(self.get_momentum_features(home_recent, away_recent))
        
        # 4. OPPONENT PSYCHOLOGY
        features.update(self.get_opponent_psychology(home_team, away_team))
        
        # 5. LOGARITHMIC RATIOS
        features.update(self.get_log_ratios(home_recent, away_recent))
        
        # TARGET
        if match['FTR'] == 'H':
            features['target'] = 2
        elif match['FTR'] == 'A':
            features['target'] = 0
        else:
            features['target'] = 1
        
        return features
    
    def get_recent_matches(self, df, team, current_idx, n=6):
        """Get recent matches for a team"""
        team_matches = []
        
        for i in range(current_idx - 1, -1, -1):
            if len(team_matches) >= n:
                break
            
            prev_match = df.iloc[i]
            
            if prev_match['HomeTeam'] == team:
                result = 'W' if prev_match['FTR'] == 'H' else 'D' if prev_match['FTR'] == 'D' else 'L'
                goals_for = prev_match['FTHG']
                goals_against = prev_match['FTAG']
            elif prev_match['AwayTeam'] == team:
                result = 'W' if prev_match['FTR'] == 'A' else 'D' if prev_match['FTR'] == 'D' else 'L'
                goals_for = prev_match['FTAG']
                goals_against = prev_match['FTHG']
            else:
                continue
            
            team_matches.append({
                'result': result,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_margin': goals_for - goals_against
            })
        
        return team_matches
    
    def get_basic_features(self, home_recent, away_recent):
        """Basic performance features"""
        home_goals_avg = np.mean([m['goals_for'] for m in home_recent])
        away_goals_avg = np.mean([m['goals_for'] for m in away_recent])
        home_conceded_avg = np.mean([m['goals_against'] for m in home_recent])
        away_conceded_avg = np.mean([m['goals_against'] for m in away_recent])
        
        home_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in home_recent) / len(home_recent)
        away_ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in away_recent) / len(away_recent)
        
        return {
            'home_goals_avg': home_goals_avg,
            'away_goals_avg': away_goals_avg,
            'home_conceded_avg': home_conceded_avg,
            'away_conceded_avg': away_conceded_avg,
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'form_difference': home_ppg - away_ppg
        }
    
    def get_psychological_features(self, home_recent, away_recent):
        """NEW: Psychological pattern features based on discoveries"""
        features = {}
        
        # Check for big loss/win in last match
        home_last_margin = home_recent[0]['goal_margin'] if home_recent else 0
        away_last_margin = away_recent[0]['goal_margin'] if away_recent else 0
        
        # POST-BIG LOSS PATTERNS
        features['home_post_big_loss'] = 1 if home_last_margin <= -3 else 0
        features['away_post_big_loss'] = 1 if away_last_margin <= -3 else 0
        
        # Bounce back probability (teams that bounce back after big losses)
        features['home_bounce_back_likely'] = 1 if (home_last_margin <= -3 and 
                                                   self.is_bounce_back_team(home_recent)) else 0
        features['away_bounce_back_likely'] = 1 if (away_last_margin <= -3 and 
                                                   self.is_bounce_back_team(away_recent)) else 0
        
        # Spiral down risk (teams that collapse after big losses)
        features['home_spiral_risk'] = 1 if (home_last_margin <= -3 and 
                                            self.is_spiral_team(home_recent)) else 0
        features['away_spiral_risk'] = 1 if (away_last_margin <= -3 and 
                                            self.is_spiral_team(away_recent)) else 0
        
        # POST-BIG WIN PATTERNS
        features['home_post_big_win'] = 1 if home_last_margin >= 3 else 0
        features['away_post_big_win'] = 1 if away_last_margin >= 3 else 0
        
        # Momentum continuation (teams that keep winning after big wins)
        features['home_momentum_likely'] = 1 if (home_last_margin >= 3 and 
                                                self.is_momentum_team(home_recent)) else 0
        features['away_momentum_likely'] = 1 if (away_last_margin >= 3 and 
                                                self.is_momentum_team(away_recent)) else 0
        
        # Complacency risk (teams that get overconfident)
        features['home_complacency_risk'] = 1 if (home_last_margin >= 3 and 
                                                 self.is_complacency_team(home_recent)) else 0
        features['away_complacency_risk'] = 1 if (away_last_margin >= 3 and 
                                                 self.is_complacency_team(away_recent)) else 0
        
        # PSYCHOLOGICAL MOMENTUM SCORE
        features['home_psych_momentum'] = self.calculate_psychological_momentum(home_recent)
        features['away_psych_momentum'] = self.calculate_psychological_momentum(away_recent)
        
        return features
    
    def is_bounce_back_team(self, recent_matches):
        """Identify teams likely to bounce back from big losses"""
        # Look for pattern of resilience in recent history
        big_loss_responses = []
        for i in range(len(recent_matches) - 1):
            if recent_matches[i+1]['goal_margin'] <= -3:  # Previous match was big loss
                big_loss_responses.append(recent_matches[i]['result'])
        
        if not big_loss_responses:
            return False
        
        # If team has mostly won after big losses, they're a bounce-back team
        win_rate_after_loss = sum(1 for r in big_loss_responses if r == 'W') / len(big_loss_responses)
        return win_rate_after_loss > 0.6
    
    def is_spiral_team(self, recent_matches):
        """Identify teams that spiral after big losses"""
        # Look for pattern of consecutive poor results
        recent_results = [m['result'] for m in recent_matches[:3]]
        poor_results = sum(1 for r in recent_results if r == 'L')
        return poor_results >= 2
    
    def is_momentum_team(self, recent_matches):
        """Identify teams that maintain momentum after big wins"""
        win_streak = 0
        for match in recent_matches:
            if match['result'] == 'W':
                win_streak += 1
            else:
                break
        return win_streak >= 3
    
    def is_complacency_team(self, recent_matches):
        """Identify teams prone to complacency"""
        # Teams with inconsistent results after good performances
        if len(recent_matches) < 4:
            return False
        
        # Look for pattern: Win -> Poor result
        inconsistent_count = 0
        for i in range(len(recent_matches) - 1):
            if recent_matches[i+1]['result'] == 'W' and recent_matches[i]['result'] == 'L':
                inconsistent_count += 1
        
        return inconsistent_count >= 2
    
    def calculate_psychological_momentum(self, recent_matches):
        """Calculate psychological momentum score (1-10)"""
        if not recent_matches:
            return 5.0
        
        # Base score from recent form
        ppg = sum(3 if m['result'] == 'W' else 1 if m['result'] == 'D' else 0 for m in recent_matches) / len(recent_matches)
        base_score = (ppg / 3.0) * 5.0 + 2.5  # Scale to 2.5-7.5
        
        # Momentum modifiers
        last_result = recent_matches[0]['result']
        last_margin = recent_matches[0]['goal_margin']
        
        # Big win boost
        if last_margin >= 3:
            base_score += 1.5
        elif last_margin >= 2:
            base_score += 0.8
        
        # Big loss penalty
        elif last_margin <= -3:
            base_score -= 1.5
        elif last_margin <= -2:
            base_score -= 0.8
        
        # Streak effects
        streak_length = self.calculate_streak_length(recent_matches)
        if recent_matches[0]['result'] == 'W' and streak_length >= 3:
            base_score += 0.5
        elif recent_matches[0]['result'] == 'L' and streak_length >= 3:
            base_score -= 0.5
        
        return max(1.0, min(10.0, base_score))
    
    def calculate_streak_length(self, recent_matches):
        """Calculate current streak length"""
        if not recent_matches:
            return 0
        
        current_result = recent_matches[0]['result']
        streak = 1
        
        for match in recent_matches[1:]:
            if match['result'] == current_result:
                streak += 1
            else:
                break
        
        return streak
    
    def get_momentum_features(self, home_recent, away_recent):
        """Momentum and streak features"""
        home_streak = self.calculate_streak_length(home_recent)
        away_streak = self.calculate_streak_length(away_recent)
        
        return {
            'home_win_streak': home_streak if home_recent[0]['result'] == 'W' else 0,
            'away_win_streak': away_streak if away_recent[0]['result'] == 'W' else 0,
            'home_loss_streak': home_streak if home_recent[0]['result'] == 'L' else 0,
            'away_loss_streak': away_streak if away_recent[0]['result'] == 'L' else 0
        }
    
    def get_opponent_psychology(self, home_team, away_team):
        """Opponent psychology features"""
        big6_teams = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham'}
        
        return {
            'home_vs_big6': 1 if away_team in big6_teams else 0,
            'away_vs_big6': 1 if home_team in big6_teams else 0,
            'big6_clash': 1 if (home_team in big6_teams and away_team in big6_teams) else 0
        }
    
    def get_log_ratios(self, home_recent, away_recent):
        """Logarithmic ratio features"""
        home_goals_avg = np.mean([m['goals_for'] for m in home_recent])
        away_goals_avg = np.mean([m['goals_for'] for m in away_recent])
        home_conceded_avg = np.mean([m['goals_against'] for m in home_recent])
        away_conceded_avg = np.mean([m['goals_against'] for m in away_recent])
        
        return {
            'log_goals_ratio': np.log((home_goals_avg + 1) / (away_goals_avg + 1)),
            'log_conceded_ratio': np.log((away_conceded_avg + 1) / (home_conceded_avg + 1))
        }
    
    def create_specialists(self):
        """Create specialized models"""
        print("\n🧟‍♂️ Creating Frankenstein specialists...")
        
        # Linear specialist for basic relationships
        self.specialists['linear'] = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        
        # Psychology specialist for patterns
        self.specialists['psychology'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=8, random_state=42
        )
        
        # Interaction specialist for complex patterns
        self.specialists['interaction'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss'
        )
        
        # Boundary specialist for close games
        self.specialists['boundary'] = SVC(
            kernel='rbf', probability=True, random_state=42
        )
        
        print("   📈 Linear: Basic stats & form")
        print("   🧠 Psychology: Patterns & momentum")
        print("   🔄 Interaction: Complex relationships")
        print("   🎯 Boundary: Close decisions")
    
    def train_frankenstein(self, features_df):
        """Train the Frankenstein ensemble"""
        print("\n⚡ Training Frankenstein Ultimate...")
        
        # Prepare data
        X = features_df.drop(['target'], axis=1)
        y = features_df['target']
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale for some models
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = list(X.columns)
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train specialists
        specialist_scores = {}
        
        for name, model in self.specialists.items():
            print(f"   🧬 Training {name} specialist...")
            
            if name in ['linear', 'boundary']:
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
                model.fit(X_scaled, y)
            else:
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                model.fit(X, y)
            
            specialist_scores[name] = cv_scores.mean()
            print(f"      Accuracy: {cv_scores.mean():.4f}")
        
        # Create adaptive ensemble
        estimators = [(name, model) for name, model in self.specialists.items()]
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        # Evaluate
        ensemble_scores = cross_val_score(self.ensemble, X, y, cv=tscv, scoring='accuracy')
        
        print(f"\n🏆 FRANKENSTEIN RESULTS:")
        print(f"   Specialists: {[f'{name}: {score:.3f}' for name, score in specialist_scores.items()]}")
        print(f"   🧟‍♂️ Ensemble: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
        
        # Feature importance from psychology specialist
        if hasattr(self.specialists['psychology'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.specialists['psychology'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔝 TOP PSYCHOLOGICAL FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(8).iterrows()):
                print(f"   {i+1}. {row['feature']:<25} {row['importance']:.4f}")
        
        return ensemble_scores
    
    def save_frankenstein(self, filepath="models/frankenstein_ultimate.pkl"):
        """Save the Frankenstein model"""
        model_data = {
            'specialists': self.specialists,
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'patterns': self.patterns
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Frankenstein saved: {filepath}")

def main():
    """Build the ultimate Frankenstein model"""
    print("🧟‍♂️ EPL PROPHET - FRANKENSTEIN ULTIMATE")
    print("="*45)
    
    # Initialize
    frankenstein = FrankensteinUltimate()
    
    # Load data with psychological features
    features_df = frankenstein.load_and_prepare_data()
    
    # Create specialists
    frankenstein.create_specialists()
    
    # Train the monster
    scores = frankenstein.train_frankenstein(features_df)
    
    # Save the creation
    frankenstein.save_frankenstein()
    
    print(f"\n🧟‍♂️ FRANKENSTEIN IS ALIVE!")
    print(f"   🎯 Accuracy: {scores.mean():.1%}")
    print(f"   🧠 Psychological patterns: INTEGRATED")
    print(f"   ⚡ Multi-specialist ensemble: ACTIVE")
    print(f"   🏆 Ready to dominate EPL predictions!")

if __name__ == "__main__":
    main() 
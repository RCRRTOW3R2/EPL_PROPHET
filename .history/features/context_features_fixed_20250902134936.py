#!/usr/bin/env python3
"""
EPL Prophet - Context Features Engine (Fixed Reddit Auth)
Advanced feature engineering with fan sentiment, travel burden, crowd dynamics, and referee psychology
"""

import pandas as pd
import numpy as np
import praw
import os
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yaml
import warnings
warnings.filterwarnings('ignore')

class ContextFeaturesEngine:
    """Extract orthogonal signals to boost EPL Prophet beyond 53.7% accuracy"""
    
    def __init__(self, config_path="config/context_config.yaml"):
        self.config = self.load_config(config_path)
        self.reddit = self.init_reddit()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def load_config(self, config_path):
        """Load configuration for context features"""
        default_config = {
            'reddit': {
                'client_id': '8TMi7XDN6MlTQrRQp0oikQ',
                'client_secret': 'K7jt8_J5ZSRg0EaAxOTKMLBWyeXH-A',
                'user_agent': 'EPL_Prophet_v1.0'
            },
            'sentiment': {
                'model': 'vader',
                'toxicity_threshold': 0.8,
                'time_window_hours': 48
            },
            'subreddits': {
                'Arsenal': 'Gunners',
                'Chelsea': 'chelseafc', 
                'Liverpool': 'LiverpoolFC',
                'Manchester City': 'MCFC',
                'Manchester United': 'reddevils',
                'Tottenham': 'coys',
                'Newcastle': 'NUFC'
            },
            'confidence_weights': [0.35, 0.25, 0.25, 0.15]
        }
        
        # Try to load from credentials file
        try:
            import sys
            sys.path.append('config')
            import reddit_credentials
            default_config['reddit']['client_id'] = reddit_credentials.REDDIT_CLIENT_ID
            default_config['reddit']['client_secret'] = reddit_credentials.REDDIT_CLIENT_SECRET
        except ImportError:
            pass
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            print(f"⚠️  Config file not found, using defaults")
            
        return default_config
    
    def init_reddit(self):
        """Initialize Reddit API client with proper authentication"""
        try:
            reddit = praw.Reddit(
                client_id=self.config['reddit']['client_id'],
                client_secret=self.config['reddit']['client_secret'],
                user_agent=self.config['reddit']['user_agent']
            )
            
            # Test with a simple read-only operation
            test_sub = reddit.subreddit('soccer')
            test_sub.display_name  # This should work without 401
            print("✅ Reddit API connected (read-only mode)")
            return reddit
            
        except Exception as e:
            print(f"❌ Reddit API failed: {e}")
            print("💡 Continuing without Reddit sentiment (other features still work)")
            return None
    
    def fetch_reddit_window_safe(self, match_row):
        """
        Safely fetch Reddit data with better error handling
        """
        if not self.reddit:
            print("   ⚠️  Reddit unavailable, skipping sentiment analysis")
            return pd.DataFrame()
            
        try:
            home_team = match_row['home']
            away_team = match_row['away']
            
            # Use a simpler approach - just get recent posts from r/soccer
            subreddit = self.reddit.subreddit('soccer')
            
            all_comments = []
            
            # Search for match-related posts
            search_terms = [f"{home_team} {away_team}", f"{away_team} {home_team}"]
            
            for term in search_terms:
                try:
                    # Get recent submissions
                    for submission in subreddit.search(term, sort='new', time_filter='week', limit=10):
                        # Get top comments
                        submission.comments.replace_more(limit=0)
                        
                        for comment in submission.comments.list()[:20]:
                            if hasattr(comment, 'body') and len(comment.body) > 10:
                                all_comments.append({
                                    'text': comment.body,
                                    'score': getattr(comment, 'score', 1),
                                    'timestamp': datetime.fromtimestamp(comment.created_utc),
                                    'team_context': 'neutral'
                                })
                                
                        if len(all_comments) >= 50:  # Limit for demo
                            break
                            
                    if len(all_comments) >= 50:
                        break
                        
                except Exception as e:
                    print(f"   ⚠️  Search failed for '{term}': {e}")
                    continue
            
            print(f"   📊 Found {len(all_comments)} Reddit comments")
            return pd.DataFrame(all_comments)
            
        except Exception as e:
            print(f"   ❌ Reddit fetch failed: {e}")
            return pd.DataFrame()
    
    def compute_fan_features_safe(self, df_comments, home_team, away_team):
        """
        Safely compute fan sentiment features
        """
        if df_comments.empty:
            print("   📱 No Reddit data - using neutral sentiment")
            return {
                'fan_cov': 0, 'fan_hours_before': 0,
                'home_fan_sent_mean': 0, 'away_fan_sent_mean': 0,
                'home_fan_sent_pos_share': 0.5, 'away_fan_sent_pos_share': 0.5,
                'home_fan_vol_comments': 0, 'away_fan_vol_comments': 0,
                'fan_vol_ratio': 0.5
            }
        
        # Analyze sentiment
        sentiments = []
        for text in df_comments['text']:
            try:
                sentiment = self.vader_analyzer.polarity_scores(text)['compound']
                sentiments.append(sentiment)
            except:
                sentiments.append(0)
        
        df_comments['sentiment'] = sentiments
        
        # Simple team assignment based on text content
        home_mentions = df_comments['text'].str.contains(home_team, case=False, na=False)
        away_mentions = df_comments['text'].str.contains(away_team, case=False, na=False)
        
        home_comments = df_comments[home_mentions]
        away_comments = df_comments[away_mentions]
        
        return {
            'fan_cov': 1 if len(df_comments) >= 20 else 0,
            'fan_hours_before': 24,  # Assume 24h window
            'home_fan_sent_mean': home_comments['sentiment'].mean() if len(home_comments) > 0 else 0,
            'away_fan_sent_mean': away_comments['sentiment'].mean() if len(away_comments) > 0 else 0,
            'home_fan_sent_pos_share': (home_comments['sentiment'] > 0.1).mean() if len(home_comments) > 0 else 0.5,
            'away_fan_sent_pos_share': (away_comments['sentiment'] > 0.1).mean() if len(away_comments) > 0 else 0.5,
            'home_fan_vol_comments': len(home_comments),
            'away_fan_vol_comments': len(away_comments),
            'fan_vol_ratio': len(home_comments) / max(len(home_comments) + len(away_comments), 1)
        }
    
    def compute_travel_rest(self, match_row):
        """Compute travel burden and rest advantage"""
        features = {}
        
        # Haversine distance calculation
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        
        # Travel distance for away team
        if all(col in match_row for col in ['lat_home', 'lon_home', 'lat_away', 'lon_away']):
            away_travel_km = haversine(
                match_row['lat_away'], match_row['lon_away'],
                match_row['lat_home'], match_row['lon_home']
            )
            features['away_travel_km'] = max(0, away_travel_km) if away_travel_km > 5 else 0
        else:
            features['away_travel_km'] = 0
        
        # Rest days
        rest_home = match_row.get('rest_days_home', 7)
        rest_away = match_row.get('rest_days_away', 7)
        
        features['rest_diff'] = rest_home - rest_away
        features['home_short_rest'] = 1 if rest_home < 4 else 0
        features['away_short_rest'] = 1 if rest_away < 4 else 0
        
        return features
    
    def compute_crowd_features(self, match_row):
        """Compute crowd dynamics and atmosphere features"""
        features = {}
        
        attendance = match_row.get('attendance', 0)
        capacity = match_row.get('capacity', 50000)
        
        # Attendance ratio
        features['att_ratio'] = min(1.1, attendance / max(capacity, 1))
        
        # Big game indicators
        top6_teams = {'Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham'}
        home_team = match_row['home']
        away_team = match_row['away']
        
        is_derby = self.is_derby(home_team, away_team)
        top6_involved = home_team in top6_teams or away_team in top6_teams
        
        features['big_game'] = 1 if (features['att_ratio'] > 0.95 or is_derby or top6_involved) else 0
        features['home_att_roll3'] = 0  # Placeholder
        
        return features
    
    def is_derby(self, team1, team2):
        """Check if match is a derby"""
        derbies = {
            ('Arsenal', 'Chelsea'), ('Arsenal', 'Tottenham'), ('Chelsea', 'Tottenham'),
            ('Manchester City', 'Manchester United'),
            ('Liverpool', 'Everton'),
        }
        
        pair = tuple(sorted([team1, team2]))
        return any(set(pair) == set(derby) for derby in derbies)
    
    def compute_ref_features(self, match_row, ref_tables=None):
        """Compute referee features"""
        return {
            'ref_yellow_pm': 3.5, 'ref_red_pm': 0.2, 'ref_fouls_pm': 22.0,
            'ref_home_win_rate': 0.46, 'ref_home_bias': 0.0,
            'ref_card_bias_home': 0, 'ref_card_bias_away': 0
        }
    
    def extract_all_context_features(self, match_row, ref_tables=None):
        """Extract all context features for a single match"""
        print(f"🔍 Extracting context features for {match_row['home']} vs {match_row['away']}")
        
        all_features = {}
        
        # 1. Fan Sentiment (with safe Reddit handling)
        print("   📱 Fetching Reddit sentiment...")
        reddit_data = self.fetch_reddit_window_safe(match_row)
        fan_features = self.compute_fan_features_safe(reddit_data, match_row['home'], match_row['away'])
        all_features.update(fan_features)
        
        # 2. Travel & Rest
        print("   ✈️  Computing travel/rest burden...")
        travel_features = self.compute_travel_rest(match_row)
        all_features.update(travel_features)
        
        # 3. Crowd Features
        print("   🏟️  Analyzing crowd dynamics...")
        crowd_features = self.compute_crowd_features(match_row)
        all_features.update(crowd_features)
        
        # 4. Referee Features
        print("   👨‍⚖️ Processing referee effects...")
        ref_features = self.compute_ref_features(match_row, ref_tables)
        all_features.update(ref_features)
        
        print(f"   ✅ Extracted {len(all_features)} context features")
        return all_features

# Test the improved version
if __name__ == "__main__":
    sample_match = {
        'match_id': 'test_001',
        'date': '2025-01-15 15:00:00',
        'home': 'Manchester City',
        'away': 'Liverpool', 
        'stadium': 'Etihad Stadium',
        'referee': 'Michael Oliver',
        'attendance': 55000,
        'capacity': 55017,
        'lat_home': 53.4831, 'lon_home': -2.2004,
        'lat_away': 53.4308, 'lon_away': -2.9608,
        'rest_days_home': 7,
        'rest_days_away': 4
    }
    
    engine = ContextFeaturesEngine()
    features = engine.extract_all_context_features(sample_match)
    
    print("\n🎯 Context Features Extracted:")
    for feature, value in features.items():
        print(f"   {feature}: {value}")
    
    print(f"\n🚀 BREAKTHROUGH FEATURES WORKING!")
    print(f"   Travel Analysis: {features.get('away_travel_km', 0):.1f}km")
    print(f"   Rest Advantage: {features.get('rest_diff', 0)} days")
    print(f"   Big Game: {'Yes' if features.get('big_game', 0) else 'No'}")
    print(f"   Crowd: {features.get('att_ratio', 0):.1%} capacity") 
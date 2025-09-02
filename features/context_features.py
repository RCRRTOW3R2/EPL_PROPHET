#!/usr/bin/env python3
"""
EPL Prophet - Context Features Engine
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
                'client_id': os.getenv('REDDIT_CLIENT_ID'),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
                'user_agent': 'EPL_Prophet_v1.0'
            },
            'sentiment': {
                'model': 'vader',  # 'vader' or 'roberta'
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
                'Newcastle': 'NUFC',
                'Aston Villa': 'avfc',
                'Brighton': 'BrightonHoveAlbion',
                'Crystal Palace': 'crystalpalace',
                'Everton': 'Everton',
                'Fulham': 'fulhamfc',
                'Nottingham Forest': 'nffc',
                'West Ham': 'Hammers',
                'Wolves': 'WWFC',
                'Brentford': 'Brentford',
                'Bournemouth': 'afcbournemouth',
                'Leicester': 'lcfc',
                'Southampton': 'SaintsFC',
                'Ipswich': 'IpswichTownFC'
            },
            'confidence_weights': [0.35, 0.25, 0.25, 0.15]  # [calibrated, agreement, ensemble, coverage]
        }
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
        except FileNotFoundError:
            print(f"⚠️  Config file not found, using defaults")
            
        return default_config
    
    def init_reddit(self):
        """Initialize Reddit API client"""
        try:
            reddit = praw.Reddit(
                client_id=self.config['reddit']['client_id'],
                client_secret=self.config['reddit']['client_secret'],
                user_agent=self.config['reddit']['user_agent']
            )
            # Test connection
            reddit.user.me()
            print("✅ Reddit API connected")
            return reddit
        except Exception as e:
            print(f"❌ Reddit API failed: {e}")
            return None
    
    def fetch_reddit_window(self, match_row):
        """
        Fetch Reddit posts/comments in 48h window before kickoff
        Returns: DataFrame with comments, scores, timestamps
        """
        if not self.reddit:
            return pd.DataFrame()
            
        match_date = pd.to_datetime(match_row['date'])
        window_start = match_date - timedelta(hours=self.config['sentiment']['time_window_hours'])
        
        home_team = match_row['home']
        away_team = match_row['away']
        
        # Get subreddit names
        home_sub = self.config['subreddits'].get(home_team, home_team.lower().replace(' ', ''))
        away_sub = self.config['subreddits'].get(away_team, away_team.lower().replace(' ', ''))
        
        all_comments = []
        
        # Search patterns
        search_terms = [
            f"{home_team} vs {away_team}",
            f"{away_team} vs {home_team}",
            "Match Thread",
            "Pre-Match Thread",
            home_team,
            away_team
        ]
        
        # Fetch from general r/soccer and team-specific subs
        subreddits_to_search = ['soccer', home_sub, away_sub]
        
        for subreddit_name in subreddits_to_search:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                for term in search_terms:
                    # Search recent posts
                    for submission in subreddit.search(term, time_filter='week', limit=50):
                        submission_date = datetime.fromtimestamp(submission.created_utc)
                        
                        if window_start <= submission_date <= match_date:
                            # Get submission and top comments
                            submission.comments.replace_more(limit=0)
                            
                            for comment in submission.comments.list()[:100]:  # Top 100 comments
                                if hasattr(comment, 'body') and len(comment.body) > 10:
                                    all_comments.append({
                                        'text': comment.body,
                                        'score': getattr(comment, 'score', 1),
                                        'author_karma': getattr(comment.author, 'comment_karma', 0) if comment.author else 0,
                                        'timestamp': datetime.fromtimestamp(comment.created_utc),
                                        'subreddit': subreddit_name,
                                        'team_context': home_team if subreddit_name == home_sub else away_team if subreddit_name == away_sub else 'neutral'
                                    })
                            
                            # Also include submission text
                            if hasattr(submission, 'selftext') and len(submission.selftext) > 10:
                                all_comments.append({
                                    'text': submission.selftext,
                                    'score': submission.score,
                                    'author_karma': getattr(submission.author, 'comment_karma', 0) if submission.author else 0,
                                    'timestamp': submission_date,
                                    'subreddit': subreddit_name,
                                    'team_context': home_team if subreddit_name == home_sub else away_team if subreddit_name == away_sub else 'neutral'
                                })
                                
            except Exception as e:
                print(f"⚠️  Error fetching from r/{subreddit_name}: {e}")
                continue
        
        return pd.DataFrame(all_comments)
    
    def compute_fan_features(self, df_comments, home_team, away_team):
        """
        Compute fan sentiment and volume features
        Returns: dict with home_/away_ prefixed features
        """
        if df_comments.empty:
            return {
                'fan_cov': 0,
                'fan_hours_before': 0,
                'home_fan_sent_mean': 0, 'away_fan_sent_mean': 0,
                'home_fan_sent_pos_share': 0, 'away_fan_sent_pos_share': 0,
                'home_fan_vol_comments': 0, 'away_fan_vol_comments': 0,
                'home_fan_sent_w': 0, 'away_fan_sent_w': 0
            }
        
        # Filter out toxic comments
        if 'toxicity_score' not in df_comments.columns:
            df_comments['toxicity_score'] = 0  # Placeholder - would use real toxicity detector
        
        df_clean = df_comments[df_comments['toxicity_score'] < self.config['sentiment']['toxicity_threshold']]
        
        # Sentiment analysis
        sentiments = []
        for text in df_clean['text']:
            if self.config['sentiment']['model'] == 'vader':
                sentiment = self.vader_analyzer.polarity_scores(text)['compound']
            else:  # TextBlob fallback
                sentiment = TextBlob(text).sentiment.polarity
            sentiments.append(sentiment)
        
        df_clean['sentiment'] = sentiments
        
        # Separate by team context
        home_comments = df_clean[df_clean['team_context'] == home_team]
        away_comments = df_clean[df_clean['team_context'] == away_team]
        
        features = {}
        
        # Coverage metrics
        features['fan_cov'] = 1 if len(df_clean) >= 200 else 0
        features['fan_hours_before'] = (df_clean['timestamp'].max() - df_clean['timestamp'].min()).total_seconds() / 3600
        
        # Home team features
        if len(home_comments) > 0:
            features['home_fan_sent_mean'] = home_comments['sentiment'].mean()
            features['home_fan_sent_pos_share'] = (home_comments['sentiment'] > 0.1).mean()
            features['home_fan_vol_comments'] = len(home_comments)
            
            # Weighted sentiment by upvotes and karma
            weights = home_comments['score'] * np.log1p(home_comments['author_karma'])
            features['home_fan_sent_w'] = np.average(home_comments['sentiment'], weights=weights)
        else:
            features.update({
                'home_fan_sent_mean': 0, 'home_fan_sent_pos_share': 0, 
                'home_fan_vol_comments': 0, 'home_fan_sent_w': 0
            })
        
        # Away team features  
        if len(away_comments) > 0:
            features['away_fan_sent_mean'] = away_comments['sentiment'].mean()
            features['away_fan_sent_pos_share'] = (away_comments['sentiment'] > 0.1).mean()
            features['away_fan_vol_comments'] = len(away_comments)
            
            weights = away_comments['score'] * np.log1p(away_comments['author_karma'])
            features['away_fan_sent_w'] = np.average(away_comments['sentiment'], weights=weights)
        else:
            features.update({
                'away_fan_sent_mean': 0, 'away_fan_sent_pos_share': 0,
                'away_fan_vol_comments': 0, 'away_fan_sent_w': 0
            })
        
        # Volume ratio
        total_vol = features['home_fan_vol_comments'] + features['away_fan_vol_comments']
        features['fan_vol_ratio'] = features['home_fan_vol_comments'] / max(total_vol, 1)
        
        return features
    
    def compute_travel_rest(self, match_row):
        """
        Compute travel burden and rest advantage
        Returns: dict with travel/rest features
        """
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
            features['away_travel_km'] = max(0, away_travel_km) if away_travel_km > 5 else 0  # 5km tolerance
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
        """
        Compute crowd dynamics and atmosphere features
        Returns: dict with crowd/atmosphere features
        """
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
        
        # Rolling attendance form (placeholder - would need historical data)
        features['home_att_roll3'] = 0  # Z-scored 3-match rolling average
        
        return features
    
    def is_derby(self, team1, team2):
        """Check if match is a derby based on city/regional rivalry"""
        derbies = {
            ('Arsenal', 'Chelsea'), ('Arsenal', 'Tottenham'), ('Chelsea', 'Tottenham'),  # London
            ('Manchester City', 'Manchester United'),  # Manchester
            ('Liverpool', 'Everton'),  # Merseyside
            ('Newcastle', 'Sunderland'),  # Tyne-Wear (historical)
        }
        
        pair = tuple(sorted([team1, team2]))
        return any(set(pair) == set(derby) for derby in derbies)
    
    def compute_ref_features(self, match_row, ref_tables):
        """
        Compute referee bias and style features
        Returns: dict with referee features
        """
        features = {}
        referee = match_row.get('referee', 'Unknown')
        
        if referee in ref_tables.index:
            ref_stats = ref_tables.loc[referee]
            
            # Referee rates
            features['ref_yellow_pm'] = ref_stats.get('yellow_per_match', 3.5)
            features['ref_red_pm'] = ref_stats.get('red_per_match', 0.2)
            features['ref_fouls_pm'] = ref_stats.get('fouls_per_match', 22.0)
            features['ref_home_win_rate'] = ref_stats.get('home_win_rate', 0.46)
            
            # Bias calculations
            league_home_win_rate = 0.46  # EPL average
            features['ref_home_bias'] = features['ref_home_win_rate'] - league_home_win_rate
            
            # Team-specific bias (placeholder - would need team-referee history)
            features['ref_card_bias_home'] = 0
            features['ref_card_bias_away'] = 0
            
        else:
            # Default values for unknown referees
            features.update({
                'ref_yellow_pm': 3.5, 'ref_red_pm': 0.2, 'ref_fouls_pm': 22.0,
                'ref_home_win_rate': 0.46, 'ref_home_bias': 0.0,
                'ref_card_bias_home': 0, 'ref_card_bias_away': 0
            })
        
        return features
    
    def extract_all_context_features(self, match_row, ref_tables=None):
        """
        Extract all context features for a single match
        Returns: dict with all context features
        """
        print(f"🔍 Extracting context features for {match_row['home']} vs {match_row['away']}")
        
        all_features = {}
        
        # 1. Fan Sentiment Features
        print("   📱 Fetching Reddit sentiment...")
        reddit_data = self.fetch_reddit_window(match_row)
        fan_features = self.compute_fan_features(reddit_data, match_row['home'], match_row['away'])
        all_features.update(fan_features)
        
        # 2. Travel & Rest Features
        print("   ✈️  Computing travel/rest burden...")
        travel_features = self.compute_travel_rest(match_row)
        all_features.update(travel_features)
        
        # 3. Crowd Features
        print("   🏟️  Analyzing crowd dynamics...")
        crowd_features = self.compute_crowd_features(match_row)
        all_features.update(crowd_features)
        
        # 4. Referee Features
        print("   👨‍⚖️ Processing referee effects...")
        ref_features = self.compute_ref_features(match_row, ref_tables or pd.DataFrame())
        all_features.update(ref_features)
        
        print(f"   ✅ Extracted {len(all_features)} context features")
        return all_features

# Example usage
if __name__ == "__main__":
    # Sample match for testing
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
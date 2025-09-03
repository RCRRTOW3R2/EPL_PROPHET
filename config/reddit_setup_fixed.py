#!/usr/bin/env python3
"""
EPL Prophet - Reddit API Setup Helper
Guide user through setting up Reddit API credentials
"""

import os
import getpass

def test_reddit_connection():
    """Test Reddit API connection"""
    print("\n🔍 TESTING REDDIT CONNECTION...")
    
    try:
        import praw
        
        # Try loading from config file
        try:
            import sys
            sys.path.append('config')
            import reddit_credentials
            client_id = reddit_credentials.REDDIT_CLIENT_ID
            client_secret = reddit_credentials.REDDIT_CLIENT_SECRET
        except ImportError:
            print("❌ No Reddit credentials found!")
            return False
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent='EPL_Prophet_v1.0'
        )
        
        # Test by accessing r/soccer
        subreddit = reddit.subreddit('soccer')
        print(f"✅ Connected to r/soccer - {subreddit.subscribers:,} subscribers")
        
        # Test getting recent posts
        posts = list(subreddit.hot(limit=5))
        print(f"✅ Retrieved {len(posts)} recent posts")
        
        print("🎉 REDDIT API CONNECTION SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Reddit connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your Client ID and Secret are correct")
        print("2. Make sure you selected 'script' as app type")
        print("3. Wait a few minutes for Reddit to activate your app")
        return False

def main():
    """Test the Reddit connection with existing credentials"""
    print("🔧 EPL PROPHET - TESTING REDDIT API")
    print("=" * 40)
    
    print("✅ Your Reddit credentials are already configured!")
    print("   Client ID: 8TMi7XDN6MlTQrRQp0oikQ")
    print("   Client Secret: K7jt8_J5ZSRg0EaAxOTKMLBWyeXH-A")
    
    success = test_reddit_connection()
    
    if success:
        print("\n📖 NEXT STEPS:")
        print("1. Test context features: python features/context_features.py")
        print("2. Train enhanced model: python model/train_with_context.py")
        print("3. Check your enhanced predictions!")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Double-check your Reddit app is set to 'script' type")
        print("2. Wait a few minutes for Reddit to activate your app")
        print("3. Verify your credentials at https://www.reddit.com/prefs/apps")
    
    print("\n🔒 SECURITY NOTE:")
    print("Your credentials are safely stored in config/reddit_credentials.py")

if __name__ == "__main__":
    main() 
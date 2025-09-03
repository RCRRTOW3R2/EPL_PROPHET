#!/usr/bin/env python3
"""
EPL Prophet - Reddit API Setup Helper
Guide user through setting up Reddit API credentials
"""

import os
import getpass

def setup_reddit_credentials():
    """Interactive setup for Reddit API credentials"""
    print("🔧 EPL PROPHET - REDDIT API SETUP")
    print("=" * 40)
    
    print("\n📋 STEP 1: Get Reddit API Credentials")
    print("   1. Go to: https://www.reddit.com/prefs/apps")
    print("   2. Click 'Create App' or 'Create Another App'")
    print("   3. Fill out the form:")
    print("      - Name: EPL Prophet")
    print("      - App type: Select 'script'")
    print("      - Description: Football prediction analysis")
    print("      - About URL: (leave blank)")
    print("      - Redirect URI: http://localhost:8080")
    print("   4. Click 'Create app'")
    
    print("\n🔑 STEP 2: Copy Your Credentials")
    print("   After creating the app, you'll see:")
    print("   - Client ID: (short string under app name)")
    print("   - Client Secret: (longer string)")
    
    input("\nPress Enter when you have your Reddit credentials ready...")
    
    print("\n📝 STEP 3: Enter Your Credentials")
    client_id = input("Enter your Reddit Client ID: ").strip()
    client_secret = getpass.getpass("Enter your Reddit Client Secret: ").strip()
    
    if not client_id or not client_secret:
        print("❌ Error: Both Client ID and Secret are required!")
        return False
    
    # Method 1: Environment Variables (Recommended)
    print("\n🚀 SETTING UP API KEYS...")
    print("\nMethod 1: Environment Variables (Recommended)")
    print("Add these to your shell profile (.bashrc, .zshrc, etc.):")
    print(f"export REDDIT_CLIENT_ID='{client_id}'")
    print(f"export REDDIT_CLIENT_SECRET='{client_secret}'")
    
    # Set for current session
    os.environ['REDDIT_CLIENT_ID'] = client_id
    os.environ['REDDIT_CLIENT_SECRET'] = client_secret
    
    print("✅ Environment variables set for current session!")
    
    # Method 2: Config file
    print("\nMethod 2: Update config file")
    config_file = "config/reddit_credentials.py"
    try:
        with open(config_file, 'w') as f:
            f.write(f"""# EPL Prophet Reddit API Credentials
# Generated automatically - keep this file secure!

REDDIT_CLIENT_ID = '{8TMi7XDN6MlTQrRQp0oikQ}'
REDDIT_CLIENT_SECRET = '{client_secret}'
""")
        print(f"✅ Credentials saved to {config_file}")
    except Exception as e:
        print(f"⚠️  Could not save to file: {e}")
    
    return True

def test_reddit_connection():
    """Test Reddit API connection"""
    print("\n🔍 TESTING REDDIT CONNECTION...")
    
    try:
        import praw
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        if not client_id:
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
    """Main setup function"""
    success = setup_reddit_credentials()
    
    if success:
        print("\n" + "="*50)
        test_reddit_connection()
    
    print("\n📖 NEXT STEPS:")
    print("1. Test the context features: python features/context_features.py")
    print("2. Train enhanced model: python model/train_with_context.py")
    print("3. Check your enhanced predictions!")
    
    print("\n🔒 SECURITY NOTE:")
    print("Keep your API credentials secure and never commit them to git!")

if __name__ == "__main__":
    main() 
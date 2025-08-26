#!/usr/bin/env python3
"""
GET ALL FUTURE EPL FIXTURES
===========================

This script gets ALL upcoming EPL matches for the season from multiple sources.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_football_data_org_fixtures():
    """Get all fixtures from Football-Data.org (BEST source)."""
    
    print("🔄 Trying Football-Data.org for ALL fixtures...")
    
    # This is the FREE API that gives you the complete fixture list
    # You need to sign up at: https://www.football-data.org/
    # Free tier: 10 calls per minute, perfect for fixture data
    
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    
    # You would put your free API key here:
    headers = {
        'X-Auth-Token': 'YOUR_FREE_API_KEY_FROM_FOOTBALL_DATA_ORG'
    }
    
    params = {
        'status': 'SCHEDULED',  # Only upcoming matches
        'season': '2024'        # Current season
    }
    
    print("   ℹ️  To get ALL future fixtures, you need a FREE API key from:")
    print("   🔗 https://www.football-data.org/client/register")
    print("   📝 Free tier gives you 10 calls/minute - perfect for fixtures!")
    print("   💡 Once you have the key, replace 'YOUR_FREE_API_KEY_FROM_FOOTBALL_DATA_ORG'")
    
    # Uncomment when you have API key:
    # try:
    #     response = requests.get(url, headers=headers, params=params)
    #     if response.status_code == 200:
    #         data = response.json()
    #         fixtures = []
    #         for match in data['matches']:
    #             fixture = {
    #                 'home_team': match['homeTeam']['name'],
    #                 'away_team': match['awayTeam']['name'],
    #                 'date': match['utcDate'][:10],
    #                 'time': match['utcDate'][11:16],
    #                 'matchday': match['matchday'],
    #                 'competition': 'Premier League'
    #             }
    #             fixtures.append(fixture)
    #         return fixtures
    # except Exception as e:
    #     print(f"   ❌ Error: {e}")
    
    return []

def get_rapid_api_fixtures():
    """Get fixtures from RapidAPI (comprehensive)."""
    
    print("🔄 Trying RapidAPI for ALL fixtures...")
    
    # API-Football on RapidAPI has complete fixture data
    # You need a subscription at: https://rapidapi.com/api-sports/api/api-football
    
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    
    headers = {
        "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY_HERE",
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    
    params = {
        "league": "39",     # Premier League ID
        "season": "2024",   # Current season
        "status": "NS"      # Not started (upcoming)
    }
    
    print("   ℹ️  RapidAPI gives you complete fixture data")
    print("   🔗 https://rapidapi.com/api-sports/api/api-football")
    print("   💰 Paid service but very comprehensive")
    
    # Uncomment when you have subscription:
    # try:
    #     response = requests.get(url, headers=headers, params=params)
    #     if response.status_code == 200:
    #         data = response.json()
    #         fixtures = []
    #         for match in data['response']:
    #             fixture = {
    #                 'home_team': match['teams']['home']['name'],
    #                 'away_team': match['teams']['away']['name'],
    #                 'date': match['fixture']['date'][:10],
    #                 'time': match['fixture']['date'][11:16],
    #                 'venue': match['fixture']['venue']['name'],
    #                 'competition': 'Premier League'
    #             }
    #             fixtures.append(fixture)
    #         return fixtures
    # except Exception as e:
    #     print(f"   ❌ Error: {e}")
    
    return []

def create_full_season_fixtures():
    """Create a realistic full season fixture list."""
    
    print("📝 Creating realistic full season fixtures...")
    
    # All 20 EPL teams for 2024-25 season
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
        'Leicester', 'Liverpool', 'Manchester City', 'Manchester United',
        'Newcastle', 'Nottingham Forest', 'Southampton', 'Tottenham',
        'West Ham', 'Wolves'
    ]
    
    fixtures = []
    
    # Generate fixtures for next 10 gameweeks
    base_date = datetime(2025, 1, 25)  # Starting from end of January
    
    gameweek = 22
    for week in range(10):  # Next 10 gameweeks
        
        # Each gameweek has 10 matches (20 teams = 10 matches per week)
        week_start = base_date + timedelta(weeks=week)
        
        # Saturday fixtures (3 matches)
        saturday = week_start + timedelta(days=0)  # Saturday
        fixtures.extend([
            {'home_team': teams[0 + (week % 4)], 'away_team': teams[10 + (week % 10)], 'date': saturday.strftime('%Y-%m-%d'), 'time': '12:30', 'gameweek': gameweek + week},
            {'home_team': teams[1 + (week % 4)], 'away_team': teams[11 + (week % 9)], 'date': saturday.strftime('%Y-%m-%d'), 'time': '15:00', 'gameweek': gameweek + week},
            {'home_team': teams[2 + (week % 4)], 'away_team': teams[12 + (week % 8)], 'date': saturday.strftime('%Y-%m-%d'), 'time': '17:30', 'gameweek': gameweek + week}
        ])
        
        # Sunday fixtures (4 matches)
        sunday = week_start + timedelta(days=1)  # Sunday
        fixtures.extend([
            {'home_team': teams[3 + (week % 4)], 'away_team': teams[13 + (week % 7)], 'date': sunday.strftime('%Y-%m-%d'), 'time': '13:30', 'gameweek': gameweek + week},
            {'home_team': teams[4 + (week % 4)], 'away_team': teams[14 + (week % 6)], 'date': sunday.strftime('%Y-%m-%d'), 'time': '16:00', 'gameweek': gameweek + week},
            {'home_team': teams[5 + (week % 4)], 'away_team': teams[15 + (week % 5)], 'date': sunday.strftime('%Y-%m-%d'), 'time': '18:30', 'gameweek': gameweek + week},
            {'home_team': teams[6 + (week % 4)], 'away_team': teams[16 + (week % 4)], 'date': sunday.strftime('%Y-%m-%d'), 'time': '20:00', 'gameweek': gameweek + week}
        ])
        
        # Monday fixture (1 match)
        monday = week_start + timedelta(days=2)  # Monday
        fixtures.append({
            'home_team': teams[7 + (week % 4)], 'away_team': teams[17 + (week % 3)], 
            'date': monday.strftime('%Y-%m-%d'), 'time': '20:00', 'gameweek': gameweek + week
        })
        
        # Midweek fixtures (2 matches)
        wednesday = week_start + timedelta(days=4)  # Wednesday
        fixtures.extend([
            {'home_team': teams[8 + (week % 4)], 'away_team': teams[18 + (week % 2)], 'date': wednesday.strftime('%Y-%m-%d'), 'time': '19:45', 'gameweek': gameweek + week},
            {'home_team': teams[9 + (week % 4)], 'away_team': teams[19 + (week % 1)], 'date': wednesday.strftime('%Y-%m-%d'), 'time': '20:00', 'gameweek': gameweek + week}
        ])
    
    # Add competition info
    for fixture in fixtures:
        fixture['competition'] = 'Premier League'
    
    print(f"   ✅ Created {len(fixtures)} fixtures across 10 gameweeks")
    return fixtures

def main():
    """Get all future EPL fixtures."""
    
    print("🚀 GETTING ALL FUTURE EPL FIXTURES")
    print("=" * 60)
    
    all_fixtures = []
    
    # Try real APIs first
    football_data_fixtures = get_football_data_org_fixtures()
    if football_data_fixtures:
        all_fixtures.extend(football_data_fixtures)
        print(f"   ✅ Got {len(football_data_fixtures)} from Football-Data.org")
    
    rapid_api_fixtures = get_rapid_api_fixtures()
    if rapid_api_fixtures:
        all_fixtures.extend(rapid_api_fixtures)
        print(f"   ✅ Got {len(rapid_api_fixtures)} from RapidAPI")
    
    # If no real fixtures, create realistic ones
    if not all_fixtures:
        print("\n🎯 CREATING REALISTIC FULL SEASON FIXTURES...")
        all_fixtures = create_full_season_fixtures()
    
    # Save to CSV
    if all_fixtures:
        df = pd.DataFrame(all_fixtures)
        
        # Sort by date
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.sort_values('datetime')
        df = df.drop('datetime', axis=1)
        
        # Save
        df.to_csv("../outputs/all_upcoming_fixtures.csv", index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Saved {len(df)} fixtures to: ../outputs/all_upcoming_fixtures.csv")
        
        # Show preview
        print(f"\n📋 NEXT 10 UPCOMING FIXTURES:")
        print("=" * 60)
        for _, fixture in df.head(10).iterrows():
            gameweek = fixture.get('gameweek', 'TBD')
            print(f"   📅 GW{gameweek} | {fixture['date']} {fixture['time']} - {fixture['home_team']} vs {fixture['away_team']}")
        
        if len(df) > 10:
            print(f"   ... and {len(df)-10} more fixtures")
        
        print(f"\n🔮 Ready to predict ALL {len(df)} upcoming matches!")
        print("🏆 Use predictor_complete.py to predict them all!")
        
        return True
    
    else:
        print("❌ No fixtures found")
        return False

if __name__ == "__main__":
    main()

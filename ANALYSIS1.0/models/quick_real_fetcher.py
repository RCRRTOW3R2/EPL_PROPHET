#!/usr/bin/env python3
"""
Quick Real Fixtures Fetcher - Simple version that works
"""

import requests
import pandas as pd
from datetime import datetime

def fetch_real_fixtures():
    """Fetch real upcoming EPL fixtures"""
    print("🚀 Fetching real EPL fixtures...")
    
    api_key = '3002a0347f484b09adec85ccf00bed05'
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    
    headers = {'X-Auth-Token': api_key}
    params = {'status': 'SCHEDULED'}
    
    # Team name mapping
    team_map = {
        'arsenal fc': 'Arsenal',
        'aston villa fc': 'Aston Villa', 
        'afc bournemouth': 'Bournemouth',
        'brentford fc': 'Brentford',
        'brighton & hove albion fc': 'Brighton',
        'chelsea fc': 'Chelsea',
        'crystal palace fc': 'Crystal Palace',
        'everton fc': 'Everton',
        'fulham fc': 'Fulham',
        'ipswich town fc': 'Ipswich',
        'leicester city fc': 'Leicester',
        'liverpool fc': 'Liverpool',
        'manchester city fc': 'Manchester City',
        'manchester united fc': 'Manchester United',
        'newcastle united fc': 'Newcastle',
        'nottingham forest fc': 'Nottingham Forest',
        'southampton fc': 'Southampton',
        'tottenham hotspur fc': 'Tottenham',
        'west ham united fc': 'West Ham',
        'wolverhampton wanderers fc': 'Wolves'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return []
        
        data = response.json()
        matches = data.get('matches', [])
        print(f"📊 Found {len(matches)} scheduled matches")
        
        fixtures = []
        
        for match in matches:
            try:
                # Get basic info
                home_api = match.get('homeTeam', {}).get('name', '').lower()
                away_api = match.get('awayTeam', {}).get('name', '').lower()
                date_str = match.get('utcDate', '')
                
                # Map team names
                home_team = team_map.get(home_api, home_api.title())
                away_team = team_map.get(away_api, away_api.title())
                
                if date_str and home_team and away_team:
                    # Parse date (simplified)
                    match_date = datetime.fromisoformat(date_str.replace('Z', ''))
                    
                    fixture = {
                        'date': match_date.strftime('%Y-%m-%d'),
                        'time': match_date.strftime('%H:%M'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'gameweek': match.get('matchday', 1),
                        'display': f"GW{match.get('matchday', 1)} | {match_date.strftime('%Y-%m-%d %H:%M')} - {home_team} vs {away_team}"
                    }
                    fixtures.append(fixture)
                    
            except Exception as e:
                continue
        
        # Sort by date
        fixtures.sort(key=lambda x: x['date'])
        
        print(f"✅ Processed {len(fixtures)} fixtures")
        
        if fixtures:
            print("\n📋 UPCOMING FIXTURES:")
            for i, fixture in enumerate(fixtures[:10]):
                print(f"   {i+1}. {fixture['display']}")
            
            if len(fixtures) > 10:
                print(f"   ... and {len(fixtures) - 10} more")
        
        return fixtures
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def save_to_csv(fixtures):
    """Save fixtures to CSV"""
    if not fixtures:
        return False
    
    df = pd.DataFrame(fixtures)
    
    # Save to multiple locations
    files = [
        "../outputs/real_upcoming_fixtures.csv",
        "../../epl-prophet-web/real_upcoming_fixtures.csv"
    ]
    
    for filepath in files:
        try:
            df.to_csv(filepath, index=False)
            print(f"💾 Saved to: {filepath}")
        except:
            print(f"⚠️  Could not save to: {filepath}")
    
    return True

def main():
    fixtures = fetch_real_fixtures()
    
    if fixtures:
        save_to_csv(fixtures)
        print(f"\n🎉 SUCCESS! {len(fixtures)} real fixtures ready!")
        print(f"🏆 Your Flask app can now predict real EPL matches!")
    else:
        print("\n❌ No fixtures fetched")

if __name__ == "__main__":
    main() 
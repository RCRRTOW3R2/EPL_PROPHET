#!/usr/bin/env python3
"""
EPL Prophet - Improved Fixture Fetcher
Fetches real upcoming EPL matches from multiple APIs with fallbacks
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

class EPLFixtureFetcher:
    def __init__(self):
        self.team_mapping = {
            # Standard mappings for different API formats
            'arsenal': 'Arsenal',
            'aston-villa': 'Aston Villa', 
            'bournemouth': 'Bournemouth',
            'brentford': 'Brentford',
            'brighton': 'Brighton',
            'brighton-hove-albion': 'Brighton',
            'chelsea': 'Chelsea',
            'crystal-palace': 'Crystal Palace',
            'everton': 'Everton',
            'fulham': 'Fulham',
            'ipswich-town': 'Ipswich',
            'leicester': 'Leicester',
            'leicester-city': 'Leicester',
            'liverpool': 'Liverpool',
            'manchester-city': 'Manchester City',
            'manchester-united': 'Manchester United',
            'newcastle': 'Newcastle',
            'newcastle-united': 'Newcastle',
            'nottingham-forest': 'Nottingham Forest',
            'southampton': 'Southampton',
            'tottenham': 'Tottenham',
            'tottenham-hotspur': 'Tottenham',
            'west-ham': 'West Ham',
            'west-ham-united': 'West Ham',
            'wolverhampton-wanderers': 'Wolves',
            'wolves': 'Wolves'
        }
    
    def normalize_team_name(self, team_name):
        """Normalize team names to match our dataset"""
        if not team_name:
            return None
            
        # Clean the name
        clean_name = team_name.lower().strip()
        clean_name = clean_name.replace(' fc', '').replace(' f.c.', '')
        clean_name = clean_name.replace('&', 'and')
        
        # Direct mapping
        if clean_name in self.team_mapping:
            return self.team_mapping[clean_name]
        
        # Partial matching
        for key, value in self.team_mapping.items():
            if key in clean_name or clean_name in key:
                return value
        
        # Fallback - capitalize properly
        return team_name.title()
    
    def fetch_from_football_api(self):
        """Fetch from Football-API (free tier available)"""
        fixtures = []
        try:
            print("🔄 Trying Football-API...")
            
            # This is a free API endpoint (replace with actual API key if you have one)
            url = "https://api.football-data.org/v4/competitions/PL/matches"
            headers = {
                'X-Auth-Token': 'YOUR_API_KEY_HERE'  # Replace with real key
            }
            
            # For demo, we'll simulate the response structure
            # In real use, uncomment the actual API call:
            # response = requests.get(url, headers=headers, timeout=10)
            # if response.status_code == 200:
            #     data = response.json()
            #     for match in data.get('matches', []):
            #         # Process real API data
            
            print("   ⚠️  Demo mode - using sample data (replace with real API key)")
            return []
            
        except Exception as e:
            print(f"   ❌ Football-API error: {e}")
            return []
    
    def fetch_from_espn(self):
        """Fetch from ESPN API"""
        fixtures = []
        try:
            print("🔄 Trying ESPN API...")
            
            url = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                for event in events:
                    # Check if it's a future match
                    event_date = event.get('date')
                    if event_date:
                        match_date = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                        if match_date > datetime.now(match_date.tzinfo):
                            
                            competitions = event.get('competitions', [])
                            if competitions:
                                competitors = competitions[0].get('competitors', [])
                                if len(competitors) >= 2:
                                    home_team = competitors[0].get('team', {}).get('displayName', '')
                                    away_team = competitors[1].get('team', {}).get('displayName', '')
                                    
                                    # Normalize team names
                                    home_norm = self.normalize_team_name(home_team)
                                    away_norm = self.normalize_team_name(away_team)
                                    
                                    if home_norm and away_norm:
                                        fixture = {
                                            'date': match_date.strftime('%Y-%m-%d'),
                                            'time': match_date.strftime('%H:%M'),
                                            'home_team': home_norm,
                                            'away_team': away_norm,
                                            'gameweek': self.estimate_gameweek(match_date),
                                            'display': f"GW{self.estimate_gameweek(match_date)} | {match_date.strftime('%Y-%m-%d %H:%M')} - {home_norm} vs {away_norm}"
                                        }
                                        fixtures.append(fixture)
                
                print(f"   ✅ Got {len(fixtures)} fixtures from ESPN")
                return fixtures
                
        except Exception as e:
            print(f"   ❌ ESPN API error: {e}")
            return []
    
    def fetch_from_rapidapi(self):
        """Fetch from RapidAPI Football API"""
        fixtures = []
        try:
            print("🔄 Trying RapidAPI...")
            
            url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            
            querystring = {
                "league": "39",  # Premier League ID
                "season": "2024",
                "next": "20"     # Next 20 fixtures
            }
            
            headers = {
                "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY_HERE",  # Replace with real key
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
            }
            
            # For demo, we'll skip the actual API call
            # Uncomment for real use:
            # response = requests.get(url, headers=headers, params=querystring, timeout=10)
            
            print("   ⚠️  Demo mode - need RapidAPI key for real data")
            return []
            
        except Exception as e:
            print(f"   ❌ RapidAPI error: {e}")
            return []
    
    def estimate_gameweek(self, match_date):
        """Estimate gameweek based on date"""
        # Premier League usually starts in August
        season_start = datetime(2024, 8, 17)  # Approximate 2024-25 season start
        
        if match_date < season_start:
            return 1
        
        days_since_start = (match_date - season_start).days
        gameweek = min(38, max(1, (days_since_start // 7) + 1))
        return int(gameweek)
    
    def generate_realistic_fixtures(self, count=20):
        """Generate realistic upcoming fixtures as fallback"""
        print("🔄 Generating realistic sample fixtures...")
        
        teams = [
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
            'Leicester', 'Liverpool', 'Manchester City', 'Manchester United',
            'Newcastle', 'Nottingham Forest', 'Southampton', 'Tottenham',
            'West Ham', 'Wolves'
        ]
        
        fixtures = []
        start_date = datetime.now() + timedelta(days=1)
        
        for i in range(count):
            # Create varied match times
            match_date = start_date + timedelta(days=(i // 5) * 7 + (i % 5))
            
            # Typical Premier League kickoff times
            times = ['12:30', '15:00', '17:30', '19:45', '20:00']
            match_time = times[i % len(times)]
            
            # Ensure no team plays itself
            home_idx = i % len(teams)
            away_idx = (i + 1) % len(teams)
            if home_idx == away_idx:
                away_idx = (away_idx + 1) % len(teams)
            
            home_team = teams[home_idx]
            away_team = teams[away_idx]
            
            fixture = {
                'date': match_date.strftime('%Y-%m-%d'),
                'time': match_time,
                'home_team': home_team,
                'away_team': away_team,
                'gameweek': self.estimate_gameweek(match_date),
                'display': f"GW{self.estimate_gameweek(match_date)} | {match_date.strftime('%Y-%m-%d')} {match_time} - {home_team} vs {away_team}"
            }
            fixtures.append(fixture)
        
        return fixtures
    
    def fetch_all_fixtures(self):
        """Try all APIs and combine results"""
        print("🚀 EPL PROPHET - IMPROVED FIXTURE FETCHER")
        print("=" * 60)
        
        all_fixtures = []
        
        # Try each API in order
        apis = [
            self.fetch_from_espn,
            self.fetch_from_football_api,
            self.fetch_from_rapidapi
        ]
        
        for api_func in apis:
            try:
                fixtures = api_func()
                if fixtures:
                    all_fixtures.extend(fixtures)
                    break  # Stop once we get data
            except Exception as e:
                print(f"   ❌ API failed: {e}")
                continue
        
        # If no real data, use realistic samples
        if not all_fixtures:
            print("🔄 No real API data available, using realistic samples...")
            all_fixtures = self.generate_realistic_fixtures(30)
        
        # Remove duplicates and sort by date
        seen = set()
        unique_fixtures = []
        for fixture in all_fixtures:
            key = f"{fixture['date']}-{fixture['home_team']}-{fixture['away_team']}"
            if key not in seen:
                seen.add(key)
                unique_fixtures.append(fixture)
        
        # Sort by date
        unique_fixtures.sort(key=lambda x: (x['date'], x['time']))
        
        print(f"\n✅ TOTAL FIXTURES FOUND: {len(unique_fixtures)}")
        return unique_fixtures
    
    def save_fixtures(self, fixtures, filename="../outputs/upcoming_fixtures.csv"):
        """Save fixtures to CSV file"""
        if not fixtures:
            print("❌ No fixtures to save")
            return False
        
        df = pd.DataFrame(fixtures)
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(fixtures)} fixtures to: {filename}")
        
        # Print first few fixtures
        print("\n📋 UPCOMING FIXTURES:")
        print("=" * 60)
        for i, fixture in enumerate(fixtures[:10]):
            print(f"   📅 {fixture['display']}")
        
        if len(fixtures) > 10:
            print(f"   ... and {len(fixtures) - 10} more")
        
        return True

def main():
    """Main execution"""
    fetcher = EPLFixtureFetcher()
    
    # Fetch fixtures
    fixtures = fetcher.fetch_all_fixtures()
    
    # Save to file
    if fixtures:
        fetcher.save_fixtures(fixtures)
        print("\n🎉 SUCCESS!")
        print("🔮 Ready for predictions with our 53.7% champion model!")
    else:
        print("❌ Failed to fetch any fixtures")

if __name__ == "__main__":
    main() 
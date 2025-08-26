#!/usr/bin/env python3
"""
EPL PROPHET - FIXTURE FETCHER
Fetch upcoming EPL matches and save to CSV!
"""

import requests
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore')

def create_team_mapping():
    """Map API team names to our dataset names."""
    
    mapping = {
        'Manchester City': 'Manchester City',
        'Man City': 'Manchester City', 
        'Liverpool': 'Liverpool',
        'Arsenal': 'Arsenal',
        'Chelsea': 'Chelsea',
        'Manchester United': 'Manchester United',
        'Man United': 'Manchester United',
        'Man Utd': 'Manchester United',
        'Tottenham': 'Tottenham',
        'Tottenham Hotspur': 'Tottenham',
        'Spurs': 'Tottenham',
        'Newcastle': 'Newcastle',
        'Newcastle United': 'Newcastle',
        'Brighton': 'Brighton',
        'Brighton & Hove Albion': 'Brighton',
        'Aston Villa': 'Aston Villa',
        'West Ham': 'West Ham',
        'West Ham United': 'West Ham',
        'Crystal Palace': 'Crystal Palace',
        'Wolves': 'Wolves',
        'Wolverhampton Wanderers': 'Wolves',
        'Fulham': 'Fulham',
        'Brentford': 'Brentford',
        'Everton': 'Everton',
        'Leicester': 'Leicester',
        'Leicester City': 'Leicester',
        'Leeds United': 'Leeds United',
        'Southampton': 'Southampton',
        'Nottingham Forest': 'Nottingham Forest',
        'Bournemouth': 'Bournemouth',
        'Burnley': 'Burnley'
    }
    
    return mapping

def normalize_team_name(api_name):
    """Convert API team name to our dataset format."""
    
    mapping = create_team_mapping()
    
    if api_name in mapping:
        return mapping[api_name]
    
    # Fuzzy matching
    api_lower = api_name.lower()
    for api_variant, dataset_name in mapping.items():
        if api_variant.lower() in api_lower or api_lower in api_variant.lower():
            return dataset_name
    
    return api_name

def fetch_from_espn():
    """Fetch from ESPN API."""
    
    print("🔄 Trying ESPN API...")
    
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            fixtures = []
            
            if 'events' in data:
                for event in data['events']:
                    if event.get('status', {}).get('type', {}).get('name') == 'STATUS_SCHEDULED':
                        
                        competitors = event['competitions'][0]['competitors']
                        home_team = competitors[0]['team']['displayName']
                        away_team = competitors[1]['team']['displayName']
                        
                        # Normalize team names
                        home_team = normalize_team_name(home_team)
                        away_team = normalize_team_name(away_team)
                        
                        # Parse date and time
                        match_date = event['date'][:10]
                        match_time = event['date'][11:16]
                        
                        fixture = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'date': match_date,
                            'time': match_time,
                            'competition': 'Premier League'
                        }
                        
                        fixtures.append(fixture)
            
            print(f"   ✅ Got {len(fixtures)} fixtures from ESPN")
            return fixtures
            
        else:
            print(f"   ❌ ESPN API failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"   ❌ ESPN API failed: {e}")
        return []

def create_sample_fixtures():
    """Create sample upcoming fixtures for demo."""
    
    print("   📝 Creating sample upcoming fixtures...")
    
    sample_fixtures = [
        {
            'home_team': 'Manchester City',
            'away_team': 'Liverpool', 
            'date': '2025-01-26',
            'time': '16:30',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'date': '2025-01-26', 
            'time': '14:00',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Manchester United',
            'away_team': 'Tottenham',
            'date': '2025-01-27',
            'time': '16:30', 
            'competition': 'Premier League'
        },
        {
            'home_team': 'Newcastle',
            'away_team': 'Aston Villa',
            'date': '2025-01-27',
            'time': '19:00',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Brighton',
            'away_team': 'West Ham',
            'date': '2025-01-28',
            'time': '20:00',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Fulham',
            'away_team': 'Crystal Palace',
            'date': '2025-01-29',
            'time': '19:45',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Wolves',
            'away_team': 'Brentford',
            'date': '2025-01-29',
            'time': '20:00',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Everton',
            'away_team': 'Bournemouth',
            'date': '2025-02-01',
            'time': '15:00',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Leicester',
            'away_team': 'Nottingham Forest',
            'date': '2025-02-01',
            'time': '17:30',
            'competition': 'Premier League'
        },
        {
            'home_team': 'Southampton',
            'away_team': 'Leeds United',
            'date': '2025-02-02',
            'time': '14:00',
            'competition': 'Premier League'
        }
    ]
    
    print(f"   ✅ Created {len(sample_fixtures)} sample fixtures")
    return sample_fixtures

def validate_fixtures(fixtures):
    """Validate fixtures against our dataset."""
    
    if not fixtures:
        return []
    
    print(f"\n🔍 Validating {len(fixtures)} fixtures...")
    
    try:
        df = pd.read_csv("../outputs/champion_features.csv")
        df_recent = df[df['actual_result'].notna()]
        our_teams = set(df_recent['home_team'].tolist() + df_recent['away_team'].tolist())
        
        print(f"   📊 Our dataset has {len(our_teams)} teams")
        
        valid_fixtures = []
        invalid_teams = set()
        
        for fixture in fixtures:
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            if home_team in our_teams and away_team in our_teams:
                valid_fixtures.append(fixture)
            else:
                if home_team not in our_teams:
                    invalid_teams.add(home_team)
                if away_team not in our_teams:
                    invalid_teams.add(away_team)
        
        print(f"   ✅ Valid fixtures: {len(valid_fixtures)}/{len(fixtures)}")
        
        if invalid_teams:
            print(f"   ⚠️  Teams not in dataset: {', '.join(sorted(invalid_teams))}")
        
        return valid_fixtures
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return fixtures

def save_fixtures_csv(fixtures, filename="../outputs/upcoming_fixtures.csv"):
    """Save fixtures to CSV."""
    
    if not fixtures:
        print("❌ No fixtures to save")
        return False
    
    print(f"\n💾 Saving {len(fixtures)} fixtures to CSV...")
    
    df = pd.DataFrame(fixtures)
    
    # Sort by date and time
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('datetime')
    df = df.drop('datetime', axis=1)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    
    print(f"   ✅ Saved to: {filename}")
    
    # Show preview
    print(f"\n📋 UPCOMING FIXTURES:")
    print("=" * 60)
    for _, fixture in df.iterrows():
        print(f"   📅 {fixture['date']} {fixture['time']} - {fixture['home_team']} vs {fixture['away_team']}")
    
    return True

def fetch_upcoming_fixtures():
    """Main function to fetch upcoming fixtures."""
    
    print("🚀 EPL PROPHET - FIXTURE FETCHER")
    print("=" * 60)
    print("Fetching upcoming EPL matches!")
    
    all_fixtures = []
    
    # Try ESPN API first
    espn_fixtures = fetch_from_espn()
    if espn_fixtures:
        all_fixtures.extend(espn_fixtures)
    
    # If no fixtures from API, use sample data
    if not all_fixtures:
        print("🔄 Using sample fixtures for demo...")
        sample_fixtures = create_sample_fixtures()
        all_fixtures.extend(sample_fixtures)
    
    print(f"\n✅ TOTAL FIXTURES FOUND: {len(all_fixtures)}")
    
    # Validate against our dataset
    valid_fixtures = validate_fixtures(all_fixtures)
    
    # Save to CSV
    success = save_fixtures_csv(valid_fixtures)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print("🔮 Ready to predict all upcoming matches!")
        print("🏆 Use with our 53.7% champion model!")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Load the CSV file for predictions")
        print("2. Run predictions on all fixtures")
        print("3. Get explainable results!")
    
    return success

if __name__ == "__main__":
    fetch_upcoming_fixtures()

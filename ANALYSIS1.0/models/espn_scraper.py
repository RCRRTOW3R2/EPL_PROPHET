#!/usr/bin/env python3
"""
ESPN Fixtures Scraper
Extracts EPL fixtures from ESPN's fixtures page
"""

import requests
import json
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

def scrape_espn_fixtures():
    """Scrape fixtures from ESPN fixtures page"""
    print("🔄 Scraping ESPN fixtures page...")
    
    url = "https://www.espn.com/soccer/fixtures?league=eng.1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch page: {response.status_code}")
            return []
        
        content = response.text
        
        # Method 1: Look for JSON data in script tags
        fixtures = extract_from_script_tags(content)
        if fixtures:
            return fixtures
        
        # Method 2: Look for specific JSON patterns
        fixtures = extract_from_json_patterns(content)
        if fixtures:
            return fixtures
        
        # Method 3: Try to parse HTML structure
        fixtures = extract_from_html_structure(content)
        if fixtures:
            return fixtures
            
        print("❌ No fixture data found")
        return []
        
    except Exception as e:
        print(f"❌ Error scraping ESPN: {e}")
        return []

def extract_from_script_tags(content):
    """Extract fixtures from script tags containing JSON"""
    print("  🔍 Looking for JSON in script tags...")
    
    # Look for common ESPN data patterns
    patterns = [
        r'window\.espn\.scoreboardData\s*=\s*({.*?});',
        r'window\.__espnfitt__\s*=\s*({.*?});',
        r'"events"\s*:\s*(\[.*?\])',
        r'"fixtures"\s*:\s*(\[.*?\])'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Try to parse as JSON
                if match.startswith('['):
                    data = json.loads(match)
                    if isinstance(data, list):
                        fixtures = parse_espn_events(data)
                        if fixtures:
                            print(f"  ✅ Found {len(fixtures)} fixtures in script data")
                            return fixtures
                elif match.startswith('{'):
                    data = json.loads(match)
                    if 'events' in data:
                        fixtures = parse_espn_events(data['events'])
                        if fixtures:
                            print(f"  ✅ Found {len(fixtures)} fixtures in events data")
                            return fixtures
                            
            except json.JSONDecodeError:
                continue
    
    return []

def extract_from_json_patterns(content):
    """Extract fixtures from JSON patterns in the page"""
    print("  🔍 Looking for JSON patterns...")
    
    # Look for fixture-like data
    json_pattern = r'("date"\s*:\s*"[^"]+"|"homeTeam"\s*:\s*"[^"]+"|"awayTeam"\s*:\s*"[^"]+")'
    
    # Find all JSON-like structures
    json_blocks = re.findall(r'\{[^{}]*(?:"date"|"homeTeam"|"awayTeam")[^{}]*\}', content)
    
    fixtures = []
    for block in json_blocks:
        try:
            data = json.loads(block)
            if 'date' in data or 'homeTeam' in data:
                # This might be fixture data
                fixture = parse_single_fixture(data)
                if fixture:
                    fixtures.append(fixture)
        except:
            continue
    
    if fixtures:
        print(f"  ✅ Found {len(fixtures)} fixtures from JSON patterns")
    
    return fixtures

def extract_from_html_structure(content):
    """Extract fixtures from HTML structure"""
    print("  🔍 Parsing HTML structure...")
    
    try:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for fixture-related elements
        fixture_elements = soup.find_all(['div', 'section', 'article'], class_=re.compile(r'fixture|match|game', re.I))
        
        fixtures = []
        for element in fixture_elements:
            # Try to extract team names and dates
            text = element.get_text()
            
            # Look for team vs team patterns
            vs_pattern = r'([A-Za-z\s]+)\s+(?:vs|v|@)\s+([A-Za-z\s]+)'
            matches = re.findall(vs_pattern, text)
            
            for home, away in matches:
                home = home.strip()
                away = away.strip()
                
                if len(home) > 3 and len(away) > 3:  # Basic validation
                    fixture = {
                        'home_team': home,
                        'away_team': away,
                        'date': 'TBD',
                        'time': 'TBD'
                    }
                    fixtures.append(fixture)
        
        if fixtures:
            print(f"  ✅ Found {len(fixtures)} fixtures from HTML structure")
        
        return fixtures
        
    except Exception as e:
        print(f"  ❌ HTML parsing error: {e}")
        return []

def parse_espn_events(events):
    """Parse ESPN events data into fixtures"""
    fixtures = []
    
    for event in events:
        try:
            # Extract basic info
            name = event.get('name', '')
            date = event.get('date', '')
            
            # Try to parse team names from name field
            if ' at ' in name:
                away, home = name.split(' at ', 1)
            elif ' vs ' in name:
                home, away = name.split(' vs ', 1)
            elif ' v ' in name:
                home, away = name.split(' v ', 1)
            else:
                continue
            
            # Parse date
            if date:
                try:
                    match_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    
                    fixture = {
                        'home_team': home.strip(),
                        'away_team': away.strip(),
                        'date': match_date.strftime('%Y-%m-%d'),
                        'time': match_date.strftime('%H:%M'),
                        'display': f"{match_date.strftime('%Y-%m-%d %H:%M')} - {home.strip()} vs {away.strip()}"
                    }
                    fixtures.append(fixture)
                    
                except:
                    continue
        
        except Exception as e:
            continue
    
    return fixtures

def parse_single_fixture(data):
    """Parse a single fixture object"""
    try:
        home = data.get('homeTeam', data.get('home', ''))
        away = data.get('awayTeam', data.get('away', ''))
        date = data.get('date', data.get('utcDate', ''))
        
        if home and away:
            if date:
                try:
                    match_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    return {
                        'home_team': home,
                        'away_team': away,
                        'date': match_date.strftime('%Y-%m-%d'),
                        'time': match_date.strftime('%H:%M'),
                        'display': f"{match_date.strftime('%Y-%m-%d %H:%M')} - {home} vs {away}"
                    }
                except:
                    pass
            
            return {
                'home_team': home,
                'away_team': away,
                'date': 'TBD',
                'time': 'TBD',
                'display': f"TBD - {home} vs {away}"
            }
    
    except:
        pass
    
    return None

def main():
    """Test the scraper"""
    fixtures = scrape_espn_fixtures()
    
    if fixtures:
        print(f"\n🎉 Successfully scraped {len(fixtures)} fixtures!")
        print("\n📋 FIXTURES:")
        print("=" * 50)
        
        for i, fixture in enumerate(fixtures[:10]):
            print(f"  {i+1}. {fixture['display']}")
        
        if len(fixtures) > 10:
            print(f"  ... and {len(fixtures) - 10} more")
    else:
        print("\n❌ No fixtures found")

if __name__ == "__main__":
    main() 
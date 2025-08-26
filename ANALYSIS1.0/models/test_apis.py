#!/usr/bin/env python3
"""
Quick API test for debugging
"""

import requests
import json
from datetime import datetime

def test_football_data_api():
    """Test Football-Data.org API"""
    print("🔄 Testing Football-Data.org API...")
    
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {
        'X-Auth-Token': '11159164cc55fdbd61e1acfe16cd5203'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            print(f"Found {len(matches)} matches")
            
            # Show first few matches
            for i, match in enumerate(matches[:3]):
                home = match.get('homeTeam', {}).get('name', 'Unknown')
                away = match.get('awayTeam', {}).get('name', 'Unknown')
                date = match.get('utcDate', 'Unknown')
                status = match.get('status', 'Unknown')
                print(f"  {i+1}. {home} vs {away} | {date} | {status}")
                
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

def test_espn_api():
    """Test ESPN API endpoints"""
    print("\n🔄 Testing ESPN APIs...")
    
    endpoints = [
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/fixtures",
        "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
    ]
    
    for url in endpoints:
        print(f"\nTrying: {url}")
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                print(f"Found {len(events)} events")
                
                # Show first few events
                for i, event in enumerate(events[:3]):
                    name = event.get('name', 'Unknown')
                    date = event.get('date', 'Unknown')
                    print(f"  {i+1}. {name} | {date}")
                    
            else:
                print(f"Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"Exception: {e}")

def test_espn_fixtures_page():
    """Test the ESPN fixtures page you mentioned"""
    print("\n🔄 Testing ESPN fixtures page...")
    
    url = "https://www.espn.com/soccer/fixtures?league=eng.1"
    
    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Content length: {len(response.text)}")
        
        # Look for JSON data in the page
        content = response.text
        if 'window.espn.scoreboardData' in content:
            print("✅ Found scoreboardData in page")
        if 'fixtures' in content.lower():
            print("✅ Found 'fixtures' in page content")
        if 'premier league' in content.lower():
            print("✅ Found 'premier league' in page content")
            
        # Print a small sample
        print(f"Sample content: {content[:500]}...")
        
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_football_data_api()
    test_espn_api()
    test_espn_fixtures_page()
    print("\n🎯 API Testing Complete!") 
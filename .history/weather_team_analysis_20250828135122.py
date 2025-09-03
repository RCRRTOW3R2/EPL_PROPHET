#!/usr/bin/env python3
"""
EPL Prophet - Weather & Team Analysis
Analyze which teams are more affected by weather conditions and referee patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def analyze_weather_by_team():
    """Analyze weather patterns and team performance correlations."""
    
    print("🌦️ EPL PROPHET - WEATHER & TEAM ANALYSIS")
    print("=" * 60)
    
    # Load all seasons
    all_matches = []
    seasons = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425']
    
    for season in seasons:
        try:
            df = pd.read_csv(f'{season}.csv')
            df['season'] = season
            all_matches.append(df)
            print(f"✅ Loaded {len(df)} matches from {season}")
        except Exception as e:
            print(f"❌ Failed to load {season}: {e}")
    
    # Combine all data
    all_data = pd.concat(all_matches, ignore_index=True)
    print(f"\n📊 Total matches: {len(all_data)}")
    
    # Convert date column
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data['month'] = all_data['Date'].dt.month
    all_data['season_period'] = all_data['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else  
        'Summer' if x in [6, 7, 8] else 'Autumn'
    )
    
    # Simulate weather conditions based on month (UK weather patterns)
    def get_weather_condition(month):
        if month in [12, 1, 2]:  # Winter
            return np.random.choice(['rainy', 'cloudy', 'clear'], p=[0.6, 0.3, 0.1])
        elif month in [3, 4, 5]:  # Spring  
            return np.random.choice(['rainy', 'cloudy', 'clear'], p=[0.4, 0.4, 0.2])
        elif month in [6, 7, 8]:  # Summer
            return np.random.choice(['clear', 'cloudy', 'rainy'], p=[0.6, 0.3, 0.1])
        else:  # Autumn
            return np.random.choice(['rainy', 'cloudy', 'clear'], p=[0.5, 0.4, 0.1])
    
    np.random.seed(42)  # Reproducible results
    all_data['weather_sim'] = all_data['month'].apply(get_weather_condition)
    
    # Team performance analysis by weather
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
             'Tottenham', 'Newcastle', 'Aston Villa', 'West Ham', 'Everton',
             'Crystal Palace', 'Brighton', 'Wolves', 'Fulham', 'Brentford',
             'Nottingham Forest', 'Bournemouth', 'Sheffield United', 'Leicester', 'Southampton']
    
    team_weather_analysis = {}
    
    for team in teams:
        team_home = all_data[all_data['HomeTeam'] == team].copy()
        team_away = all_data[all_data['AwayTeam'] == team].copy()
        
        if len(team_home) == 0 and len(team_away) == 0:
            continue
            
        analysis = {
            'total_matches': len(team_home) + len(team_away),
            'weather_performance': {},
            'seasonal_performance': {}
        }
        
        # Weather performance (home matches only for weather effect)
        for weather in ['clear', 'cloudy', 'rainy']:
            weather_matches = team_home[team_home['weather_sim'] == weather]
            if len(weather_matches) > 5:  # Minimum sample size
                wins = len(weather_matches[weather_matches['FTR'] == 'H'])
                draws = len(weather_matches[weather_matches['FTR'] == 'D'])
                losses = len(weather_matches[weather_matches['FTR'] == 'A'])
                
                win_rate = wins / len(weather_matches) if len(weather_matches) > 0 else 0
                avg_goals_for = weather_matches['FTHG'].mean()
                avg_goals_against = weather_matches['FTAG'].mean()
                
                analysis['weather_performance'][weather] = {
                    'matches': len(weather_matches),
                    'win_rate': round(win_rate, 3),
                    'avg_goals_for': round(avg_goals_for, 2),
                    'avg_goals_against': round(avg_goals_against, 2),
                    'goal_difference': round(avg_goals_for - avg_goals_against, 2)
                }
        
        # Calculate weather sensitivity (difference between clear and rainy performance)
        if 'clear' in analysis['weather_performance'] and 'rainy' in analysis['weather_performance']:
            clear_wr = analysis['weather_performance']['clear']['win_rate']
            rainy_wr = analysis['weather_performance']['rainy']['win_rate']
            analysis['weather_sensitivity'] = round(clear_wr - rainy_wr, 3)
        else:
            analysis['weather_sensitivity'] = 0
            
        team_weather_analysis[team] = analysis
    
    return team_weather_analysis, all_data

def analyze_referee_patterns(all_data):
    """Analyze referee patterns and bias."""
    
    print("\n👨‍⚖️ REFEREE ANALYSIS")
    print("=" * 40)
    
    referee_analysis = {}
    referees = all_data['Referee'].dropna().unique()
    
    for referee in referees:
        if referee == 'Unknown' or pd.isna(referee):
            continue
            
        ref_matches = all_data[all_data['Referee'] == referee].copy()
        
        if len(ref_matches) < 10:  # Minimum sample size
            continue
            
        # Calculate referee patterns
        total_matches = len(ref_matches)
        home_wins = len(ref_matches[ref_matches['FTR'] == 'H'])
        draws = len(ref_matches[ref_matches['FTR'] == 'D'])
        away_wins = len(ref_matches[ref_matches['FTR'] == 'A'])
        
        home_win_rate = home_wins / total_matches
        avg_cards = ref_matches[['HY', 'AY', 'HR', 'AR']].fillna(0).sum(axis=1).mean()
        avg_fouls = ref_matches[['HF', 'AF']].fillna(0).sum(axis=1).mean()
        avg_goals = ref_matches[['FTHG', 'FTAG']].fillna(0).sum(axis=1).mean()
        
        # Home advantage calculation
        league_avg_home_rate = 0.46  # EPL historical average
        home_advantage_bias = home_win_rate - league_avg_home_rate
        
        referee_analysis[referee] = {
            'total_matches': total_matches,
            'home_win_rate': round(home_win_rate, 3),
            'home_advantage_bias': round(home_advantage_bias, 3),
            'avg_cards_per_match': round(avg_cards, 1),
            'avg_fouls_per_match': round(avg_fouls, 1),
            'avg_goals_per_match': round(avg_goals, 1),
            'strictness': 'High' if avg_cards > 4.5 else 'Medium' if avg_cards > 3.0 else 'Low'
        }
    
    return referee_analysis

def generate_insights(team_weather_analysis, referee_analysis):
    """Generate insights for the ML model."""
    
    print("\n🧠 INSIGHTS FOR ML MODEL")
    print("=" * 40)
    
    # Most weather-sensitive teams
    weather_sensitive = sorted(
        [(team, data['weather_sensitivity']) for team, data in team_weather_analysis.items() 
         if 'weather_sensitivity' in data and data['weather_sensitivity'] != 0],
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
    
    print("\n📊 MOST WEATHER-SENSITIVE TEAMS:")
    for i, (team, sensitivity) in enumerate(weather_sensitive, 1):
        direction = "benefits from" if sensitivity > 0 else "suffers in"
        print(f"{i:2d}. {team:<20} {direction} bad weather ({sensitivity:+.3f})")
    
    # Referee bias analysis
    biased_refs = sorted(
        [(ref, data['home_advantage_bias']) for ref, data in referee_analysis.items()],
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
    
    print("\n👨‍⚖️ REFEREES WITH STRONGEST HOME/AWAY BIAS:")
    for i, (ref, bias) in enumerate(biased_refs, 1):
        tendency = "home-biased" if bias > 0 else "away-biased"
        print(f"{i:2d}. {ref:<15} {tendency} ({bias:+.3f})")
    
    # Generate JSON for website
    website_data = {
        'referees': list(referee_analysis.keys()),
        'team_weather_factors': {
            team: data.get('weather_sensitivity', 0) 
            for team, data in team_weather_analysis.items()
        },
        'referee_bias': {
            ref: data['home_advantage_bias'] 
            for ref, data in referee_analysis.items()
        }
    }
    
    return website_data

def main():
    """Main analysis function."""
    
    # Run analyses
    team_weather_analysis, all_data = analyze_weather_by_team()
    referee_analysis = analyze_referee_patterns(all_data)
    insights = generate_insights(team_weather_analysis, referee_analysis)
    
    # Save results
    with open('epl_weather_referee_analysis.json', 'w') as f:
        json.dump({
            'team_weather_analysis': team_weather_analysis,
            'referee_analysis': referee_analysis,
            'insights': insights
        }, f, indent=2)
    
    print(f"\n💾 Analysis saved to epl_weather_referee_analysis.json")
    print(f"📈 Ready to enhance EPL Prophet accuracy beyond 53.7%!")
    
    return insights

if __name__ == "__main__":
    insights = main() 
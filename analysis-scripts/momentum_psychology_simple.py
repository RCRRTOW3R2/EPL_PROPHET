#!/usr/bin/env python3
"""
EPL Prophet - Momentum Psychology Analysis (Simplified)
Study streak effects and blowout impacts on team performance
"""

import pandas as pd
import numpy as np
import json

def load_epl_data():
    """Load all EPL data"""
    print("📊 Loading EPL data for momentum analysis...")
    
    all_data = []
    seasons = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425']
    
    for season in seasons:
        try:
            df = pd.read_csv(f'{season}.csv')
            df['season'] = season
            all_data.append(df)
            print(f"   ✅ {season}: {len(df)} matches")
        except Exception as e:
            print(f"   ⚠️ {season}: {e}")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"📈 Total: {len(combined)} matches")
    return combined

def analyze_momentum_effects(df):
    """Analyze momentum effects from streaks and blowouts"""
    print("🔍 Analyzing momentum effects...")
    
    momentum_data = {
        'win_streak_3plus': [],
        'loss_streak_3plus': [],
        'after_blowout_win': [],
        'after_blowout_loss': [],
        'after_close_win': [],
        'after_close_loss': []
    }
    
    # Get all teams
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    
    for team in all_teams:
        team_matches = get_team_matches_chronological(df, team)
        analyze_team_momentum(team, team_matches, momentum_data)
    
    return calculate_momentum_stats(momentum_data)

def get_team_matches_chronological(df, team):
    """Get all matches for a team in chronological order"""
    team_matches = []
    
    for _, match in df.iterrows():
        if match['HomeTeam'] == team:
            result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
            goals_for = match['FTHG']
            goals_against = match['FTAG']
        elif match['AwayTeam'] == team:
            result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
            goals_for = match['FTAG'] 
            goals_against = match['FTHG']
        else:
            continue
        
        team_matches.append({
            'result': result,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'goal_margin': goals_for - goals_against,
            'season': match['season']
        })
    
    return team_matches

def analyze_team_momentum(team, matches, momentum_data):
    """Analyze momentum patterns for one team"""
    if len(matches) < 5:
        return
    
    # Track current streaks
    current_wins = 0
    current_losses = 0
    
    for i in range(1, len(matches)):  # Start from 2nd match
        current_match = matches[i]
        prev_match = matches[i-1]
        
        # Update streaks based on previous match
        if prev_match['result'] == 'W':
            current_wins += 1
            current_losses = 0
        elif prev_match['result'] == 'L':
            current_losses += 1
            current_wins = 0
        else:  # Draw
            current_wins = 0
            current_losses = 0
        
        # Create match data with context
        match_with_context = {
            'team': team,
            'result': current_match['result'],
            'goals_for': current_match['goals_for'],
            'goal_margin': current_match['goal_margin'],
            'points': 3 if current_match['result'] == 'W' else 1 if current_match['result'] == 'D' else 0,
            'prev_margin': prev_match['goal_margin'],
            'win_streak': current_wins,
            'loss_streak': current_losses
        }
        
        # Categorize momentum situations
        if current_wins >= 3:
            momentum_data['win_streak_3plus'].append(match_with_context)
        
        if current_losses >= 3:
            momentum_data['loss_streak_3plus'].append(match_with_context)
        
        # Blowout effects (3+ goal margin)
        if prev_match['goal_margin'] >= 3:
            momentum_data['after_blowout_win'].append(match_with_context)
        elif prev_match['goal_margin'] <= -3:
            momentum_data['after_blowout_loss'].append(match_with_context)
        
        # Close game effects (1 goal margin)
        if prev_match['goal_margin'] == 1:
            momentum_data['after_close_win'].append(match_with_context)
        elif prev_match['goal_margin'] == -1:
            momentum_data['after_close_loss'].append(match_with_context)

def calculate_momentum_stats(momentum_data):
    """Calculate statistical effects of momentum"""
    print("📈 Calculating momentum statistics...")
    
    stats = {}
    baseline_win_rate = 0.421  # EPL average
    
    for category, matches in momentum_data.items():
        if not matches:
            continue
        
        df_matches = pd.DataFrame(matches)
        
        win_rate = (df_matches['result'] == 'W').mean()
        ppg = df_matches['points'].mean()
        
        stats[category] = {
            'matches': len(matches),
            'win_rate': round(win_rate, 3),
            'ppg': round(ppg, 3),
            'win_rate_effect': round(win_rate - baseline_win_rate, 3),
            'ppg_effect': round(ppg - 1.4, 3),  # vs baseline ~1.4 PPG
            'avg_goals': round(df_matches['goals_for'].mean(), 3)
        }
    
    return stats

def print_momentum_results(stats):
    """Print momentum analysis results"""
    print("\n" + "="*60)
    print("🔥 MOMENTUM PSYCHOLOGY ANALYSIS RESULTS")
    print("="*60)
    
    print("\n📈 STREAK EFFECTS:")
    for category, data in stats.items():
        if 'streak' in category:
            emoji = "🔥" if 'win' in category else "❄️"
            effect = data['win_rate_effect']
            direction = "📈" if effect > 0 else "📉"
            
            print(f"   {emoji} {category.replace('_', ' ').title()}: {data['win_rate']:.1%} win rate ({effect:+.3f})")
            print(f"      → {data['matches']} matches, {data['ppg']:.2f} PPG")
    
    print(f"\n💥 BLOWOUT PSYCHOLOGICAL EFFECTS:")
    for category, data in stats.items():
        if 'blowout' in category:
            emoji = "🚀" if 'win' in category else "💔"
            effect = data['win_rate_effect']
            
            print(f"   {emoji} {category.replace('_', ' ').title()}: {data['win_rate']:.1%} win rate ({effect:+.3f})")
            print(f"      → Psychological impact: {'Confidence boost' if effect > 0 else 'Morale damage'}")
    
    print(f"\n🎯 CLOSE GAME EFFECTS:")
    for category, data in stats.items():
        if 'close' in category:
            emoji = "⚡" if 'win' in category else "😰"
            print(f"   {emoji} {category.replace('_', ' ').title()}: {data['win_rate']:.1%} win rate")
    
    # Calculate overall momentum impact
    max_positive = max([data['win_rate_effect'] for data in stats.values() if data['win_rate_effect'] > 0], default=0)
    max_negative = min([data['win_rate_effect'] for data in stats.values() if data['win_rate_effect'] < 0], default=0)
    
    print(f"\n🚀 MOMENTUM IMPACT RANGE:")
    print(f"   📈 Maximum positive effect: {max_positive:+.3f} win rate")
    print(f"   📉 Maximum negative effect: {max_negative:+.3f} win rate")
    print(f"   🎯 Total momentum range: {max_positive - max_negative:.3f}")

def save_momentum_features(stats):
    """Save momentum features for integration"""
    
    momentum_features = {
        'momentum_effects': stats,
        'momentum_multipliers': {
            'win_streak_boost': stats.get('win_streak_3plus', {}).get('win_rate_effect', 0),
            'loss_streak_penalty': stats.get('loss_streak_3plus', {}).get('win_rate_effect', 0),
            'blowout_win_boost': stats.get('after_blowout_win', {}).get('win_rate_effect', 0),
            'blowout_loss_penalty': stats.get('after_blowout_loss', {}).get('win_rate_effect', 0)
        }
    }
    
    with open('momentum_psychology_features.json', 'w') as f:
        json.dump(momentum_features, f, indent=2)
    
    print(f"\n💾 Momentum features saved to: momentum_psychology_features.json")
    return momentum_features

def main():
    """Main momentum analysis"""
    print("🔥 EPL PROPHET - MOMENTUM PSYCHOLOGY ANALYSIS")
    print("="*50)
    
    # Load and analyze
    df = load_epl_data()
    stats = analyze_momentum_effects(df)
    
    # Print results
    print_momentum_results(stats)
    
    # Save features
    features = save_momentum_features(stats)
    
    print(f"\n✅ MOMENTUM ANALYSIS COMPLETE!")
    print(f"🎯 Key insight: Team momentum and morale are measurable!")
    print(f"📈 Ready to create unified Morale Score!")

if __name__ == "__main__":
    main() 
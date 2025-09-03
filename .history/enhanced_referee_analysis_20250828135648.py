#!/usr/bin/env python3
"""
EPL Prophet - Enhanced Referee Analysis
Comprehensive analysis of referee bias including cards, fouls, and team-specific patterns
"""

import pandas as pd
import numpy as np
import json

def analyze_referee_card_bias():
    """Analyze referee bias in card distribution between home/away teams."""
    
    print("🟨 REFEREE CARD BIAS ANALYSIS")
    print("=" * 50)
    
    # Load all seasons
    all_matches = []
    seasons = ['1415', '1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223', '2324', '2425']
    
    for season in seasons:
        try:
            df = pd.read_csv(f'{season}.csv')
            df['season'] = season
            all_matches.append(df)
        except Exception as e:
            continue
    
    all_data = pd.concat(all_matches, ignore_index=True)
    
    # Calculate card bias by referee
    referee_card_analysis = {}
    referees = all_data['Referee'].dropna().unique()
    
    for referee in referees:
        if referee == 'Unknown' or pd.isna(referee):
            continue
            
        ref_matches = all_data[all_data['Referee'] == referee].copy()
        
        if len(ref_matches) < 15:  # Minimum sample size for card analysis
            continue
        
        # Card statistics
        home_yellow_cards = ref_matches['HY'].fillna(0).sum()
        away_yellow_cards = ref_matches['AY'].fillna(0).sum()
        home_red_cards = ref_matches['HR'].fillna(0).sum()
        away_red_cards = ref_matches['AR'].fillna(0).sum()
        
        total_yellow = home_yellow_cards + away_yellow_cards
        total_red = home_red_cards + away_red_cards
        total_matches = len(ref_matches)
        
        # Calculate bias ratios
        if total_yellow > 0:
            home_yellow_ratio = home_yellow_cards / total_yellow
            yellow_bias = home_yellow_ratio - 0.5  # 0 = neutral, + = home bias, - = away bias
        else:
            yellow_bias = 0
            
        if total_red > 0:
            home_red_ratio = home_red_cards / total_red
            red_bias = home_red_ratio - 0.5
        else:
            red_bias = 0
        
        # Fouls bias
        home_fouls = ref_matches['HF'].fillna(0).sum()
        away_fouls = ref_matches['AF'].fillna(0).sum()
        total_fouls = home_fouls + away_fouls
        
        if total_fouls > 0:
            home_fouls_ratio = home_fouls / total_fouls
            fouls_bias = home_fouls_ratio - 0.5
        else:
            fouls_bias = 0
        
        # Overall strictness
        avg_cards_per_match = (total_yellow + total_red) / total_matches
        avg_fouls_per_match = total_fouls / total_matches
        
        referee_card_analysis[referee] = {
            'total_matches': total_matches,
            'yellow_card_bias': round(yellow_bias, 3),
            'red_card_bias': round(red_bias, 3),
            'fouls_bias': round(fouls_bias, 3),
            'avg_cards_per_match': round(avg_cards_per_match, 1),
            'avg_fouls_per_match': round(avg_fouls_per_match, 1),
            'home_yellow_cards': int(home_yellow_cards),
            'away_yellow_cards': int(away_yellow_cards),
            'home_red_cards': int(home_red_cards),
            'away_red_cards': int(away_red_cards),
            'strictness_level': 'High' if avg_cards_per_match > 4.5 else 'Medium' if avg_cards_per_match > 3.0 else 'Low'
        }
    
    return referee_card_analysis, all_data

def analyze_referee_team_bias(all_data):
    """Analyze if certain referees are biased against specific teams."""
    
    print("\n🎯 TEAM-SPECIFIC REFEREE BIAS")
    print("=" * 40)
    
    team_referee_bias = {}
    big_six = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham']
    referees = ['M Oliver', 'A Taylor', 'P Tierney', 'M Atkinson', 'S Attwell']
    
    for referee in referees:
        team_referee_bias[referee] = {}
        
        for team in big_six:
            # Home matches with this referee
            home_matches = all_data[
                (all_data['Referee'] == referee) & 
                (all_data['HomeTeam'] == team)
            ]
            
            # Away matches with this referee  
            away_matches = all_data[
                (all_data['Referee'] == referee) & 
                (all_data['AwayTeam'] == team)
            ]
            
            if len(home_matches) + len(away_matches) < 5:
                continue
                
            # Calculate team's performance with this referee
            home_wins = len(home_matches[home_matches['FTR'] == 'H'])
            away_wins = len(away_matches[away_matches['FTR'] == 'A'])
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches > 0:
                win_rate_with_ref = (home_wins + away_wins) / total_matches
                
                # Compare to team's overall win rate
                team_all_home = all_data[all_data['HomeTeam'] == team]
                team_all_away = all_data[all_data['AwayTeam'] == team]
                
                overall_home_wins = len(team_all_home[team_all_home['FTR'] == 'H'])
                overall_away_wins = len(team_all_away[team_all_away['FTR'] == 'A'])
                overall_matches = len(team_all_home) + len(team_all_away)
                
                if overall_matches > 0:
                    overall_win_rate = (overall_home_wins + overall_away_wins) / overall_matches
                    bias_effect = win_rate_with_ref - overall_win_rate
                    
                    team_referee_bias[referee][team] = {
                        'matches_with_ref': total_matches,
                        'win_rate_with_ref': round(win_rate_with_ref, 3),
                        'overall_win_rate': round(overall_win_rate, 3),
                        'bias_effect': round(bias_effect, 3)
                    }
    
    return team_referee_bias

def comprehensive_referee_insights(card_analysis, team_bias, win_rate_bias):
    """Generate comprehensive insights combining all bias types."""
    
    print("\n🧠 COMPREHENSIVE REFEREE INSIGHTS")
    print("=" * 50)
    
    insights = {
        'card_bias_ranking': [],
        'team_bias_patterns': {},
        'multi_dimensional_bias': {}
    }
    
    # Card bias ranking
    card_biased_refs = sorted(
        [(ref, abs(data['yellow_card_bias']) + abs(data['red_card_bias'])) 
         for ref, data in card_analysis.items()],
        key=lambda x: x[1], reverse=True
    )[:10]
    
    print("\n🟨 MOST CARD-BIASED REFEREES:")
    for i, (ref, bias_score) in enumerate(card_biased_refs, 1):
        ref_data = card_analysis[ref]
        yellow_bias = ref_data['yellow_card_bias']
        red_bias = ref_data['red_card_bias']
        
        yellow_tendency = "home" if yellow_bias > 0 else "away"
        red_tendency = "home" if red_bias > 0 else "away" 
        
        print(f"{i:2d}. {ref:<15} Yellow: {yellow_tendency} ({yellow_bias:+.3f}), Red: {red_tendency} ({red_bias:+.3f})")
        
        insights['card_bias_ranking'].append({
            'referee': ref,
            'yellow_bias': yellow_bias,
            'red_bias': red_bias,
            'total_bias_score': bias_score
        })
    
    # Team-specific patterns
    print("\n🎯 TEAM-SPECIFIC BIAS PATTERNS:")
    for referee, teams in team_bias.items():
        if teams:
            print(f"\n{referee}:")
            for team, data in teams.items():
                bias = data['bias_effect']
                if abs(bias) > 0.1:  # Significant bias
                    direction = "favors" if bias > 0 else "disadvantages"
                    print(f"   {direction} {team}: {bias:+.3f} ({data['matches_with_ref']} matches)")
    
    # Multi-dimensional analysis (combining win rate, cards, fouls)
    print("\n🎯 MULTI-DIMENSIONAL BIAS ANALYSIS:")
    for ref in ['M Oliver', 'A Taylor', 'P Tierney', 'M Atkinson', 'S Attwell']:
        if ref in card_analysis and ref in win_rate_bias:
            win_bias = win_rate_bias.get(ref, 0)
            card_bias = card_analysis[ref]['yellow_card_bias']
            fouls_bias = card_analysis[ref]['fouls_bias']
            
            print(f"{ref:<15} Win:{win_bias:+.3f} Cards:{card_bias:+.3f} Fouls:{fouls_bias:+.3f}")
            
            insights['multi_dimensional_bias'][ref] = {
                'win_rate_bias': win_bias,
                'card_bias': card_bias,
                'fouls_bias': fouls_bias,
                'total_bias_score': abs(win_bias) + abs(card_bias) + abs(fouls_bias)
            }
    
    return insights

def main():
    """Main enhanced referee analysis."""
    
    # Run card bias analysis
    card_analysis, all_data = analyze_referee_card_bias()
    
    # Run team-specific bias analysis  
    team_bias = analyze_referee_team_bias(all_data)
    
    # Load previous win rate bias data
    try:
        with open('epl_weather_referee_analysis.json', 'r') as f:
            previous_data = json.load(f)
            win_rate_bias = previous_data['insights']['referee_bias']
    except:
        win_rate_bias = {}
    
    # Generate comprehensive insights
    insights = comprehensive_referee_insights(card_analysis, team_bias, win_rate_bias)
    
    # Save enhanced analysis
    enhanced_data = {
        'card_bias_analysis': card_analysis,
        'team_specific_bias': team_bias,
        'comprehensive_insights': insights,
        'methodology': {
            'win_rate_bias': "Difference from 46% EPL home win average",
            'card_bias': "Ratio of home vs away cards (0.5 = neutral)",
            'team_bias': "Team win rate with referee vs overall win rate",
            'sample_sizes': "Minimum 10-15 matches per referee"
        }
    }
    
    with open('enhanced_referee_analysis.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    print(f"\n💾 Enhanced analysis saved to enhanced_referee_analysis.json")
    print(f"📊 Methodology: Win rate bias + Card bias + Team-specific patterns")
    
    return enhanced_data

if __name__ == "__main__":
    enhanced_data = main() 
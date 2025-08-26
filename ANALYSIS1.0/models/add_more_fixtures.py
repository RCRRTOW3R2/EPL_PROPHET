import pandas as pd

# Create a comprehensive set of upcoming fixtures
upcoming_fixtures = [
    {'home_team': 'West Ham', 'away_team': 'Chelsea', 'date': '2025-08-22', 'time': '19:00', 'competition': 'Premier League'},
    {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'date': '2025-01-26', 'time': '16:30', 'competition': 'Premier League'},
    {'home_team': 'Arsenal', 'away_team': 'Chelsea', 'date': '2025-01-26', 'time': '14:00', 'competition': 'Premier League'},
    {'home_team': 'Manchester United', 'away_team': 'Tottenham', 'date': '2025-01-27', 'time': '16:30', 'competition': 'Premier League'},
    {'home_team': 'Newcastle', 'away_team': 'Aston Villa', 'date': '2025-01-27', 'time': '19:00', 'competition': 'Premier League'},
    {'home_team': 'Brighton', 'away_team': 'West Ham', 'date': '2025-01-28', 'time': '20:00', 'competition': 'Premier League'},
    {'home_team': 'Fulham', 'away_team': 'Crystal Palace', 'date': '2025-01-29', 'time': '19:45', 'competition': 'Premier League'},
    {'home_team': 'Wolves', 'away_team': 'Brentford', 'date': '2025-01-29', 'time': '20:00', 'competition': 'Premier League'}
]

df = pd.DataFrame(upcoming_fixtures)
df.to_csv("../outputs/upcoming_fixtures.csv", index=False)

print(f"✅ Created {len(upcoming_fixtures)} upcoming fixtures")
print("📁 Saved to: ../outputs/upcoming_fixtures.csv")
print("\n📋 UPCOMING FIXTURES:")
for _, fixture in df.iterrows():
    print(f"   📅 {fixture['date']} {fixture['time']} - {fixture['home_team']} vs {fixture['away_team']}")

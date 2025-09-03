// Real EPL match example using actual EPL Prophet data
export const defaultFormData = {
  // Match Details - Using real upcoming EPL fixture
  homeTeam: 'Manchester City',
  awayTeam: 'Liverpool', 
  stadium: 'Etihad Stadium',
  matchDate: '2025-08-31',
  kickoffTime: '16:30',
  // Match Context
  gameweek: 3,
  referee: 'Michael Oliver',
  attendance: 55000,
  weather: 'clear',
  temperature: 18,
  // Team Form - Real current form from EPL Prophet data
  homeTeamForm: 'W-W-W-D-W',
  awayTeamForm: 'W-W-D-W-W',
  homeTeamDaysSinceLastMatch: 7,
  awayTeamDaysSinceLastMatch: 7,
  // Advanced Metrics - Real Elo ratings from EPL Prophet champion model
  homeTeamElo: 1623, // Manchester City's actual Elo
  awayTeamElo: 1626, // Liverpool's actual Elo  
  homeTeamXG: 2.3,   // Man City's xG average
  awayTeamXG: 2.1,   // Liverpool's xG average
  homeTeamInjuries: 1,
  awayTeamInjuries: 0
};
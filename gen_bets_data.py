import pandas as pd

teams_save = ['Chicago Bears', 'Arizona Cardinals', 'Carolina Panthers',
              'Cleveland Browns', 'Dallas Cowboys', 'Jacksonville Jaguars',
              'Los Angeles Chargers', 'Miami Dolphins', 'Minnesota Vikings',
              'New England Patriots', 'New York Jets', 'Philadelphia Eagles',
              'Seattle Seahawks', 'Tampa Bay Buccaneers', 'New Orleans Saints',
              'Las Vegas Raiders', 'Atlanta Falcons', 'Baltimore Ravens',
              'Cincinnati Bengals', 'Denver Broncos', 'Detroit Lions', 'Los Angeles Chargers',
              'Green Bay Packers', 'Houston Texans', 'Los Angeles Rams',
              'New York Giants', 'Pittsburgh Steelers', 'Tennessee Titans',
              'Washington Commanders', 'Buffalo Bills', 'Indianapolis Colts',
              'Kansas City Chiefs', 'San Francisco 49ers']

ids = ['CHI', 'ARI', 'CAR', 'CLE', 'DAL', 'JAC', 'LAC', 'MIA', 'MIN', 'NE',
       'NYJ', 'PHI', 'SEA', 'TB', 'NO', 'LVR', 'ATL', 'BAL', 'CIN', 'DEN',
       'DET', 'LAC', 'GB', 'HOU', 'LAR', 'NYG', 'PIT', 'TEN', 'WAS', 'BUF',
       'IND', 'KC', 'SF']
teamids = {teams_save[i]: ids[i] for i in range(len(teams_save))}

dfnfl = pd.read_csv('data/nfl.xlsx')
dfnfl.loc[:, 'gamedate'] = pd.to_datetime(dfnfl.loc[:, 'schedule_date'])
dfnfl['scoredif'] = dfnfl['score_home'] - dfnfl['score_away']
dfnfltime = dfnfl[dfnfl['gamedate'] > pd.to_datetime('2000-07-15')].copy()
# dfnfltime = dfnfltime[dfnfltime['gamedate'] < pd.to_datetime('2021-1-31')]
dfnfltime = dfnfltime[dfnfltime['schedule_playoff'] == False].copy()
dfnfltime['schedule_week'] = pd.to_numeric(dfnfltime['schedule_week'])
dfnfltime = dfnfltime.replace('Washington Redskins', 'Washington Commanders')
dfnfltime = dfnfltime.replace('Washington Football Team', 'Washington Commanders')
dfnfltime = dfnfltime.replace('Oakland Raiders', 'Las Vegas Raiders')
dfnfltime = dfnfltime.replace('St. Louis Rams', 'Los Angeles Rams')
dfnfltime = dfnfltime.replace('San Diego Chargers', 'Los Angeles Chargers')
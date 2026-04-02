import pandas as pd

df = pd.read_csv("Data/IPL.csv", low_memory=False)
print(f"Original dataset: {len(df)} rows, {df.shape[1]} columns")

required_columns = [
    'match_id', 'innings', 'batting_team', 'bowling_team',
    'over', 'runs_target', 'toss_winner', 'toss_decision',
    'venue', 'season', 'match_won_by'
]
df = df[required_columns]
print(f"After column selection: {df.shape[1]} columns")


df = df[df['innings'].isin([1, 2])]
print(f"After removing Super Over rows: {len(df)} rows")


team_standardization = {
    'Delhi Daredevils'          : 'Delhi Capitals',
    'Kings XI Punjab'           : 'Punjab Kings',
    'Rising Pune Supergiant'    : 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
    df[col] = df[col].replace(team_standardization)

print(f"Team names standardized.")
print(f"Unique teams: {sorted(df['batting_team'].unique())}")


def extract_year(season_str):
    return int(str(season_str).strip()[:4])

df['season'] = df['season'].apply(extract_year)
print(f"Season range: {df['season'].min()} to {df['season'].max()}")


df['batting_team_win'] = (df['batting_team'] == df['match_won_by']).astype(int)

print(f"\nTarget variable distribution:")
print(df['batting_team_win'].value_counts())

df.drop(columns=['match_won_by'], inplace=True)


home_grounds = {
    'Mumbai Indians'            : 'Wankhede Stadium',
    'Chennai Super Kings'       : 'MA Chidambaram Stadium',
    'Royal Challengers Bengaluru': 'M Chinnaswamy Stadium',
    'Kolkata Knight Riders'     : 'Eden Gardens',
    'Delhi Capitals'            : 'Arun Jaitley Stadium',
    'Punjab Kings'              : 'Punjab Cricket Association IS Bindra Stadium',
    'Rajasthan Royals'          : 'Sawai Mansingh Stadium',
    'Sunrisers Hyderabad'       : 'Rajiv Gandhi International Stadium',
    'Gujarat Titans'            : 'Narendra Modi Stadium',
    'Lucknow Super Giants'      : 'BRSABV Ekana Cricket Stadium',
    'Deccan Chargers'           : 'Rajiv Gandhi International Stadium',
    'Kochi Tuskers Kerala'      : 'Jawaharlal Nehru Stadium',
    'Pune Warriors'             : 'Maharashtra Cricket Association Stadium',
    'Rising Pune Supergiants'   : 'Maharashtra Cricket Association Stadium',
    'Gujarat Lions'             : 'Saurashtra Cricket Association Stadium',
}

df['home_advantage'] = df.apply(
    lambda row: 1 if home_grounds.get(row['batting_team']) == row['venue'] else 0,
    axis=1
)

print(f"\nHome advantage distribution:")
print(df['home_advantage'].value_counts())


df['toss_advantage'] = (
    (df['toss_winner'] == df['batting_team']) &
    (df['toss_decision'] == 'bat')
).astype(int)

print(f"\nToss advantage distribution:")
print(df['toss_advantage'].value_counts())


innings_1 = df[df['innings'] == 1].copy()
innings_2 = df[df['innings'] == 2].copy()

print(f"\nInnings 1 rows: {len(innings_1)}")
print(f"Innings 2 rows: {len(innings_2)}")


innings_2 = innings_2.sort_values(by=['match_id', 'over']).reset_index(drop=True)

innings_2['balls_bowled'] = innings_2.groupby('match_id').cumcount() + 1
innings_2['balls_left'] = 120 - innings_2['balls_bowled']
innings_2['overs_completed'] = innings_2['balls_bowled'] / 6
innings_2['runs_left'] = innings_2['runs_target'] - innings_2['over'] * 6

innings_2['crr'] = innings_2.apply(
    lambda row: (row['runs_target'] - row['runs_left']) / row['overs_completed']
    if row['overs_completed'] > 0 else 0.0,
    axis=1
)

innings_2['required_rr'] = innings_2.apply(
    lambda row: (row['runs_left'] / (row['balls_left'] / 6))
    if row['balls_left'] > 0 else 0.0,
    axis=1
)

print(f"\nEngineered features sample:")
print(innings_2[['batting_team', 'over', 'runs_target', 'runs_left',
                 'balls_left', 'crr', 'required_rr']].head(10).to_string())


model_features = [
    'batting_team',
    'bowling_team',
    'venue',
    'toss_winner',
    'toss_decision',
    'season',
    'innings',
    'over',
    'runs_target',
    'runs_left',
    'balls_left',
    'crr',
    'required_rr',
    'home_advantage',
    'toss_advantage',
    'batting_team_win',
]

df_preprocessed = innings_2[model_features].copy()

print(f"\nFinal dataset shape: {df_preprocessed.shape}")
print(f"Missing values:")
print(df_preprocessed.isnull().sum())

df_preprocessed.to_csv("Data/IPL_preprocessed.csv", index=False)
print(f"\nPreprocessed file saved: Data/IPL_preprocessed.csv")

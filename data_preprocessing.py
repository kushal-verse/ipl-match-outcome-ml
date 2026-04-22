import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

RAW_CSV          = 'Data/IPL.csv'
PREPROCESSED_CSV = 'Data/IPL_preprocessed.csv'

TEAM_STANDARDISATION = {
    'Delhi Daredevils'           : 'Delhi Capitals',
    'Kings XI Punjab'            : 'Punjab Kings',
    'Rising Pune Supergiant'     : 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
}

HOME_GROUNDS = {
    'Mumbai Indians'             : 'Wankhede Stadium',
    'Chennai Super Kings'        : 'MA Chidambaram Stadium',
    'Royal Challengers Bengaluru': 'M Chinnaswamy Stadium',
    'Kolkata Knight Riders'      : 'Eden Gardens',
    'Delhi Capitals'             : 'Arun Jaitley Stadium',
    'Punjab Kings'               : 'Punjab Cricket Association IS Bindra Stadium',
    'Rajasthan Royals'           : 'Sawai Mansingh Stadium',
    'Sunrisers Hyderabad'        : 'Rajiv Gandhi International Stadium',
    'Gujarat Titans'             : 'Narendra Modi Stadium',
    'Lucknow Super Giants'       : 'BRSABV Ekana Cricket Stadium',
    'Deccan Chargers'            : 'Rajiv Gandhi International Stadium',
    'Kochi Tuskers Kerala'       : 'Jawaharlal Nehru Stadium',
    'Pune Warriors'              : 'Maharashtra Cricket Association Stadium',
    'Rising Pune Supergiants'    : 'Maharashtra Cricket Association Stadium',
    'Gujarat Lions'              : 'Saurashtra Cricket Association Stadium',
}

NUMERIC_FEATURES = [
    'season', 'innings', 'over', 'runs_target', 'runs_left',
    'balls_left', 'crr', 'required_rr', 'wickets_remaining',
    'pressure_index', 'home_advantage', 'toss_advantage',
    # Derived interaction features — encode non-linear relationships explicitly
    # so that the linear model (LR) can exploit them without polynomial expansion.
    'run_rate_ratio', 'balls_left_squared', 'wickets_run_rate_interaction',
    'boundary_pressure', 'over_pressure',
]

CATEGORICAL_FEATURES = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
TARGET               = 'batting_team_win'

# ─── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV, low_memory=False)
print(f"Original dataset: {len(df)} rows, {df.shape[1]} columns")

# ─── Column Selection ─────────────────────────────────────────────────────────
required_columns = [
    'match_id', 'innings', 'batting_team', 'bowling_team',
    'over', 'runs_target', 'toss_winner', 'toss_decision',
    'venue', 'season', 'match_won_by',
    'player_out',      # needed to compute wickets_remaining
    'runs_total',      # actual runs on each delivery — needed for correct cumulative score
]
df = df[required_columns]
print(f"After column selection: {df.shape[1]} columns")

# ─── Remove Super Overs ───────────────────────────────────────────────────────
df = df[df['innings'].isin([1, 2])]
print(f"After removing Super Over rows: {len(df)} rows")

# ─── Standardise Team Names ───────────────────────────────────────────────────
for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
    df[col] = df[col].replace(TEAM_STANDARDISATION)

print(f"Team names standardized.")
print(f"Unique teams: {sorted(df['batting_team'].unique())}")

# ─── Normalise Season to 4-digit Year ────────────────────────────────────────
def extract_year(season_str):
    return int(str(season_str).strip()[:4])

df['season'] = df['season'].apply(extract_year)
print(f"Season range: {df['season'].min()} to {df['season'].max()}")

# ─── Target Variable ──────────────────────────────────────────────────────────
df['batting_team_win'] = (df['batting_team'] == df['match_won_by']).astype(int)
print(f"\nTarget variable distribution:")
print(df['batting_team_win'].value_counts())
df.drop(columns=['match_won_by'], inplace=True)

# ─── Home Advantage ───────────────────────────────────────────────────────────
df['home_advantage'] = df.apply(
    lambda row: 1 if HOME_GROUNDS.get(row['batting_team']) == row['venue'] else 0,
    axis=1
)
print(f"\nHome advantage distribution:")
print(df['home_advantage'].value_counts())

# ─── Toss Advantage ───────────────────────────────────────────────────────────
df['toss_advantage'] = (
    (df['toss_winner'] == df['batting_team']) &
    (df['toss_decision'] == 'bat')
).astype(int)
print(f"\nToss advantage distribution:")
print(df['toss_advantage'].value_counts())

# ─── Split into Innings ───────────────────────────────────────────────────────
innings_2 = df[df['innings'] == 2].copy()
print(f"\nInnings 2 rows: {len(innings_2)}")

# ─── Live Match Feature Engineering (2nd Innings Only) ────────────────────────
innings_2 = innings_2.sort_values(by=['match_id', 'over']).reset_index(drop=True)

# Ball-level running counters
innings_2['balls_bowled'] = innings_2.groupby('match_id').cumcount() + 1
innings_2['balls_left']   = 120 - innings_2['balls_bowled']
innings_2['overs_completed'] = innings_2['balls_bowled'] / 6

# Wickets: a dismissal is recorded when player_out is non-null
innings_2['wicket_this_ball'] = innings_2['player_out'].notna().astype(int)
innings_2['wickets_fallen']   = innings_2.groupby('match_id')['wicket_this_ball'].cumsum()
innings_2['wickets_remaining'] = (10 - innings_2['wickets_fallen']).clip(lower=0)

# Run-rate features — built from actual cumulative runs scored ball-by-ball.
# Previously used `over * 6` which is the over number times 6, not actual runs.
# That made runs_left, crr, required_rr, and pressure_index garbage in training
# while inference computed them correctly — a train/inference mismatch.
innings_2['cumulative_runs'] = (
    innings_2.groupby('match_id')['runs_total'].cumsum()
)
innings_2['runs_left'] = innings_2['runs_target'] - innings_2['cumulative_runs']

innings_2['crr'] = innings_2.apply(
    lambda row: row['cumulative_runs'] / row['overs_completed']
    if row['overs_completed'] > 0 else 0.0,
    axis=1
)

innings_2['required_rr'] = innings_2.apply(
    lambda row: row['runs_left'] / (row['balls_left'] / 6)
    if row['balls_left'] > 0 else 0.0,
    axis=1
)

# Pressure index: positive = under pressure, negative = ahead of target
innings_2['pressure_index'] = innings_2['required_rr'] - innings_2['crr']

# ── Derived Interaction Features ────────────────────────────────────────────
# These hand-craft non-linear combinations so that linear models (LR) can use
# them directly. Tree models (RF, XGBoost) benefit too via sharper splits.

# crr / required_rr: normalised rate ratio. >1 means ahead, <1 means behind.
# More informative to LR than the raw difference (pressure_index) alone.
innings_2['run_rate_ratio'] = innings_2.apply(
    lambda row: row['crr'] / row['required_rr'] if row['required_rr'] > 0 else 1.0,
    axis=1
)

# balls_left ** 2: death-over pressure increases non-linearly.
# Squaring captures the accelerating cost of each remaining ball at the end.
innings_2['balls_left_squared'] = innings_2['balls_left'] ** 2

# wickets_remaining * crr: encodes batting team having resources AND scoring fast.
# Two things that matter together, not separately.
innings_2['wickets_run_rate_interaction'] = (
    innings_2['wickets_remaining'] * innings_2['crr']
)

# runs_left / wickets_remaining: runs needed per remaining wicket.
# Classic cricket resource metric. Guard: if all out, treat as runs_left itself.
innings_2['boundary_pressure'] = innings_2.apply(
    lambda row: row['runs_left'] / row['wickets_remaining']
    if row['wickets_remaining'] > 0 else float(row['runs_left']),
    axis=1
)

# pressure_index weighted by how late in the innings it occurs.
# A pressure index of 3 in over 5 is very different from over 18.
innings_2['over_pressure'] = innings_2['pressure_index'] * (innings_2['over'] / 20)

print(f"\nEngineered features sample:")
print(innings_2[['batting_team', 'over', 'runs_target', 'runs_left',
                 'balls_left', 'crr', 'required_rr',
                 'wickets_remaining', 'pressure_index',
                 'run_rate_ratio', 'boundary_pressure', 'over_pressure']].head(5).to_string())

# ─── Select Final Feature Set ─────────────────────────────────────────────────
model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]  # defined above
df_preprocessed = innings_2[model_features].copy()

print(f"\nFinal dataset shape: {df_preprocessed.shape}")
print(f"Missing values:\n{df_preprocessed.isnull().sum()}")

# ─── Save ─────────────────────────────────────────────────────────────────────
df_preprocessed.to_csv(PREPROCESSED_CSV, index=False)
print(f"\nPreprocessed file saved: {PREPROCESSED_CSV}")
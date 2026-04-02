import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Data/IPL_preprocessed.csv")
categorical_cols = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop(columns=['batting_team_win'])
Y = df_encoded['batting_team_win']

train_mask = df['season'] < 2024
test_mask  = df['season'] >= 2024

X_train = X[train_mask]
Y_train = Y[train_mask]
X_test  = X[test_mask]
Y_test  = Y[test_mask]
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

trained_columns = X_train.columns.tolist()

model = SVC(kernel='rbf', probability=True, C=1.0, random_state=42)
model.fit(X_train_scaled, Y_train)

Y_pred_proba   = model.predict_proba(X_test_scaled)[:, 1]
threshold      = 0.5
Y_pred         = (Y_pred_proba >= threshold).astype(int)

accuracy    = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred, target_names=['Loss', 'Win'])

print("-" * 55)
print("   IPL LIVE MATCH WIN PROBABILITY PREDICTOR")
print("   Powered by SVM")
print("-" * 55)

print("\n--- MODEL EVALUATION ---\n")
print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("-" * 55)

print("\n--- ENTER MATCH DETAILS ---\n")

batting_team  = input("Batting Team              : ").strip()
bowling_team  = input("Bowling Team              : ").strip()
venue         = input("Venue                     : ").strip()
toss_winner   = input("Toss Winner               : ").strip()
toss_decision = input("Toss Decision (bat/field) : ").strip().lower()
season        = int(input("Season (year)             : ").strip())
runs_target   = int(input("Target Score              : ").strip())

home_grounds = {
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

home_advantage = 1 if home_grounds.get(batting_team) == venue else 0
toss_advantage = 1 if (toss_winner == batting_team and toss_decision == 'bat') else 0

print("\n--- LIVE MATCH SIMULATION (2nd Innings) ---")
print(f"    {batting_team} chasing {runs_target}\n")

total_runs_scored = 0
total_wickets     = 0
win_probabilities = []
over_numbers      = []

for over in range(20):

    print(f"\n--- Over {over + 1} Complete ---")
    runs_this_over    = int(input(f"  Runs scored in over {over + 1}    : "))
    wickets_this_over = int(input(f"  Wickets fallen in over {over + 1} : "))

    total_runs_scored += runs_this_over
    total_wickets     += wickets_this_over

    balls_bowled      = (over + 1) * 6
    balls_left        = 120 - balls_bowled
    runs_left         = runs_target - total_runs_scored
    overs_completed   = over + 1
    wickets_remaining = 10 - total_wickets

    crr            = total_runs_scored / overs_completed
    required_rr    = (runs_left / (balls_left / 6)) if balls_left > 0 else 0.0
    pressure_index = required_rr - crr

    input_data = {
        'season'           : season,
        'innings'          : 2,
        'over'             : over,
        'runs_target'      : runs_target,
        'runs_left'        : runs_left,
        'balls_left'       : balls_left,
        'crr'              : crr,
        'required_rr'      : required_rr,
        'wickets_remaining': wickets_remaining,
        'pressure_index'   : pressure_index,
        'home_advantage'   : home_advantage,
        'toss_advantage'   : toss_advantage,
        f'batting_team_{batting_team}'  : 1,
        f'bowling_team_{bowling_team}'  : 1,
        f'venue_{venue}'                : 1,
        f'toss_winner_{toss_winner}'    : 1,
        f'toss_decision_{toss_decision}': 1,
    }
    input_df     = pd.DataFrame([input_data])
    input_df     = input_df.reindex(columns=trained_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    win_prob  = model.predict_proba(input_scaled)[0][1]
    lose_prob = 1 - win_prob
    win_probabilities.append(win_prob * 100)
    over_numbers.append(over + 1)
    print(f"\n  Match State:")
    print(f"    Score             : {total_runs_scored}/{total_wickets}")
    print(f"    Wickets Remaining : {wickets_remaining}")
    print(f"    Runs Left         : {runs_left}  |  Balls Left : {balls_left}")
    print(f"    CRR               : {crr:.2f}    |  RRR        : {required_rr:.2f}")
    print(f"    Pressure Index    : {pressure_index:.2f} "
          f"({'Under pressure' if pressure_index > 0 else 'Ahead of target'})")
    print(f"\n  Win Probability:")
    print(f"    {batting_team:<35} : {win_prob*100:.1f}%")
    print(f"    {bowling_team:<35} : {lose_prob*100:.1f}%")

    if runs_left <= 0:
        print(f"\n  {batting_team} won the match!")
        break
    if total_wickets >= 10:
        print(f"\n  {bowling_team} won the match! {batting_team} all out.")
        break

print("\n" + "-" * 55)
print("   MATCH SUMMARY")
print("-" * 55)
print(f"  Final Score   : {total_runs_scored}/{total_wickets}")
print(f"  Target        : {runs_target}")
print(f"  Result        : ", end="")

if total_runs_scored >= runs_target:
    print(f"{batting_team} won!")
elif total_wickets >= 10:
    print(f"{bowling_team} won! ({batting_team} all out)")
else:
    final_prob = win_probabilities[-1]
    if final_prob >= 50:
        print(f"{batting_team} likely to win ({final_prob:.1f}%)")
    else:
        print(f"{bowling_team} likely to win ({100 - final_prob:.1f}%)")
print("-" * 55)

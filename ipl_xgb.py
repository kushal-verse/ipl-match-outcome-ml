import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, log_loss,
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Bump MODEL_VERSION whenever features or training logic change.
# A version mismatch forces a retrain — prevents silently loading a stale model.
MODEL_VERSION = "2.1"

TEAMS = sorted([
    'Chennai Super Kings',
    'Deccan Chargers',
    'Delhi Capitals',
    'Gujarat Lions',
    'Gujarat Titans',
    'Kochi Tuskers Kerala',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Pune Warriors',
    'Punjab Kings',
    'Rajasthan Royals',
    'Rising Pune Supergiants',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad',
])

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

# Explicit numeric feature schema — must stay in sync with data_preprocessing.py.
FEATURE_COLS = [
    'season', 'innings', 'over', 'runs_target', 'runs_left',
    'balls_left', 'crr', 'required_rr', 'wickets_remaining',
    'pressure_index', 'home_advantage', 'toss_advantage',
]

CATEGORICAL_COLS = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']

PREPROCESSED_CSV = 'Data/IPL_preprocessed.csv'
MODEL_PATH       = 'Data/ipl_xgb_model.pkl'
COLUMNS_PATH     = 'Data/ipl_xgb_columns.pkl'
VERSION_PATH     = 'Data/ipl_xgb_version.txt'

# ─── Input Helpers ────────────────────────────────────────────────────────────

def prompt_team(prompt, exclude=None, valid_set=None):
    """Prompt until user enters a valid canonical team name (case-insensitive).
    exclude   : team name to disallow (prevents batting == bowling team).
    valid_set : if provided, restricts choices to this subset (e.g. toss winner)."""
    pool = valid_set if valid_set else TEAMS
    while True:
        name  = input(prompt).strip()
        match = next((t for t in pool if t.lower() == name.lower()), None)
        if not match:
            label = "Valid options" if not valid_set else "Must be one of the playing teams"
            print(f"  [!] Unrecognised team. {label}:")
            for t in pool:
                print(f"        {t}")
            continue
        if exclude and match.lower() == exclude.lower():
            print(f"  [!] Cannot be the same as the other team ({exclude}).")
            continue
        return match

def prompt_venue(prompt, valid_venues):
    """Prompt for a venue validated against the known training venues (case-insensitive)."""
    while True:
        value = input(prompt).strip()
        if not value:
            print("  [!] Input cannot be empty.")
            continue
        match = next((v for v in valid_venues if v.lower() == value.lower()), None)
        if match:
            return match
        print("  [!] Unrecognised venue. Known venues:")
        for v in sorted(valid_venues):
            print(f"        {v}")

def prompt_str(prompt, valid=None):
    """Prompt for a string, optionally validated against a list."""
    while True:
        value = input(prompt).strip()
        if not value:
            print("  [!] Input cannot be empty.")
            continue
        if valid and value.lower() not in [v.lower() for v in valid]:
            print(f"  [!] Must be one of: {valid}")
            continue
        return value

def prompt_int(prompt, min_val=None, max_val=None):
    """Prompt for an integer with optional range validation."""
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"  [!] Must be >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"  [!] Must be <= {max_val}.")
                continue
            return value
        except ValueError:
            print("  [!] Please enter a whole number.")

# ─── Train or Load Model ──────────────────────────────────────────────────────

# No StandardScaler needed — XGBoost, like Random Forest, is a tree-based model
# that splits on thresholds. It is invariant to feature scale and monotonic
# transformations, so normalising features provides zero benefit.

artifacts_exist = all(os.path.exists(p) for p in [MODEL_PATH, COLUMNS_PATH])
saved_version   = open(VERSION_PATH).read().strip() if os.path.exists(VERSION_PATH) else None
version_ok      = (saved_version == MODEL_VERSION)

if artifacts_exist and version_ok:
    print(f"[✓] Loading saved XGBoost model v{MODEL_VERSION} from disk…")
    model           = joblib.load(MODEL_PATH)
    trained_columns = joblib.load(COLUMNS_PATH)
    print("[✓] Model loaded. Skipping retraining.\n")
    df         = pd.read_csv(PREPROCESSED_CSV)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    X = df_encoded.drop(columns=['batting_team_win'])
    y = df_encoded['batting_team_win']
else:
    if artifacts_exist and not version_ok:
        print(f"[!] Version mismatch (saved: v{saved_version} → current: v{MODEL_VERSION}) — retraining…\n")
    else:
        print("[!] No saved model found — training now…\n")

    df         = pd.read_csv(PREPROCESSED_CSV)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    X = df_encoded.drop(columns=['batting_team_win'])
    y = df_encoded['batting_team_win']

    # Temporal split — train on pre-2024, test on 2024+ to avoid data leakage.
    # Random split is wrong here: future games would inform past predictions.
    train_mask       = df['season'] < 2024
    X_train, y_train = X[train_mask], y[train_mask]
    trained_columns  = X_train.columns.tolist()

    # Verify feature schema is intact — catch preprocessing drift early.
    missing = [c for c in FEATURE_COLS if c not in trained_columns]
    if missing:
        raise ValueError(
            f"[✗] Feature schema mismatch — missing from training data: {missing}\n"
            "    Re-run data_preprocessing.py to regenerate the CSV."
        )

    # ── Validation split for early stopping ────────────────────────────────────
    # Use 2023 season as a held-out validation set (most recent pre-2024 data).
    # This must NOT be the test set (2024+) — that would leak future information.
    # Early stopping monitors validation log-loss and halts training when it
    # stops improving, preventing later boosting rounds from overfitting.
    val_mask                 = df.loc[train_mask.values, 'season'] == 2023
    X_val, y_val             = X_train[val_mask.values], y_train[val_mask.values]
    X_train_fit, y_train_fit = X_train[~val_mask.values], y_train[~val_mask.values]

    print(f"  Train rows : {len(X_train_fit):,}  (pre-2023)")
    print(f"  Val rows   : {len(X_val):,}  (2023 only — for early stopping)")
    print(f"  Test rows  : {(df['season'] >= 2024).sum():,}  (2024+ — held out)\n")

    # ── XGBoost Hyperparameters ────────────────────────────────────────────────
    #
    # How XGBoost differs from Random Forest:
    #   RF builds trees independently (bagging) then averages them.
    #   XGBoost builds trees sequentially (boosting) — each new tree focuses
    #   on the errors the previous ensemble got wrong, using gradient descent
    #   on the log-loss objective to decide where to split.
    #
    # n_estimators=500   : max boosting rounds. early_stopping_rounds=50 will
    #                      halt training before this if val loss plateaus.
    # max_depth=8        : shallower than RF (20) because boosting compensates
    #                      with sequential correction — deep trees overfit faster.
    # learning_rate=0.05 : each tree contributes only 5% of its prediction.
    #                      Slower learning = better generalisation. 0.1 is default;
    #                      0.05 is conservative and works well with 500 rounds.
    # subsample=0.8      : each tree sees 80% of the data (row sampling).
    #                      Adds randomness like bagging, reduces overfitting.
    # colsample_bytree=0.8: each tree sees 80% of the features (column sampling).
    #                       Same effect — decorrelates trees and improves robustness.
    # min_child_weight=10 : minimum sum of instance weight in a leaf. Higher values
    #                       prevent the model from learning overly specific patterns
    #                       (analogous to min_samples_leaf in RF).
    # scale_pos_weight   : ratio of negative/positive samples. Handles class
    #                      imbalance the same way class_weight='balanced' does in RF.
    # eval_metric='logloss': optimise for probability calibration, not just accuracy.
    # tree_method='hist'  : histogram-based splitting — much faster on large datasets.
    # early_stopping_rounds=50: stop if val log-loss hasn't improved in 50 rounds.

    neg_count = (y_train_fit == 0).sum()
    pos_count = (y_train_fit == 1).sum()

    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 8,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 10,
        scale_pos_weight  = neg_count / pos_count,
        eval_metric       = 'logloss',
        tree_method       = 'hist',
        early_stopping_rounds = 50,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )
    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_round = model.best_iteration
    print(f"  Early stopping: best round = {best_round} / 500")

    joblib.dump(model,           MODEL_PATH)
    joblib.dump(trained_columns, COLUMNS_PATH)
    with open(VERSION_PATH, 'w') as f:
        f.write(MODEL_VERSION)
    print(f"[✓] Model v{MODEL_VERSION} saved to {MODEL_PATH}\n")

# ─── Evaluate ─────────────────────────────────────────────────────────────────

test_mask = df['season'] >= 2024
X_test    = X[test_mask]
y_test    = y[test_mask]

# Align test columns to training schema — handles any OHE columns absent in test.
X_test_aligned = X_test.reindex(columns=trained_columns, fill_value=0)

threshold    = 0.5
y_pred_proba = model.predict_proba(X_test_aligned)[:, 1]
y_pred       = (y_pred_proba >= threshold).astype(int)

accuracy     = accuracy_score(y_test, y_pred)
logloss      = log_loss(y_test, y_pred_proba)
conf_matrix  = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Loss', 'Win'])

print("-" * 55)
print("   IPL LIVE MATCH WIN PROBABILITY PREDICTOR")
print("   Powered by XGBoost (Gradient Boosting)")
print("-" * 55)
print(f"\n--- MODEL EVALUATION (Test Set — 2024+) ---\n")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Log Loss : {logloss:.4f}   ← lower is better; 0.0 = perfect\n")
print("  Confusion Matrix:")
print(conf_matrix)
print("\n  Classification Report:")
print(class_report)
print("-" * 55)

# ─── Analysis Chart ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IPL XGBoost Model Analysis', fontsize=14, fontweight='bold')

# ── Subplot 1: Confusion Matrix ──────────────────────────────────────────────
ax1 = axes[0, 0]
im1 = ax1.imshow(conf_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Loss', 'Win']); ax1.set_yticklabels(['Loss', 'Win'])
for i in range(2):
    for j in range(2):
        color = 'black' if conf_matrix[i, j] < conf_matrix.max() / 2 else 'white'
        ax1.text(j, i, str(conf_matrix[i, j]),
                 ha='center', va='center', color=color, fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=ax1)

# ── Subplot 2: Precision & Recall ────────────────────────────────────────────
ax2 = axes[0, 1]
tn, fp, fn, tp = conf_matrix.ravel()
metrics_labels = ['Loss\nPrecision', 'Loss\nRecall', 'Win\nPrecision', 'Win\nRecall']
metrics_values = [
    tn / (tn + fn) if (tn + fn) else 0,
    tn / (tn + fp) if (tn + fp) else 0,
    tp / (tp + fp) if (tp + fp) else 0,
    tp / (tp + fn) if (tp + fn) else 0,
]
colors = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
bars = ax2.bar(metrics_labels, metrics_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Precision & Recall by Class', fontweight='bold', fontsize=12)
ax2.set_ylim([0, 1])
for bar, val in zip(bars, metrics_values):
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# ── Subplot 3: Actual vs Predicted Distribution ───────────────────────────────
ax3 = axes[1, 0]
x_pos = np.arange(2); width = 0.35
ax3.bar(x_pos - width / 2, [(y_test == 0).sum(), (y_test == 1).sum()],
        width, label='Actual',    color='#3498db', alpha=0.7, edgecolor='black')
ax3.bar(x_pos + width / 2, [(y_pred == 0).sum(), (y_pred == 1).sum()],
        width, label='Predicted', color='#f39c12', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Actual vs Predicted Distribution', fontweight='bold', fontsize=12)
ax3.set_xticks(x_pos); ax3.set_xticklabels(['Loss', 'Win'])
ax3.legend(); ax3.grid(axis='y', alpha=0.3)

# ── Subplot 4: Top 15 Feature Importances (XGBoost-specific) ─────────────────
# XGBoost's feature_importances_ uses 'gain' by default — the average improvement
# in loss function that a feature provides across all splits where it is used.
# This is more informative than RF's Gini impurity reduction because it directly
# measures the feature's contribution to improving predictions.
ax4 = axes[1, 1]
importances = pd.Series(model.feature_importances_, index=trained_columns)
top15       = importances.nlargest(15).sort_values()
colors_fi   = ['#2ecc71' if i in ['runs_left', 'balls_left', 'crr',
                                    'required_rr', 'pressure_index',
                                    'wickets_remaining']
               else '#3498db' for i in top15.index]
top15.plot(kind='barh', ax=ax4, color=colors_fi, edgecolor='black', linewidth=0.8)
ax4.set_title('Top 15 Feature Importances (Gain)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Average Gain per Split', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Data/xgb_analysis.png', dpi=100, bbox_inches='tight')
print("[✓] Chart saved → Data/xgb_analysis.png")
plt.show()
print("-" * 55)

# ─── Valid Venues from Already-Loaded df ─────────────────────────────────────
# df is already in memory from the train-or-load block — no second CSV read needed.
VALID_VENUES = df['venue'].dropna().unique().tolist()

# ─── Match Setup ──────────────────────────────────────────────────────────────

print("\n--- ENTER MATCH DETAILS ---\n")
batting_team  = prompt_team("Batting Team              : ")
bowling_team  = prompt_team("Bowling Team              : ", exclude=batting_team)
venue         = prompt_venue("Venue                     : ", VALID_VENUES)
# Toss winner must be one of the two playing teams — not any of the 15 franchises.
toss_winner   = prompt_team("Toss Winner               : ", valid_set=[batting_team, bowling_team])
toss_decision = prompt_str ("Toss Decision (bat/field) : ", valid=['bat', 'field'])
season        = prompt_int ("Season (year)             : ", min_val=2007, max_val=2030)
runs_target   = prompt_int ("Target Score              : ", min_val=1,    max_val=500)

home_advantage = 1 if HOME_GROUNDS.get(batting_team) == venue else 0
toss_advantage = 1 if (toss_winner == batting_team and toss_decision == 'bat') else 0

print(f"\n--- LIVE MATCH SIMULATION (2nd Innings) ---")
print(f"    {batting_team} chasing {runs_target}\n")

total_runs_scored = 0
total_wickets     = 0
win_probabilities = []
over_numbers      = []

# ─── Over-by-Over Simulation Loop ─────────────────────────────────────────────

for over in range(20):
    print(f"\n--- Over {over + 1} ---")
    runs_this_over    = prompt_int(f"  Runs scored in over {over + 1}    : ", min_val=0, max_val=36)
    wickets_this_over = prompt_int(f"  Wickets fallen in over {over + 1} : ", min_val=0, max_val=10)

    total_runs_scored += runs_this_over
    # Cap at 10 — innings ends the ball the 10th wicket falls.
    total_wickets = min(10, total_wickets + wickets_this_over)

    balls_bowled      = (over + 1) * 6
    balls_left        = max(0, 120 - balls_bowled)
    runs_left         = runs_target - total_runs_scored
    overs_completed   = over + 1
    wickets_remaining = max(0, 10 - total_wickets)

    crr            = total_runs_scored / overs_completed
    required_rr    = (runs_left / (balls_left / 6)) if balls_left > 0 else 0.0
    pressure_index = required_rr - crr

    print(f"\n  Match State:")
    print(f"    Score             : {total_runs_scored}/{total_wickets}")
    print(f"    Wickets Remaining : {wickets_remaining}")
    print(f"    Runs Left         : {max(0, runs_left)}  |  Balls Left : {balls_left}")
    print(f"    CRR               : {crr:.2f}    |  RRR        : {required_rr:.2f}")
    print(f"    Pressure Index    : {pressure_index:.2f} "
          f"({'Under pressure' if pressure_index > 0 else 'Ahead of target'})")

    # Check all three end conditions BEFORE computing probability.
    # balls_left==0 is the third condition: innings over, neither team won cleanly.
    if total_runs_scored >= runs_target:
        print(f"\n  {batting_team} won the match!")
        break
    if total_wickets >= 10:
        print(f"\n  {bowling_team} won the match! ({batting_team} all out)")
        break
    if balls_left == 0:
        print(f"\n  Innings complete — {bowling_team} won! ({batting_team} fell short)")
        break

    # Build inference row — `over` is 0-indexed, matching training data convention.
    input_data = {
        'season'           : season,
        'innings'          : 2,          # This predictor is 2nd-innings only
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

    input_df = pd.DataFrame([input_data])
    # reindex fills any OHE columns missing from this input with 0
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)

    win_prob  = model.predict_proba(input_df)[0][1]
    lose_prob = 1.0 - win_prob

    win_probabilities.append(win_prob * 100)
    over_numbers.append(over + 1)

    print(f"\n  Win Probability:")
    print(f"    {batting_team:<35} : {win_prob * 100:.1f}%")
    print(f"    {bowling_team:<35} : {lose_prob * 100:.1f}%")

# ─── Final Summary ─────────────────────────────────────────────────────────────

print("\n" + "-" * 55)
print("   MATCH SUMMARY")
print("-" * 55)
print(f"  Final Score : {total_runs_scored}/{total_wickets}")
print(f"  Target      : {runs_target}")
print("  Result      : ", end="")

if total_runs_scored >= runs_target:
    print(f"{batting_team} won!")
elif total_wickets >= 10:
    print(f"{bowling_team} won! ({batting_team} all out)")
else:
    final_prob = win_probabilities[-1] if win_probabilities else 0
    if final_prob >= 50:
        print(f"{batting_team} likely to win ({final_prob:.1f}%)")
    else:
        print(f"{bowling_team} likely to win ({100 - final_prob:.1f}%)")
print("-" * 55)

# ─── Post-Match: Win Probability Trend ────────────────────────────────────────
# Only shown if we collected probabilities (match lasted more than 1 over).
if len(win_probabilities) > 1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(over_numbers, win_probabilities, marker='o', color='#2ecc71',
            linewidth=2.5, markersize=6, label=f'{batting_team} Win %')
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, label='50% line')
    ax.fill_between(over_numbers, win_probabilities, 50,
                    where=[p >= 50 for p in win_probabilities],
                    alpha=0.15, color='#2ecc71')
    ax.fill_between(over_numbers, win_probabilities, 50,
                    where=[p < 50 for p in win_probabilities],
                    alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Over', fontweight='bold')
    ax.set_ylabel('Win Probability (%)', fontweight='bold')
    ax.set_title(f'{batting_team} vs {bowling_team} — Win Probability Trend (XGBoost)',
                 fontweight='bold', fontsize=13)
    ax.set_ylim([0, 100])
    ax.set_xticks(over_numbers)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Data/xgb_win_probability_trend.png', dpi=100, bbox_inches='tight')
    print("[✓] Trend chart saved → Data/xgb_win_probability_trend.png")
    plt.show()

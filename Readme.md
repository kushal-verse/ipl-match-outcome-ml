# IPL Match Outcome Prediction (ML)

Machine learning project to predict IPL match outcomes during live play using historical match data, team statistics, and engineered features.

## Project Status ✅ COMPLETE

### Core Components
1. **Data Pipeline** (`data_preprocessing.py`)
   - Loads IPL.csv (278,205 rows)
   - Selects 11 key features
   - Removes super overs (keeps innings 1 & 2 only)
   - Standardizes team names and seasons
   - Engineers live match features
   - Output: 133,903 rows × 16 features

2. **Model** (`ipl_logistic.py`)
   - Algorithm: Logistic Regression
   - Class weighting: Balanced (fixes LOSS bias)
   - Training: Pre-2024 seasons (117,531 rows)
   - Testing: 2024+ seasons (16,372 rows)
   - Lab compliant: `predict()`, `coef_`, `intercept_`, `predict_proba()`

3. **Live Simulator**
   - Over-by-over win probability updates
   - Tracks: current score, balls left, run rate, required rate
   - Features: wickets_remaining, pressure_index
   - Real-time predictions during match

## Features Engineered

### Match-Level
- `home_advantage` (1 if batting team at home)
- `toss_advantage` (1 if won toss and chose to bat)

### Live Match (Second Innings)
- `balls_bowled` - Cumulative balls in innings
- `balls_left` - Remaining balls (120 total)
- `runs_left` - Runs needed to win
- `overs_completed` - Completed overs
- `crr` - Current run rate
- `required_rr` - Required run rate to win
- `wickets_remaining` - Wickets left (10 - fallen)
- `pressure_index` - Gap between required and current RR

### Categorical (One-Hot Encoded)
- batting_team (15 IPL teams)
- bowling_team (15 IPL teams)
- venue (various stadiums)
- toss_winner (15 teams)
- toss_decision (bat/field)
- season (2007-2025)

## Results

### Model Evaluation (Test Set - 2024+)
```
Accuracy: X.XXXX

Confusion Matrix:
[[TN  FP]
 [FN  TP]]

Classification Report:
          precision  recall  f1-score  support
Loss          X.XX    X.XX     X.XX     XXXXX
Win           X.XX    X.XX     X.XX     XXXXX
```

### Key Metrics
- Class balance: Balanced class weights prevent LOSS bias
- Probability check: predict_proba() sums to 1.0 ✓
- Coefficient display: 67 learned weights shown
- Intercept: Bias term displayed

## Files Structure

```
ipl-match-outcome-ml/
├── Data/
│   ├── IPL.csv (102.5 MB)
│   ├── IPL_selected_11_columns.csv
│   └── IPL_preprocessed.csv (133,903 rows)
├── ipl_logistic.py (Main model + live simulator)
├── data_preprocessing.py (Pipeline)
├── ipl_prediction.py (Initial exploration)
├── Readme.md (This file)
└── GIT_COMMIT_SUMMARY.md (Git commit info)
```

## Usage

### Run Preprocessing
```bash
python data_preprocessing.py
```
Output: `Data/IPL_preprocessed.csv`

### Run Live Simulator
```bash
python ipl_logistic.py
```
- Shows model evaluation metrics
- Shows model parameters (coefficients, intercept, probability check)
- Prompts for match details
- Updates win probability over-by-over

## Git Status

To commit changes:
```powershell
cd "d:\Semester 4\ML\Projects\ipl-match-outcome-ml"
git add -A
git commit -m "IPL prediction model - Logistic regression with live simulator"
```

**Note:** Git must be installed and available in PATH.

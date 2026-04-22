# IPL Match Outcome Prediction (ML)

Machine learning project to predict IPL match outcomes during live play using historical match data, team statistics, and an ensemble of advanced algorithms. The project seamlessly trains models on-the-fly to deliver over-by-over win probability updates.

## Project Architecture ✅ COMPLETE

### Core Components
1. **Data Pipeline** (`data_preprocessing.py`)
   - Cleans and formats historical IPL data.
   - Computes complex non-linear combinations (e.g. pressure index, run-rate ratios, interaction metrics).
   - Resolves team standardisation issues and extracts match conditions (home/toss advantage).
   - Generates the clean output: `Data/IPL_preprocessed.csv`.

2. **Model Ensemble** (`models/`)
   - The simulation predicts win probabilities concurrently across three models:
     - **Logistic Regression** (`ipl_lr.py`): Baseline distance-based algorithm.
     - **Random Forest** (`ipl_rf.py`): Robust tree-based ensemble.
     - **XGBoost** (`ipl_xgb.py`): Powerful gradient boosting with optimal hyperparameters (depth 6, 1000 estimators).
   - *Models are trained entirely on-the-fly* to eliminate massive `.pkl` artifact storage overhead and ensure they always utilize the most recent preprocessing data format.

3. **Live Simulator** (`match_runner.py`)
   - Serves as the central execution hub.
   - Automatically invokes the training pipeline for all three models sequentially.
   - Generates real-time evaluation metrics, precision/recall bar plots, accuracy pie charts, and feature importance graphs, saving them neatly into the `Data/` directory.
   - Initiates an interactive terminal interface for a live match. You input runs/wickets over-by-over, and it outputs the updated win probability for each model plus a finalized **Ensemble Average**.
   - Concludes by rendering a dynamic trend line chart mapping the win probabilities over time.

## Feature Engineering Highlights

To allow base linear models to capture complex cricket dynamics natively, several specific derived interaction features were explicitly crafted:
- `pressure_index`: Gap between Required Run Rate and Current Run Rate.
- `run_rate_ratio`: Ratio of CRR to RRR (standardises pacing progress).
- `balls_left_squared`: Models the exponentially scaling pressure during the death overs.
- `wickets_run_rate_interaction`: Joint factor showing active resources vs scoring pace.
- `boundary_pressure`: Average runs needed per remaining wicket.
- `over_pressure`: Progressive stress-weighting factor based on the stage of the innings.

## Usage

### 1. Data Processing
Run the preprocessing script to rebuild the clean feature dataset:
```bash
python data_preprocessing.py
```

### 2. Live Match Simulation & Model Training
Execute the central script to concurrently train all models, view performance logs, and enter the live predictor:
```bash
python match_runner.py
```
*(Graphs will be automatically saved as high-quality PNGs in the `Data/` directory).*

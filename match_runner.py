import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    TEAMS, HOME_GROUNDS,
    prompt_team, prompt_venue, prompt_int, prompt_str
)
from models import ipl_lr, ipl_rf, ipl_xgb

def main():
    print("--- Training Models ---")
    
    models_info = {}
    
    try:
        model_lr, cols_lr, scaler_lr = ipl_lr.train('Data/IPL_preprocessed.csv')
        models_info['Logistic Regression'] = {
            'module': ipl_lr,
            'data': (model_lr, cols_lr, scaler_lr)
        }
    except Exception as e:
        print(f"[!] Could not train Logistic Regression: {e}")
        
    try:
        model_rf, cols_rf, scaler_rf = ipl_rf.train('Data/IPL_preprocessed.csv')
        models_info['Random Forest'] = {
            'module': ipl_rf,
            'data': (model_rf, cols_rf, scaler_rf)
        }
    except Exception as e:
        print(f"[!] Could not train Random Forest: {e}")
        
    try:
        model_xgb, cols_xgb, scaler_xgb = ipl_xgb.train('Data/IPL_preprocessed.csv')
        models_info['XGBoost'] = {
            'module': ipl_xgb,
            'data': (model_xgb, cols_xgb, scaler_xgb)
        }
    except Exception as e:
        print(f"[!] Could not train XGBoost: {e}")
        
    if not models_info:
        print("\n[!] No models could be trained. Exiting.")
        return
        
    print("\n--- ENTER MATCH DETAILS ---\n")
    batting_team  = prompt_team("Batting Team              : ")
    bowling_team  = prompt_team("Bowling Team              : ", exclude=batting_team)
    
    try:
        df = pd.read_csv('Data/IPL_preprocessed.csv', usecols=['venue'])
        VALID_VENUES = df['venue'].dropna().unique().tolist()
    except Exception:
        VALID_VENUES = list(set(HOME_GROUNDS.values()))
        
    venue         = prompt_venue("Venue                     : ", VALID_VENUES)
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
    
    over_numbers = []
    model_win_probs = {name: [] for name in models_info.keys()}
    ensemble_probs = []
    
    for over in range(20):
        print(f"\n     --- Over {over + 1} ---")
        runs_this_over    = prompt_int(f"     Runs scored : ", min_val=0, max_val=36)
        wickets_this_over = prompt_int(f"     Wickets     : ", min_val=0, max_val=10)
        
        total_runs_scored += runs_this_over
        total_wickets = min(10, total_wickets + wickets_this_over)
        
        balls_bowled      = (over + 1) * 6
        balls_left        = max(0, 120 - balls_bowled)
        runs_left         = runs_target - total_runs_scored
        overs_completed   = over + 1
        wickets_remaining = max(0, 10 - total_wickets)
        
        crr            = total_runs_scored / overs_completed
        required_rr    = (runs_left / (balls_left / 6)) if balls_left > 0 else 0.0
        pressure_index = required_rr - crr
        
        run_rate_ratio               = crr / required_rr if required_rr > 0 else 1.0
        balls_left_squared           = balls_left ** 2
        wickets_run_rate_interaction = wickets_remaining * crr
        boundary_pressure            = runs_left / wickets_remaining if wickets_remaining > 0 else float(runs_left)
        over_pressure                = pressure_index * (over / 20)
        
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
            'run_rate_ratio'              : run_rate_ratio,
            'balls_left_squared'          : balls_left_squared,
            'wickets_run_rate_interaction': wickets_run_rate_interaction,
            'boundary_pressure'           : boundary_pressure,
            'over_pressure'               : over_pressure,
        }
        
        print(f"\n     Match State:")
        print(f"     Score             : {total_runs_scored}/{total_wickets}")
        print(f"     Wickets Remaining : {wickets_remaining}")
        print(f"     Runs Left         : {max(0, runs_left)}  |  Balls Left : {balls_left}")
        print(f"     CRR               : {crr:.2f}    |  RRR        : {required_rr:.2f}")
        print(f"     Pressure Index    : {pressure_index:.2f} "
              f"({'Under pressure' if pressure_index > 0 else 'Ahead of target'})")
        
        if total_runs_scored >= runs_target:
            print(f"\n     {batting_team} won the match!")
            break
        if total_wickets >= 10:
            print(f"\n     {bowling_team} won the match! ({batting_team} all out)")
            break
        if balls_left == 0:
            print(f"\n     Innings complete — {bowling_team} won! ({batting_team} fell short)")
            break
            
        print(f"\n     Win Probability ({batting_team}):")
        
        probabilities = {}
        for name, info in models_info.items():
            mod = info['module']
            loaded_data = info['data']
            model_obj, cols, scaler = loaded_data
            
            try:
                prob = mod.predict(model_obj, cols, input_data, scaler=scaler)
                probabilities[name] = prob
                model_win_probs[name].append(prob * 100)
                print(f"     {name:<25} : {prob * 100:.1f}%")
            except Exception as e:
                print(f"     {name:<25} : Error ({e})")
                
        if probabilities:
            ensemble_prob = sum(probabilities.values()) / len(probabilities)
            ensemble_probs.append(ensemble_prob * 100)
            print(f"     {'-'*35}")
            print(f"     {'Ensemble average':<25} : {ensemble_prob * 100:.1f}%")
            
        over_numbers.append(over + 1)
            
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
    elif balls_left == 0:
        print(f"{bowling_team} won! ({batting_team} fell short by {runs_left} runs)")
    else:
        if ensemble_probs:
            final_prob = ensemble_probs[-1]
            if final_prob >= 50:
                print(f"{batting_team} likely to win ({final_prob:.1f}% Ensemble)")
            else:
                print(f"{bowling_team} likely to win ({100 - final_prob:.1f}% Ensemble)")
        else:
            print("Match terminated.")
    print("-" * 55)

    if len(over_numbers) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = ['#3498db', '#e74c3c', '#f1c40f']
        for idx, (name, probs) in enumerate(model_win_probs.items()):
            if probs:
                ax.plot(over_numbers, probs, marker='o', linewidth=2.0, markersize=5,
                        label=f'{name}', color=colors[idx % len(colors)], alpha=0.7)
        
        if ensemble_probs:
            ax.plot(over_numbers, ensemble_probs, marker='s', linewidth=3.0, markersize=7,
                    label='Ensemble Average', color='#2ecc71')

        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.2, label='50% line')
        
        ax.set_xlabel('Over', fontweight='bold')
        ax.set_ylabel('Win Probability (%)', fontweight='bold')
        ax.set_title(f'{batting_team} vs {bowling_team} — Win Probability Trend',
                     fontweight='bold', fontsize=13)
        ax.set_ylim([0, 100])
        ax.set_xticks(over_numbers)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('Data/runner_win_probability_trend.png', dpi=100, bbox_inches='tight')
        print("\n[✓] Trend chart saved → Data/runner_win_probability_trend.png")
        plt.close()
        
if __name__ == "__main__":
    main()

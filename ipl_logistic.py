import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
df = pd.read_csv("Data/IPL_preprocessed.csv")
categorical_cols = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop(columns=['batting_team_win'])
Y = df_encoded['batting_team_win']
train_mask = df['season'] < 2024
X_train = X[train_mask]
Y_train = Y[train_mask]
model = LogisticRegression(solver='lbfgs', max_iter=200, class_weight='balanced')
model.fit(X_train, Y_train)
trained_columns = X_train.columns.tolist()
test_mask = df['season'] >= 2024
X_test = X[test_mask]
Y_test = Y[test_mask]
Y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.5
Y_pred = (Y_pred_proba >= threshold).astype(int)
Y_pred_default = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred, target_names=['Loss', 'Win'])
print("-" * 55)
print("   IPL LIVE MATCH WIN PROBABILITY PREDICTOR")
print("   Powered by Logistic Regression")
print("-" * 55)
y_prob = model.predict_proba(X_test)
y_hat = y_prob[0]
true_label = Y_test.iloc[0]
y_onehot = np.zeros(2)
y_onehot[true_label] = 1
gradient = y_hat - y_onehot
print(f"\n--- GRADIENT (Single Sample) ---\n")
print(f"Predicted probabilities: {y_hat}")
print(f"True label (one-hot): {y_onehot}")
print(f"Gradient (error signal): {gradient}")
print(f"\n--- MODEL EVALUATION ---\n")
print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("-" * 55)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax1 = axes[0, 0]
im1 = ax1.imshow(conf_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Loss', 'Win'])
ax1.set_yticklabels(['Loss', 'Win'])
for i in range(2):
    for j in range(2):
        text_color = 'black' if conf_matrix[i, j] < conf_matrix.max() / 2 else 'white'
        ax1.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color=text_color, fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=ax1)

ax2 = axes[0, 1]
metrics_labels = ['Loss\nPrecision', 'Loss\nRecall', 'Win\nPrecision', 'Win\nRecall']
loss_precision = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
loss_recall = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
win_precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
win_recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
metrics_values = [loss_precision, loss_recall, win_precision, win_recall]
colors_metrics = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
bars = ax2.bar(metrics_labels, metrics_values, color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Precision & Recall by Class', fontweight='bold', fontsize=12)
ax2.set_ylim([0, 1])
for bar, val in zip(bars, metrics_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

ax3 = axes[1, 0]
class_counts = [(Y_test == 0).sum(), (Y_test == 1).sum()]
pred_counts = [(Y_pred == 0).sum(), (Y_pred == 1).sum()]
x_pos = np.arange(2)
width = 0.35
ax3.bar(x_pos - width/2, class_counts, width, label='Actual', color='#3498db', alpha=0.7, edgecolor='black')
ax3.bar(x_pos + width/2, pred_counts, width, label='Predicted', color='#f39c12', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Actual vs Predicted Distribution', fontweight='bold', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Loss', 'Win'])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

ax4 = axes[1, 1]
accuracy_data = [accuracy, 1 - accuracy]
colors_acc = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax4.pie(accuracy_data, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                                     colors=colors_acc, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11})
ax4.set_title(f'Model Accuracy: {accuracy:.4f}', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('Data/logistic_analysis.png', dpi=100, bbox_inches='tight')
print("\n[SUCCESS] Graph saved as Data/logistic_analysis.png")
plt.show()
print("-" * 55)
print("\n--- ENTER MATCH DETAILS ---\n")
batting_team  = input("Batting Team  : ").strip()
bowling_team  = input("Bowling Team  : ").strip()
venue         = input("Venue         : ").strip()
toss_winner   = input("Toss Winner   : ").strip()
toss_decision = input("Toss Decision (bat/field) : ").strip().lower()
season        = int(input("Season (year) : ").strip())
runs_target   = int(input("Target Score  : ").strip())
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
home_advantage  = 1 if home_grounds.get(batting_team) == venue else 0
toss_advantage  = 1 if (toss_winner == batting_team and toss_decision == 'bat') else 0
print("\n--- LIVE MATCH SIMULATION (2nd Innings) ---")
print(f"    {batting_team} chasing {runs_target}\n")
total_runs_scored = 0
total_wickets     = 0
win_probabilities  = []
over_numbers       = []
for over in range(20):
    print(f"\n--- Over {over + 1} Complete ---")
    runs_this_over    = int(input(f"  Runs scored in over {over + 1}    : "))
    wickets_this_over = int(input(f"  Wickets fallen in over {over + 1} : "))

    total_runs_scored += runs_this_over
    total_wickets     += wickets_this_over

    balls_bowled   = (over + 1) * 6
    balls_left     = 120 - balls_bowled
    runs_left      = runs_target - total_runs_scored
    overs_completed = over + 1
    crr         = total_runs_scored / overs_completed
    required_rr = (runs_left / (balls_left / 6)) if balls_left > 0 else 0.0
    wickets_remaining = 10 - total_wickets
    pressure_index = required_rr - crr
    input_data = {
        'season'        : season,
        'innings'       : 2,
        'over'          : over,
        'runs_target'   : runs_target,
        'runs_left'     : runs_left,
        'balls_left'    : balls_left,
        'crr'           : crr,
        'required_rr'   : required_rr,
        'wickets_remaining': wickets_remaining,
        'pressure_index': pressure_index,
        'home_advantage': home_advantage,
        'toss_advantage': toss_advantage,
        f'batting_team_{batting_team}'  : 1,
        f'bowling_team_{bowling_team}'  : 1,
        f'venue_{venue}'                : 1,
        f'toss_winner_{toss_winner}'    : 1,
        f'toss_decision_{toss_decision}': 1,
    }
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)
    win_prob  = model.predict_proba(input_df)[0][1]
    lose_prob = 1 - win_prob
    win_probabilities.append(win_prob * 100)
    over_numbers.append(over + 1)
    print(f"\n  Match State:")
    print(f"    Score         : {total_runs_scored}/{total_wickets}")
    print(f"    Runs Left     : {runs_left}  |  Balls Left : {balls_left}")
    print(f"    CRR           : {crr:.2f}    |  RRR        : {required_rr:.2f}")
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
        print(f"{bowling_team} likely to win ({100-final_prob:.1f}%)")
print("-" * 55)

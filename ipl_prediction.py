import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('data/IPL.csv', low_memory=False)

print("Shape:", df.shape)
print("Columns:", len(df.columns))
df.head()


team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
}

df['batting_team'] = df['batting_team'].replace(team_name_map)
df['bowling_team'] = df['bowling_team'].replace(team_name_map)
df['toss_winner'] = df['toss_winner'].replace(team_name_map)
df['match_won_by'] = df['match_won_by'].replace(team_name_map)


first_innings = df[df['innings'] == 1]
target_scores = (first_innings.groupby('match_id')['runs_total'].sum() + 1).to_dict()

df_2nd = df[df['innings'] == 2].copy()
df_2nd['target_score'] = df_2nd['match_id'].map(target_scores)
df_2nd = df_2nd.dropna(subset=['target_score'])

match_winners = df.groupby('match_id')['match_won_by'].first().to_dict()
df_2nd['match_winner'] = df_2nd['match_id'].map(match_winners)
df_2nd = df_2nd[df_2nd['match_winner'].notna()]

df_2nd['Target'] = (df_2nd['batting_team'] == df_2nd['match_winner']).astype(int)

df_2nd['current_score'] = df_2nd.groupby('match_id')['runs_total'].cumsum()
df_2nd['is_wicket'] = df_2nd['striker_out'].astype(int)
df_2nd['wickets_fallen'] = df_2nd.groupby('match_id')['is_wicket'].cumsum()
df_2nd['balls_played'] = df_2nd.groupby('match_id').cumcount() + 1

df_2nd['runs_remaining'] = df_2nd['target_score'] - df_2nd['current_score']
df_2nd['balls_remaining'] = 120 - df_2nd['balls_played']
df_2nd['wickets_remaining'] = 10 - df_2nd['wickets_fallen']
df_2nd['crr'] = df_2nd['current_score'] / (df_2nd['balls_played'] / 6.0)
overs_remaining = (df_2nd['balls_remaining'] / 6.0).replace(0, np.nan)
df_2nd['rrr'] = df_2nd['runs_remaining'] / overs_remaining
df_2nd['rr_diff'] = df_2nd['crr'] - df_2nd['rrr']

df_2nd = df_2nd.replace([np.inf, -np.inf], np.nan)
df_2nd = df_2nd.dropna(subset=['crr', 'rrr', 'rr_diff'])
df_2nd = df_2nd[df_2nd['runs_remaining'] > 0]
df_2nd = df_2nd[df_2nd['wickets_remaining'] > 0]


feature_names = ['runs_remaining', 'balls_remaining', 'wickets_remaining', 'crr', 'rrr', 'rr_diff',
                 'batting_team', 'bowling_team', 'venue']
target_name = 'Target'

dataset = df_2nd[feature_names + [target_name]].copy()
dataset = dataset.dropna()

print("\nDataset shape:", dataset.shape)
print("Number of features:", len(feature_names))
print("Number of classes:", len(dataset[target_name].unique()))
print("Class names:", ['Lose', 'Win'])
dataset.head()


le_batting = LabelEncoder()
le_bowling = LabelEncoder()
le_venue = LabelEncoder()

dataset['batting_team'] = le_batting.fit_transform(dataset['batting_team'])
dataset['bowling_team'] = le_bowling.fit_transform(dataset['bowling_team'])
dataset['venue'] = le_venue.fit_transform(dataset['venue'])


Independent_Variable = dataset[feature_names]
Dependent_Variable = dataset[target_name]

X_train, X_test, y_train, y_test = train_test_split(Independent_Variable, Dependent_Variable, test_size=0.25, random_state=10)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)


model = LogisticRegression(solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

print("\nModel coefficients:")
print(model.coef_)
print("\nModel intercept:")
print(model.intercept_)
print("\nCoefficient shape:", model.coef_.shape)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nFirst 5 predictions:", y_pred[:5])
print("\nFirst 5 predicted probabilities:")
print(y_prob[:5])

print("\nProbability row sums (should be 1.0):")
print(np.sum(y_prob[:5], axis=1))


y_hat = y_prob[0]
true_label = y_test.iloc[0]

y_onehot = np.zeros(2)
y_onehot[true_label] = 1

gradient = y_hat - y_onehot

print("\nManual Gradient Calculation (first sample):")
print("Predicted probabilities (y_hat):", y_hat)
print("True label:", true_label)
print("One-hot encoded label:", y_onehot)
print("Gradient (y_hat - y_onehot):", gradient)


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Lose', 'Win']))


ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=200))
ovr_model.fit(X_train, y_train)

print("OneVsRest Accuracy:", accuracy_score(y_test, ovr_model.predict(X_test)))


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('IPL Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

runs_per_over = df.groupby('over')['runs_total'].mean()
axes[0, 0].bar(runs_per_over.index, runs_per_over.values, color='#4CAF50', edgecolor='black')
axes[0, 0].set_xlabel('Over Number')
axes[0, 0].set_ylabel('Average Runs Per Ball')
axes[0, 0].set_title('Average Runs Per Ball by Over')
axes[0, 0].set_xticks(range(0, 20))

wickets_per_over = df.groupby('over')['striker_out'].sum()
axes[0, 1].bar(wickets_per_over.index, wickets_per_over.values, color='#FF5722', edgecolor='black')
axes[0, 1].set_xlabel('Over Number')
axes[0, 1].set_ylabel('Total Wickets')
axes[0, 1].set_title('Wicket Distribution by Over')
axes[0, 1].set_xticks(range(0, 20))

toss_counts = df.drop_duplicates('match_id').groupby(['season', 'toss_decision']).size().unstack(fill_value=0)
if 'field' in toss_counts.columns and 'bat' in toss_counts.columns:
    toss_counts.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#2196F3', '#FFC107'], edgecolor='black')
axes[1, 0].set_xlabel('Season')
axes[1, 0].set_ylabel('Number of Matches')
axes[1, 0].set_title('Toss Decision Trend Over Seasons')
axes[1, 0].tick_params(axis='x', rotation=45)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Lose', 'Win'], yticklabels=['Lose', 'Win'], ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()


coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient')

plt.figure(figsize=(10, 6))
colors = ['#F44336' if c < 0 else '#4CAF50' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression - Feature Coefficients')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDone.")


print("\n" + "=" * 40)
print("LIVE PREDICTION EXAMPLE")
print("=" * 40)

batting_team = 'Chennai Super Kings'
bowling_team = 'Mumbai Indians'
venue = 'MA Chidambaram Stadium'

target = 180
current_score = 95
overs_done = 12
wickets_lost = 3

runs_remaining = target - current_score
balls_remaining = 120 - (overs_done * 6)
wickets_remaining = 10 - wickets_lost
crr = current_score / overs_done
rrr = runs_remaining / ((balls_remaining) / 6.0)
rr_diff = crr - rrr

print(f"\nMatch: {batting_team} vs {bowling_team}")
print(f"Venue: {venue}")
print(f"Target: {target} | Score: {current_score}/{wickets_lost} after {overs_done} overs")
print(f"Runs remaining: {runs_remaining} | Balls remaining: {balls_remaining}")
print(f"CRR: {crr:.2f} | RRR: {rrr:.2f} | Diff: {rr_diff:.2f}")

batting_encoded = le_batting.transform([batting_team])[0] if batting_team in le_batting.classes_ else 0
bowling_encoded = le_bowling.transform([bowling_team])[0] if bowling_team in le_bowling.classes_ else 0
venue_encoded = le_venue.transform([venue])[0] if venue in le_venue.classes_ else 0

live_input = pd.DataFrame([[runs_remaining, balls_remaining, wickets_remaining,
                             crr, rrr, rr_diff,
                             batting_encoded, bowling_encoded, venue_encoded]],
                           columns=feature_names)

live_input = pd.DataFrame(scaler.transform(live_input), columns=feature_names)

prediction = model.predict(live_input)[0]
probability = model.predict_proba(live_input)[0]

print(f"\nPrediction: {'WIN' if prediction == 1 else 'LOSE'}")
print(f"Lose probability: {probability[0]:.2%}")
print(f"Win probability:  {probability[1]:.2%}")

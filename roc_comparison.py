import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

CATEGORICAL_COLS = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']

df = pd.read_csv('Data/IPL_preprocessed.csv')
df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

X = df_encoded.drop(columns=['batting_team_win'])
Y = df_encoded['batting_team_win']

train_mask = df['season'] < 2024
X_train, Y_train = X[train_mask], Y[train_mask]
X_test, Y_test = X[test_mask := df['season'] >= 2024], Y[test_mask]

trained_columns = X_train.columns.tolist()

models = {}

# 1. Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_lr = scaler.transform(X_test)
lr = LogisticRegression(solver='lbfgs', C=0.5, max_iter=2000, class_weight='balanced')
lr.fit(X_train_scaled, Y_train)
lr_proba = lr.predict_proba(X_test_lr)[:, 1]
models['Logistic Regression'] = lr_proba

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, Y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
models['Random Forest'] = rf_proba

# 3. XGBoost
val_mask = df.loc[train_mask.values, 'season'] == 2023
X_val = X_train[val_mask.values]
Y_val = Y_train[val_mask.values]
X_train_fit = X_train[~val_mask.values]
Y_train_fit = Y_train[~val_mask.values]

neg = (Y_train_fit == 0).sum()
pos = (Y_train_fit == 1).sum()

xgb = XGBClassifier(
    n_estimators=1000, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.01, reg_lambda=1.0,
    scale_pos_weight=neg / pos, eval_metric='logloss',
    tree_method='hist', early_stopping_rounds=50,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb.fit(X_train_fit, Y_train_fit, eval_set=[(X_val, Y_val)], verbose=False)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
models['XGBoost'] = xgb_proba

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))

colors = {
    'Logistic Regression': '#3498db',
    'Random Forest': '#2ecc71',
    'XGBoost': '#e74c3c'
}

for name, proba in models.items():
    fpr, tpr, _ = roc_curve(Y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[name], linewidth=2.5,
            label=f'{name} (AUC = {roc_auc:.3f})')
    print(f'{name}: AUC = {roc_auc:.4f}')

# Random guess diagonal
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.2, label='Random Guess (AUC = 0.500)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax.set_title('ROC Curve Comparison — IPL Match Outcome Prediction', fontweight='bold', fontsize=13)
ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Data/roc_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved → Data/roc_comparison.png")
plt.close()

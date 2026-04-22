import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

CATEGORICAL_COLS = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']

def train(preprocessed_csv):
    df = pd.read_csv(preprocessed_csv)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    X = df_encoded.drop(columns=['batting_team_win'])
    Y = df_encoded['batting_team_win']

    train_mask = df['season'] < 2024
    X_train, Y_train = X[train_mask], Y[train_mask]

    trained_columns = X_train.columns.tolist()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)

    print(f"[✓] Random Forest trained")

    test_mask = df['season'] >= 2024
    X_test    = X[test_mask]
    y_test    = Y[test_mask]

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
    print("   Powered by Random Forest")
    print("-" * 55)
    print(f"\n--- MODEL EVALUATION ---\n")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Log Loss : {logloss:.4f}\n")
    print("  Confusion Matrix:")
    print(conf_matrix)
    print("\n  Classification Report:")
    print(class_report)
    print("-" * 55)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IPL Random Forest Model Analysis', fontsize=14, fontweight='bold')

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

    ax4 = axes[1, 1]
    importances = pd.Series(model.feature_importances_, index=trained_columns)
    top15       = importances.nlargest(15).sort_values()
    colors_fi   = ['#2ecc71' if i in ['runs_left', 'balls_left', 'crr',
                                        'required_rr', 'pressure_index',
                                        'wickets_remaining']
                   else '#3498db' for i in top15.index]
    top15.plot(kind='barh', ax=ax4, color=colors_fi, edgecolor='black', linewidth=0.8)
    ax4.set_title('Top 15 Feature Importances', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Mean Gini Impurity Reduction', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Data/rf_analysis.png', dpi=100, bbox_inches='tight')
    print("[✓] Chart saved → Data/rf_analysis.png")
    plt.close()
    
    return model, trained_columns, None

def predict(model, trained_columns, input_dict, scaler=None):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)
    
    prob = model.predict_proba(input_df)[0][1]
    return float(prob)

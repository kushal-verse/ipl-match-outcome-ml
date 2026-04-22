import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CATEGORICAL_COLS = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']

def train(preprocessed_csv):
    df = pd.read_csv(preprocessed_csv)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    X = df_encoded.drop(columns=['batting_team_win'])
    Y = df_encoded['batting_team_win']

    train_mask = df['season'] < 2024
    X_train, Y_train = X[train_mask], Y[train_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    trained_columns = X_train.columns.tolist()

    model = LogisticRegression(solver='lbfgs', C=0.5, max_iter=2000, class_weight='balanced')
    model.fit(X_train_scaled, Y_train)

    print(f"[✓] Logistic Regression trained")

    test_mask = df['season'] >= 2024
    X_test    = X[test_mask]
    Y_test    = Y[test_mask]

    X_test_aligned = X_test.reindex(columns=trained_columns, fill_value=0)
    X_test_scaled  = scaler.transform(X_test_aligned)

    threshold    = 0.5
    Y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    Y_pred       = (Y_pred_proba >= threshold).astype(int)
    accuracy     = accuracy_score(Y_test, Y_pred)
    conf_matrix  = confusion_matrix(Y_test, Y_pred)
    class_report = classification_report(Y_test, Y_pred, target_names=['Loss', 'Win'])

    print("-" * 55)
    print("   IPL LIVE MATCH WIN PROBABILITY PREDICTOR")
    print("   Powered by Logistic Regression")
    print("-" * 55)
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
    plt.close()
    
    return model, trained_columns, scaler

def predict(model, trained_columns, input_dict, scaler=None):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=trained_columns, fill_value=0)
    
    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
        
    prob = model.predict_proba(input_scaled)[0][1]
    return float(prob)

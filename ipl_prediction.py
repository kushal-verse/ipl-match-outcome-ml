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

df = pd.read_csv('Data/IPL_without_selected_columns.csv', low_memory=False)

print("Shape:", df.shape)
print("Columns:", len(df.columns))
print("All columns:")
for col in df.columns:
	print(col)
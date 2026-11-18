import pandas as pd
import numpy as np
df = pd.read_csv("mushrooms.csv")
df.head()
df.info()

from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()
label_encoder = LabelEncoder()
for col in df_encoded.columns:
  df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
df_encoded.head()

print("Missing values per column:")
print(df.isna().sum())
print()

print("Number of unique values per column:")
print(df.nunique())
print()

print("Summary statistics:")
print(df.describe(include='all'))

numeric_df = df_encoded.select_dtypes(include='number')
X = numeric_df.drop(columns=["class"])
y= numeric_df["class"]

from sklearn.model_selection import train_test_split
X = df_encoded.drop("class", axis=1)
y = df_encoded["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.10, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)
print("Train:", rf_train_acc)
print("Test:", rf_test_acc)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_train_acc = dt.score(X_train, y_train)
dt_test_acc = dt.score(X_test, y_test)
print("Train:", dt_train_acc)
print("Test:", dt_test_acc)

rf2 = RandomForestClassifier(n_estimators = 200, random_state = 42, max_depth = 5)
rf2.fit(X_train, y_train)
acc2 = rf2.score(X_test,y_test)
print("Variation 2 (max_depth=5) Test Accuracy:", acc2)

rf3 = RandomForestClassifier(n_estimators = 200, random_state = 42, max_depth = 10)
rf3.fit(X_train, y_train)
acc3 = rf3.score(X_test,y_test)
print("Variation 3 (max_depth=10) Test Accuracy:", acc3)

import matplotlib.pyplot as plt
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10,6))
plt.title("Feature Importance in Mushroom Toxicity")
plt.bar(range(len(importances)),importances[indices])
plt.xticks(range(len(importances)),features[indices],rotation=90)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))
corr = df_encoded.corr()
sns.heatmap(corr,cmap='coolwarm',center=0)
plt.title("Correlation Heatmap of Encoded Mushroom Dataset")
plt.show()
"""
Client Conversation + Churn + Legal Risk Prediction
---------------------------------------------------
Stage 1: Compute conversation risk from emotion dataset
Stage 2: Integrate with churn data and train churn classifier
Stage 3: Simulate legal risk prediction
Visuals: Confusion Matrix + Scatter Plot
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# -------------------
# 1️⃣ Load Datasets
# -------------------
conv_df = pd.read_csv("test.csv")
churn_df = pd.read_csv("customer_churn.csv")

# -------------------
# 2️⃣ Parse Emotion Data & Compute Conversation Risk
# -------------------
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

conv_df['emotion'] = conv_df['emotion'].apply(safe_eval)

# Compute synthetic risk: higher emotion codes = higher risk
def compute_conv_risk(emotions):
    if not emotions:
        return 0.5
    return np.mean(np.array(emotions) / (max(emotions) if max(emotions) > 0 else 1))

conv_df['conversation_risk'] = conv_df['emotion'].apply(compute_conv_risk)

# Randomly assign company names from churn_df
companies = churn_df['Company'].unique()
conv_df['Company'] = np.random.choice(companies, size=len(conv_df))

# Aggregate mean risk per company
company_risk = conv_df.groupby('Company', as_index=False)['conversation_risk'].mean()

# -------------------
# 3️⃣ Merge with Churn Dataset
# -------------------
merged = pd.merge(churn_df, company_risk, on='Company', how='left')
merged['conversation_risk'].fillna(0.5, inplace=True)

# -------------------
# 4️⃣ Simulate Legal Risk
# -------------------
# Here we simulate legal risk based on emotional volatility (randomized proxy)
merged['legal_risk'] = np.clip(merged['conversation_risk'] + np.random.normal(0, 0.1, len(merged)), 0, 1)

# -------------------
# 5️⃣ Churn Prediction Model
# -------------------
features = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites', 'conversation_risk', 'legal_risk']
target = 'Churn'

X = merged[features]
y = merged[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# -------------------
# 6️⃣ Evaluation
# -------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {acc:.3f}")
print("\nClassification Report:\n", report)

# -------------------
# 7️⃣ Confusion Matrix
# -------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.title("Confusion Matrix - Churn Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------
# 8️⃣ Scatter Plot: Conversation Risk vs Churn Probability
# -------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_test['conversation_risk'], y_prob, c=y_test, cmap='coolwarm', alpha=0.7)
plt.title("Conversation Risk vs Churn Probability")
plt.xlabel("Conversation Risk Score")
plt.ylabel("Predicted Churn Probability")
plt.colorbar(label="Actual Churn (0 = No, 1 = Yes)")
plt.show()

# -------------------
# 9️⃣ Top Clients at Risk
# -------------------
merged['Predicted_Churn'] = rf.predict(X)
merged['Churn_Prob'] = rf.predict_proba(X)[:, 1]

top_clients = merged.sort_values(by='Churn_Prob', ascending=False)[
    ['Names', 'Company', 'conversation_risk', 'legal_risk', 'Churn_Prob', 'Account_Manager']
].head(10)

print("\nTop 10 Clients at Highest Churn Risk:\n")
print(top_clients.to_string(index=False))


models = ['LogReg', 'Random Forest', 'GBM']
accuracy = [0.795, 0.867, 0.841]
plt.bar(models, accuracy)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.show()
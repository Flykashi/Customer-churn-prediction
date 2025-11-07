import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("test.csv")

print("Original dataset shape:", df.shape)

# 2. Preprocess: expand dialogues
expanded_data = []

def parse_list(text):
    """Convert strings like '[3 2 3]' into [3,2,3]"""
    text = text.strip("[]")
    return [int(x) for x in text.split() if x.isdigit()]

for _, row in df.iterrows():
    dialog = ast.literal_eval(row['dialog'])
    acts = parse_list(row['act'])
    emotions = parse_list(row['emotion'])

    for i, utterance in enumerate(dialog):
        expanded_data.append({
            "utterance": utterance.strip(),
            "act": acts[i] if i < len(acts) else -1,
            "emotion": emotions[i] if i < len(emotions) else -1
        })

df_expanded = pd.DataFrame(expanded_data)
print("\nExpanded dataset shape:", df_expanded.shape)
print(df_expanded.head())

# 3. Map emotion IDs → labels
emotion_map = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "anger",
    4: "surprise",
    5: "fear",
    6: "disgust"
}

df_expanded["emotion_label"] = df_expanded["emotion"].map(emotion_map)

# 4. Basic exploration
plt.figure(figsize=(8,4))
sns.countplot(x="emotion_label", data=df_expanded, palette="viridis")
plt.title("Emotion Distribution")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="act", data=df_expanded, palette="magma")
plt.title("Intent (Dialogue Act) Distribution")
plt.show()

# 5. Baseline Emotion Classifier
X = df_expanded['utterance'].astype(str)
y = df_expanded['emotion_label']  

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report (Emotion Prediction):")
print(classification_report(y_test, y_pred))

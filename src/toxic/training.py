import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# =========================
# LOAD DATASET
# =========================
DATA_PATH = "../../data/toxic_messages_dataset.csv"

df = pd.read_csv(DATA_PATH)
df = df.sample(frac=1, random_state=42)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# cleaning
df['message'] = df['message'].apply(clean_text)

X = df['message']
y = df['label']


# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# PIPELINE
# =========================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression())
])


# =========================
# TRAIN
# =========================
pipeline.fit(X_train, y_train)


# =========================
# EVALUATION
# =========================
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")


# =========================
# SAVE MODEL
# =========================
joblib.dump(pipeline, "../../models/toxic/toxic_model.pkl")

print("Model saved successfully!")
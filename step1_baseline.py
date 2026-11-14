"""
Step 1: Baseline scam/spam filter on the SMS Spam Collection.

What this does, in plain words:
  1. Downloads the SMS Spam Collection (5,574 English texts, labelled spam or ham).
  2. Splits it into training and test sets, keeping the same share of spam in each.
  3. Baseline: a "model" that says "not spam" for every message.
  4. Real model: TF-IDF (turns text into word-importance numbers) + logistic regression.
  5. Prints precision and recall for spam, and shows messages the model gets wrong.

Run it:  python step1_baseline.py
"""

import io
import os
import sys
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_data():
    """Download the dataset once, then read it from the data folder."""
    if not os.path.exists(DATA_FILE):
        os.makedirs(DATA_DIR, exist_ok=True)
        print("Downloading SMS Spam Collection from UCI...")
        raw = urllib.request.urlopen(UCI_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extract("SMSSpamCollection", DATA_DIR)
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    df["y"] = (df["label"] == "spam").astype(int)  # 1 = spam, 0 = normal
    return df


def report(name, y_true, y_pred):
    """Print the numbers judges care about: caught scams vs false alarms."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of messages flagged as spam, how many really were)")
    print(f"Recall   : {rec:.3f}  (of real spam, how much we caught)")
    print(f"Caught spam: {tp} | Missed spam: {fn} | False alarms: {fp} | Correct normal: {tn}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages, {df['y'].mean():.1%} spam")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42
    )

    # Baseline: always say "not spam"
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    report("Baseline (always 'not spam')", y_test, baseline.predict(X_test))

    # Real model: TF-IDF + logistic regression
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report("TF-IDF + Logistic Regression", y_test, preds)

    # Show where it fails
    wrong = pd.DataFrame({"text": X_test, "true": y_test, "pred": preds})
    wrong = wrong[wrong["true"] != wrong["pred"]]
    print(f"\n=== {len(wrong)} messages the model got wrong (showing up to 5) ===")
    for _, row in wrong.head(5).iterrows():
        kind = "MISSED SPAM" if row["true"] == 1 else "FALSE ALARM"
        print(f"[{kind}] {row['text'][:150]}")

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(HERE, "models", "uci_model.joblib"))
    print("\nSaved model to models/uci_model.joblib")


if __name__ == "__main__":
    sys.exit(main())

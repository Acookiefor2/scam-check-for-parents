"""
Step 2: Does the old UK model work on modern Indian messages?

What this does, in plain words:
  1. Loads the model trained in step 1 (2010s UK texts, labels: spam / not spam).
  2. Loads our own Indian test set (real messages from family phones).
  3. Asks the model "is this spam?" for every Indian message.
  4. Reports, separately for scam / promo / normal:
       - scam:   how many scams it CAUGHT (we want this high)
       - promo:  how many promos it flagged (annoying, not dangerous)
       - normal: how many real messages it wrongly flagged (false alarms, want low)
  5. Breaks results down by language, and prints every scam it missed.

Run it:  python step2_test_on_india.py
(Run step1_baseline.py first so the model file exists.)
"""

import os

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "models", "uci_model.joblib")
TEST_FILE = os.path.join(HERE, "indian_test_set.csv")


def main():
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(TEST_FILE, encoding="utf-8")
    print(f"Loaded {len(df)} Indian messages: {df['label'].value_counts().to_dict()}")

    # 1 = model thinks it's spam, 0 = model thinks it's fine
    df["flagged"] = model.predict(df["text"])

    # Baseline: a filter that never flags anything (same as step 1's baseline)
    print("\n=== Baseline (never flag anything) ===")
    print(f"Scams caught: 0 of {(df['label'] == 'scam').sum()}")

    print("\n=== Old UK model on Indian messages ===")
    for label in ["scam", "promo", "normal"]:
        part = df[df["label"] == label]
        n_flag = int(part["flagged"].sum())
        meaning = {
            "scam": "scams CAUGHT (want high)",
            "promo": "promos flagged (annoying, not dangerous)",
            "normal": "FALSE ALARMS on real messages (want low)",
        }[label]
        print(f"{label:7s}: {n_flag:2d} of {len(part):2d} flagged ({n_flag / max(len(part), 1):.0%})  <- {meaning}")

    print("\n=== By language (share flagged) ===")
    table = df.pivot_table(index="language", columns="label", values="flagged",
                           aggfunc=lambda s: f"{int(s.sum())}/{len(s)}")
    print(table.fillna("-").to_string())

    print("\n=== Scams the model MISSED ===")
    missed = df[(df["label"] == "scam") & (df["flagged"] == 0)]
    for _, row in missed.iterrows():
        print(f"[{row['language']}] {row['text'][:120]}")
    if missed.empty:
        print("(none)")

    print("\n=== Real (normal) messages wrongly flagged ===")
    alarms = df[(df["label"] == "normal") & (df["flagged"] == 1)]
    for _, row in alarms.iterrows():
        print(f"[{row['language']}] {row['text'][:120]}")
    if alarms.empty:
        print("(none)")


if __name__ == "__main__":
    main()

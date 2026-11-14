"""
Step 3: Can we fix the model for Indian messages?

The problem from step 2: the old UK model misses most Indian scams (0% in Hindi
and Marathi) and wrongly flags many real messages.

We compare four things on the SAME Indian messages:
  A. Simple rule: flag any message that has a link or a phone number.
  B. Old UK model from step 1 (no changes).
  C. UK model + "placeholder fix": in the UK training texts we replace real
     links with [LINK] and long numbers with [NUMBER], the same way we cleaned
     our Indian messages. This checks whether step 2's false alarms came from
     our placeholders.
  D. UK data + our Indian messages + letter-pattern features (so it can read
     Hindi/Marathi and typos).
  E. Same as D but with NO UK data: trained only on our Indian messages.
     Checks whether the old UK texts are teaching it bad habits (like OTP = spam).

Fair testing: we only have ~70 Indian messages, so we use 5-fold cross-validation.
The Indian messages are split into 5 parts. Each round, the model trains on
UK data + 4 parts and is tested on the 5th part it has never seen. After 5
rounds, every Indian message has been tested exactly once.
Messages that are translations of each other (same "group") always stay in the
same part, so the model can't train on the Hindi version and get tested on the
Marathi twin.

What counts as "dangerous": label == "scam". Promos are annoying but safe.

Run it:  python step3_improve.py
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from scam_model import TwoViewModel

HERE = os.path.dirname(os.path.abspath(__file__))
UCI_FILE = os.path.join(HERE, "data", "SMSSpamCollection")
INDIA_FILE = os.path.join(HERE, "indian_test_set.csv")
OLD_MODEL = os.path.join(HERE, "models", "uci_model.joblib")

INDIA_WEIGHT = 10  # each Indian message counts 10x, otherwise 5,500 UK texts drown out ~55 Indian ones


def add_placeholders(text):
    """Clean UK texts the same way we cleaned Indian ones."""
    text = re.sub(r"(https?://\S+|www\.\S+|\S+\.(com|co\.uk|net|org)\S*)", "[LINK]", text)
    text = re.sub(r"\d{5,}", "[NUMBER]", text)
    return text


def word_only_model():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def summarise(name, df, col):
    """One row of the results table."""
    row = {"method": name}
    for label, nice in [("scam", "scams caught"), ("normal", "false alarms"), ("promo", "promos flagged")]:
        part = df[df["label"] == label]
        row[nice] = f"{int(part[col].sum())}/{len(part)} ({part[col].mean():.0%})"
    return row


def main():
    uk = pd.read_csv(UCI_FILE, sep="\t", header=None, names=["label", "text"], quoting=3, encoding="utf-8")
    uk["y"] = (uk["label"] == "spam").astype(int)
    uk["text_clean"] = uk["text"].apply(add_placeholders)

    india = pd.read_csv(INDIA_FILE, encoding="utf-8")
    if "group" not in india.columns:
        india["group"] = [f"g{i}" for i in range(len(india))]
    india["y"] = (india["label"] == "scam").astype(int)
    print(f"UK messages: {len(uk)} | Indian messages: {len(india)} {india['label'].value_counts().to_dict()}")

    # A. Simple rule
    india["A_rule"] = india["text"].str.contains(r"\[LINK\]|\[NUMBER\]").astype(int)

    # B. Old UK model, unchanged
    india["B_old"] = joblib.load(OLD_MODEL).predict(india["text"])

    # C. UK only, with placeholder fix (no Indian data, so no cross-validation needed)
    c_model = word_only_model().fit(uk["text_clean"], uk["y"])
    india["C_placeholder"] = c_model.predict(india["text"])

    # D. UK + Indian data, letter patterns, 5-fold cross-validation
    india["D_new"] = 0
    india["D_prob"] = 0.0
    india["E_india_only"] = 0
    folds = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(folds.split(india, india["y"], india["group"]), 1):
        train_in = india.iloc[train_idx]
        texts = pd.concat([uk["text_clean"], train_in["text"]], ignore_index=True)
        y = np.concatenate([uk["y"].values, train_in["y"].values])
        w = np.concatenate([np.ones(len(uk)), np.full(len(train_in), INDIA_WEIGHT)])
        m = TwoViewModel().fit(texts, y, w)
        test_texts = india.iloc[test_idx]["text"]
        india.loc[india.index[test_idx], "D_new"] = m.predict(test_texts)
        india.loc[india.index[test_idx], "D_prob"] = m.scam_probability(test_texts)
        # E. Indian data only (no UK texts, no weighting needed)
        e = TwoViewModel().fit(train_in["text"], train_in["y"], np.ones(len(train_in)))
        india.loc[india.index[test_idx], "E_india_only"] = e.predict(test_texts)
        print(f"  fold {fold}: trained on {len(train_in)} Indian + {len(uk)} UK, tested on {len(test_idx)} unseen Indian")

    results = pd.DataFrame([
        summarise("A. Rule: has link or number", india, "A_rule"),
        summarise("B. Old UK model", india, "B_old"),
        summarise("C. UK + placeholder fix", india, "C_placeholder"),
        summarise("D. UK + Indian data + letters", india, "D_new"),
        summarise("E. Indian data only + letters", india, "E_india_only"),
    ])
    print("\n=== Results on Indian messages (each tested while unseen) ===")
    print(results.to_string(index=False))

    print("\n=== Scams caught, by language ===")
    scams = india[india["label"] == "scam"]
    by_lang = scams.groupby("language")[["A_rule", "B_old", "C_placeholder", "D_new", "E_india_only"]].agg(
        lambda s: f"{int(s.sum())}/{len(s)}")
    print(by_lang.to_string())

    print("\n=== Real-only scams (from family phones, not self-written) ===")
    real = india[india["source"].isin(["mine", "mom", "dad"])]
    print(pd.DataFrame([summarise(n, real, c) for n, c in [
        ("B. Old UK model", "B_old"), ("D. UK + Indian", "D_new"), ("E. Indian only", "E_india_only")]]).to_string(index=False))

    for col in ["D_new", "E_india_only"]:
        print(f"\n=== Where {col} still fails ===")
        for _, r in india[(india["label"] == "scam") & (india[col] == 0)].iterrows():
            print(f"[MISSED SCAM][{r['language']}][{r['source']}] {r['text'][:110]}")
        for _, r in india[(india["label"] == "normal") & (india[col] == 1)].iterrows():
            print(f"[FALSE ALARM][{r['language']}] {r['text'][:110]}")
    # Final model for the demo: method E (Indian data only), trained on ALL Indian messages.
    # We picked E because it had far fewer false alarms in the test above.
    final = TwoViewModel().fit(india["text"], india["y"], np.ones(len(india)))
    joblib.dump(final, os.path.join(HERE, "models", "india_model.joblib"))
    india.to_csv(os.path.join(HERE, "step3_predictions.csv"), index=False, encoding="utf-8")
    print("\nSaved demo model to models/india_model.joblib and every prediction to step3_predictions.csv")


if __name__ == "__main__":
    main()

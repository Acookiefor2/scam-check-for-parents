"""
Step 4: Final test on messages the model has NEVER seen.

fresh_test.csv holds 11 messages we put aside before training anything on them.
They were never used in step 3, never used to pick a model, never looked at
while tuning. This is the most honest test we have.

We train the final model (method E: Indian data only, word + letter patterns)
on every message in indian_test_set.csv, then test it ONCE on fresh_test.csv.
The old UK model from step 1 is tested on the same messages for comparison.

Note: the fresh messages are self-written (Hindi/Marathi ones translated with
a translation tool), not real messages from phones. Real ones would be better.

Run it:  python step4_final_test.py
"""

import os

import joblib
import numpy as np
import pandas as pd

from scam_model import TwoViewModel

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    train = pd.read_csv(os.path.join(HERE, "indian_test_set.csv"), encoding="utf-8")
    fresh = pd.read_csv(os.path.join(HERE, "fresh_test.csv"), encoding="utf-8")
    train["y"] = (train["label"] == "scam").astype(int)

    # Safety check: no fresh message may also be in the training data
    overlap = set(train["text"]) & set(fresh["text"])
    assert not overlap, f"{len(overlap)} fresh messages leaked into training!"

    model = TwoViewModel().fit(train["text"], train["y"], np.ones(len(train)))
    old = joblib.load(os.path.join(HERE, "models", "uci_model.joblib"))

    fresh["new_prob"] = model.scam_probability(fresh["text"])
    fresh["new_flag"] = (fresh["new_prob"] >= 0.5).astype(int)
    fresh["old_flag"] = old.predict(fresh["text"])

    print(f"Trained on {len(train)} messages. Testing once on {len(fresh)} never-seen messages.\n")
    for name, col in [("Old UK model", "old_flag"), ("New model (E)", "new_flag")]:
        s = fresh[fresh["label"] == "scam"]
        n = fresh[fresh["label"] == "normal"]
        print(f"{name:14s}: scams caught {int(s[col].sum())}/{len(s)} | false alarms {int(n[col].sum())}/{len(n)}")

    print("\nEvery fresh message (new model):")
    for _, r in fresh.iterrows():
        right = (r["new_flag"] == 1) == (r["label"] == "scam")
        mark = "OK   " if right else "WRONG"
        print(f"{mark} [{r['label']:6s}][{r['language']:8s}] scam prob {r['new_prob']:.0%} | {r['text'][:80]}")


if __name__ == "__main__":
    main()

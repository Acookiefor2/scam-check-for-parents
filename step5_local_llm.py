"""
Step 5: Model 2 - a local LLM (runs on this laptop with Ollama, nothing is sent to the cloud).

Why: model 1 (step 3/4) learned the LANGUAGE instead of the scam ("Hindi/Marathi = scam,
English = safe") and failed the never-seen test. A pretrained multilingual LLM is NOT
trained on our messages at all, so it can't learn that shortcut from our data.
It reads each message and answers SCAM, PROMO or NORMAL.

Fairness rules we follow:
  - The model (llama3.1:8b) and the prompt below were chosen BEFORE seeing any results.
  - We do not edit the prompt to fix mistakes. If we ever change it, we must report that
    and test on brand-new messages.
  - Because the LLM never trains on our data, every message we collected is a fair test.

Run it:  python step5_local_llm.py
(Ollama must be running. It takes a while on a laptop CPU; answers are saved to
 step5_llm_answers.csv, so if you stop it, running again continues where it left off.)

This project is a prototype and is not financial advice.
"""

import json
import os
import sys
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CACHE = os.path.join(HERE, "step5_llm_answers.csv")

PROMPT = """You help an elderly parent in India decide if a text message (SMS or WhatsApp) is dangerous.
The message may be in English, Hindi, Marathi or Hinglish.
For privacy, real links, phone numbers, names and codes were replaced with [LINK], [NUMBER], [NAME] and [CODE].

Classify the message as exactly one of:
SCAM   - tries to steal money or personal details (fake KYC/bank/electricity/parcel alerts, fake loan approvals, fake jobs, lottery prizes, stock tips, requests to send money or share a code)
PROMO  - a genuine advertisement or offer from a real business
NORMAL - a genuine personal, service, bank, government or transaction message (OTPs, bills, bookings, alerts)

Message:
\"\"\"{message}\"\"\"

Answer with one word only: SCAM, PROMO or NORMAL."""


def ask_llm(message):
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT.format(message=message),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 5},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def parse(answer):
    """Turn the LLM's reply into scam / promo / normal (or 'unclear')."""
    a = answer.strip().upper()
    for word in ["SCAM", "PROMO", "NORMAL"]:
        if a.startswith(word) or word in a.split():
            return word.lower()
    return "unclear"


def summarise(df, col):
    out = {}
    for label, nice in [("scam", "scams caught"), ("normal", "false alarms"), ("promo", "promos flagged")]:
        part = df[df["label"] == label]
        out[nice] = f"{int(part[col].sum())}/{len(part)} ({part[col].mean():.0%})" if len(part) else "-"
    return out


def main():
    train = pd.read_csv(os.path.join(HERE, "indian_test_set.csv"), encoding="utf-8")
    fresh = pd.read_csv(os.path.join(HERE, "fresh_test.csv"), encoding="utf-8")
    train["set"] = "main"
    fresh["set"] = "fresh"
    df = pd.concat([train, fresh], ignore_index=True)

    done = pd.read_csv(CACHE, encoding="utf-8") if os.path.exists(CACHE) else pd.DataFrame(columns=["text", "answer"])
    answers = dict(zip(done["text"], done["answer"]))

    todo = [t for t in df["text"] if t not in answers]
    print(f"Model: {MODEL} | {len(df)} messages | {len(todo)} still to ask")
    try:
        for i, text in enumerate(todo, 1):
            start = time.time()
            answers[text] = ask_llm(text)
            pd.DataFrame({"text": list(answers), "answer": list(answers.values())}).to_csv(CACHE, index=False, encoding="utf-8")
            print(f"  {i}/{len(todo)} ({time.time() - start:.0f}s) -> {parse(answers[text]):7s} | {text[:60]}")
    except urllib.error.URLError:
        sys.exit("Can't reach Ollama. Open the Ollama app (or run 'ollama serve') and try again.")

    df["llm_answer"] = df["text"].map(answers).fillna("")
    df["llm_label"] = df["llm_answer"].apply(parse)
    df["llm_flag"] = (df["llm_label"] == "scam").astype(int)

    print(f"\nUnclear answers: {(df['llm_label'] == 'unclear').sum()} (counted as 'not scam')")

    print("\n=== Local LLM on ALL messages (none were used to train it) ===")
    print(pd.DataFrame([{"messages": "all", **summarise(df, "llm_flag")}]).to_string(index=False))

    print("\n=== Never-seen fresh set (compare with step 4: old UK 3/5 & 0/6, model 1 0/5 & 5/6) ===")
    f = df[df["set"] == "fresh"]
    print(pd.DataFrame([{"messages": "fresh", **summarise(f, "llm_flag")}]).to_string(index=False))

    print("\n=== By language (scams caught | false alarms) ===")
    for lang, part in df.groupby("language"):
        s, n = part[part["label"] == "scam"], part[part["label"] == "normal"]
        print(f"{lang:9s} scams {int(s['llm_flag'].sum())}/{len(s)} | false alarms {int(n['llm_flag'].sum())}/{len(n)}")

    print("\n=== Real messages from family phones only ===")
    real = df[df["source"].isin(["mine", "mom", "dad"])]
    print(pd.DataFrame([{"messages": "real", **summarise(real, "llm_flag")}]).to_string(index=False))

    print("\n=== 3-way labels: how often the LLM's label matches ours ===")
    print(pd.crosstab(df["label"], df["llm_label"], rownames=["our label"], colnames=["LLM said"]).to_string())

    print("\n=== Where the LLM fails ===")
    for _, r in df[(df["label"] == "scam") & (df["llm_flag"] == 0)].iterrows():
        print(f"[MISSED SCAM][{r['language']}][{r['source']}] {r['text'][:100]}")
    for _, r in df[(df["label"] == "normal") & (df["llm_flag"] == 1)].iterrows():
        print(f"[FALSE ALARM][{r['language']}][{r['source']}] {r['text'][:100]}")

    df.to_csv(os.path.join(HERE, "step5_predictions.csv"), index=False, encoding="utf-8")
    print("\nSaved every answer to step5_predictions.csv")


if __name__ == "__main__":
    main()

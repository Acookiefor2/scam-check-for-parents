# Scam check for parents

**Track:** Finance

A tool that helps an older parent in India decide whether an SMS or WhatsApp message is a scam,
in English, Hindi, Marathi or Hinglish. It runs fully on a laptop: messages never leave the computer.

> This project is a prototype and is not financial advice.

*(Full write-up coming. This file currently explains how to run the code.)*

## How to run

Needs Python 3 and, for steps 5 and the app, [Ollama](https://ollama.com) with `llama3.1:8b`
(`ollama pull llama3.1:8b`).

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

python step1_baseline.py       # old UK spam model + "always not spam" baseline
python step2_test_on_india.py  # old model tested on our Indian messages
python step3_improve.py        # model 1: trained on our messages, 5-fold cross-validation
python step4_final_test.py     # model 1 tested once on never-seen messages (it fails)
python step5_local_llm.py      # model 2: local LLM, tested on every message (~10-20 min)
python app.py                  # demo app at http://127.0.0.1:7860
```

## Files

| File | What it is |
|---|---|
| `indian_test_set.csv` | Our messages: real ones from family phones (`mine`/`mom`/`dad`) and self-written ones (`self_written`, Hindi/Marathi ones `self_written_translated` with Google Translate). Links, numbers, names and codes replaced with `[LINK]`, `[NUMBER]`, `[NAME]`, `[CODE]`. |
| `fresh_test.csv` | 11 messages set aside and never used for training or choosing a model |
| `scam_model.py` | Model 1 (word + letter-pattern features, logistic regression) |
| `step5_llm_answers.csv` | Every raw answer the local LLM gave |

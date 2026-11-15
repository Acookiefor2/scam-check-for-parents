# Scam Check for Parents

**Track:** Finance
**Team:** Sankalp Mhaske

> This project is a prototype and is not financial advice.

## The problem

Older people in India get a constant stream of SMS and WhatsApp messages in English, Hindi, Marathi and Hinglish: OTPs, bank alerts, ads, loan offers and scams. In my family's phones, the real scams (fake stock tips, fake "loan approved" messages, "VIP number" sellers) look a lot like genuine messages. Even the humans and the big filters get it wrong: my mom had reported a genuine Amazon receipt and a bank maintenance notice as spam, and Google Messages put a real government (CERT-In) warning into my spam folder. Parents need a quick second opinion they can trust before they click, call or pay.

## What we built

A small app where you paste one message and get one of four answers: **Likely scam**, **Looks like an ad/offer**, **No strong scam signs**, or **Not sure**, plus a one-line reason in simple English and what to do (e.g. "don't click links, never share a code, ask family first"). It runs fully on a laptop with a local language model, so the message never leaves the computer.

Real examples from the demo: *"Hello, this is CEO of SBI… purchase some gift cards"* → **Likely scam** ("asks you to buy gift cards, a common trick"). *"Oh sorry, I thought this was Sarah. How have you been?"* → **No strong scam signs**, which is wrong: it is a known "wrong number" opener for investment scams.

## What we added

Started from suggested project **4.2.1 Scam text filter**. What we added:

- **Our own data:** 100 messages from today's India. 70 are real messages from my phone and my parents' phones; 30 I wrote myself (some Hindi/Marathi ones translated with Google Translate), plus 11 more self-written ones kept aside as a never-seen test set.
- **Four languages** (English, Hindi, Marathi, Hinglish) and a check of whether the model treats them fairly.
- **Three labels instead of two:** scam / promo / normal, because for a parent an ad is annoying but a scam costs money.
- **A specific user** (an elderly parent), explanations, and privacy: everything runs locally.

## How it works

**Data.** The UCI SMS Spam Collection (5,574 English messages from the 2010s, 13% spam) for the baseline. Our Indian set: 100 messages (31 scam, 33 promo, 36 normal; 68 English, 13 Hinglish, 10 Hindi, 9 Marathi). Links, phone numbers, names and codes were replaced with `[LINK]`, `[NUMBER]`, `[NAME]`, `[CODE]`, and two personal messages were removed for privacy. 11 extra messages were set aside before any model was chosen and never used for training.

**Models, in the order we tried them.**
1. **Old UK model:** TF-IDF + logistic regression trained on the UCI data (the suggested baseline project).
2. **Model 1:** TF-IDF word + letter-pattern features with logistic regression, trained on our Indian messages, tested with 5-fold cross-validation (translations of the same message kept in the same fold).
3. **Model 2 (final):** **Llama 3.1 8B**, run locally with Ollama, with one fixed prompt asking for SCAM / PROMO / NORMAL. It is not trained on our messages at all, so every message is a fair test. We picked the model and wrote the prompt before seeing any results, and did not change the prompt afterwards.

## Results

"Scam" is the positive class. Recall = share of scams caught; precision = share of warnings that were really scams.

| Method | Recall | Precision | Real family scams caught | Never-seen set (scams caught / false alarms) |
|---|---|---|---|---|
| Never flag anything | 0% | – | 0/8 | 0/5, 0/6 |
| Simple rule: "has a link or number" | 64% | 29% | – | – |
| Old UK model | 31% | 28% | 3/8 | 3/5, 0/6 |
| Model 1 (our data) | 48% (cross-val) | – | 2/8 | **0/5, 5/6** |
| **Model 2 (local LLM)** | **94% (34/36)** | **49%** | **8/8** | **5/5, 2/6** |

- The old UK model caught 93% of spam on its own UK test data, but only 31% of scams in our Indian messages, and 0 of 12 Hindi/Marathi scams.
- **Model 1 learned the language instead of the scam.** In our data most Hindi/Marathi messages were scams and most English ones were safe, so it learned "Devanagari = scam, English = safe". It looked fine in cross-validation, then on the never-seen set it missed all 5 English scams and flagged 5 of 6 genuine messages (every Hindi and Marathi one). It was also fragile: removing 2 messages for privacy dropped it from 5/8 to 2/8 on real scams. So we moved to model 2.
- **Model 2** caught every real scam from my family's phones and showed no script bias (0 of 5 genuine Hindi messages flagged).
- **Where model 2 fails:** it is too suspicious. It flagged **17 of 42 genuine messages (40%)**, mostly OTPs and verification codes, and called 19 of 33 genuine ads scams, so only about half of its warnings are right. It missed the Hindi and Marathi electricity-cutoff scams and the "wrong number" opener. We think it reacts to words like "code", "don't share" and "loan", and it judges one message at a time, so scams that only become dangerous later in a conversation slip through.
- **Limits:** only 8 of the 31 scams are real; the rest are self-written, which are cleaner than real ones. The never-seen set is only 11 messages. These are early results, not proof.

## What we'd do next

1. Test it with my parents on the messages they actually receive, and see if they trust it.
2. Cut the OTP false alarms using the sender ID (registered senders like `VM-PHONPE-S` vs random 10-digit numbers), and collect more real Hindi/Marathi messages, then test again on a fresh never-seen set.

## Sources

- SMS Spam Collection, UCI Machine Learning Repository (Almeida & Gómez Hidalgo): https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Llama 3.1 8B (Meta), run with Ollama: https://ollama.com/library/llama3.1
- scikit-learn: https://scikit-learn.org · pandas: https://pandas.pydata.org · Gradio: https://www.gradio.app
- Google Translate (for some self-written Hindi/Marathi messages): https://translate.google.com
- Claude (Anthropic), AI assistant used to write the code and draft this write-up: https://claude.ai
- Our own messages: collected from my and my parents' phones with their permission

---

## How to run

Needs Python 3 and, for step 5 and the app, [Ollama](https://ollama.com) with `llama3.1:8b` (`ollama pull llama3.1:8b`).

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

python step1_baseline.py       # old UK model + "always not spam" baseline
python step2_test_on_india.py  # old model tested on our Indian messages
python step3_improve.py        # model 1, 5-fold cross-validation
python step4_final_test.py     # model 1 tested once on never-seen messages
python step5_local_llm.py      # model 2 (local LLM) on every message, ~10-20 min
python app.py                  # demo app at http://127.0.0.1:7860
```

| File | What it is |
|---|---|
| `indian_test_set.csv` | Our 100 messages. `source`: `mine`/`mom`/`dad` = real; `self_written`; `self_written_translated` = translated with Google Translate |
| `fresh_test.csv` | 11 never-seen messages, never used for training or choosing a model |
| `scam_model.py` | Model 1 |
| `step5_llm_answers.csv`, `step5_predictions.csv` | Every raw answer and prediction from model 2 |

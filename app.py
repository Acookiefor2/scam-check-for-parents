"""
Demo app: paste a text message, get a verdict from the local LLM (model 2) plus a one-line reason.

Built for: an older parent who gets lots of SMS (OTPs, ads, loan offers, scams) and wants a quick
second opinion. Runs fully on this laptop with Ollama: the message is never sent to the cloud.

Run it:  python app.py   (Ollama must be running)
Then open the link it prints (usually http://127.0.0.1:7860) in your browser.

The verdict uses EXACTLY the same model and prompt as step5_local_llm.py, so the app behaves the
way our results say it does (including its known weakness: it often flags genuine OTPs).

This project is a prototype and is not financial advice.
"""

import json
import urllib.error
import urllib.request

import gradio as gr

from step5_local_llm import MODEL, OLLAMA_URL, ask_llm, parse

REASON_PROMPT = """A text message was classified as {label}.
Message:
\"\"\"{message}\"\"\"
In ONE short simple English sentence that an elderly person can understand, say why. No more than 25 words."""


def get_reason(message, label):
    body = json.dumps({
        "model": MODEL,
        "prompt": REASON_PROMPT.format(label=label.upper(), message=message),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 60},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def check(message):
    message = (message or "").strip()
    if not message:
        return "Paste a message first.", ""
    try:
        label = parse(ask_llm(message))
        reason = get_reason(message, label) if label != "unclear" else ""
    except urllib.error.URLError:
        return "Can't reach Ollama. Open the Ollama app and try again.", ""

    if label == "scam":
        verdict = "⚠️ LIKELY SCAM\nDon't click links, don't call numbers in it, don't pay anything, never share a code. Ask family first."
    elif label == "promo":
        verdict = "🛍️ LOOKS LIKE AN AD / OFFER\nProbably not dangerous by itself, but never pay an upfront fee to get a loan, prize or job."
    elif label == "normal":
        verdict = ("✅ NO STRONG SCAM SIGNS\nThis is not a guarantee. If it asks for money, a fee, a click, "
                   "an app download or a code, still ask family first.")
    else:
        verdict = "🤔 NOT SURE\nThe model couldn't decide. Check with someone before acting on it."
    return verdict, reason


demo = gr.Interface(
    fn=check,
    inputs=gr.Textbox(lines=5, label="Paste the SMS / WhatsApp message here (English, Hindi, Marathi or Hinglish)"),
    outputs=[gr.Textbox(label="Verdict"), gr.Textbox(label="Reason (written by the AI, may not be perfect)")],
    title="Scam check for parents",
    description=(
        f"Runs on this laptop with {MODEL} via Ollama: your message never leaves the computer. "
        "In our tests it caught 34 of 36 scams (including all 8 real ones from family phones), "
        "but it also wrongly flagged 40% of genuine messages, especially OTPs and verification codes. "
        "This project is a prototype and is not financial advice."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

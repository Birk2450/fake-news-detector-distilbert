import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/best"
MAX_LENGTH = 256
LABELS = ["Fake", "Real"]

_reuters_pat = re.compile(r'[\(\[]\s*Reuters\s*[\)\]]|^\s*Reuters\s*-\s*', re.IGNORECASE)

def clean_text(text: str) -> str:
    text = text or ""
    return _reuters_pat.sub("", text).strip()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

_TOKENIZER, _MODEL = None, None

def predict(text: str) -> dict:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER, _MODEL = load_model()

    text = clean_text(text)

    inputs = _TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = _MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()

    pred_id = int(torch.argmax(torch.tensor(probs)).item())
    return {
        "label": LABELS[pred_id],
        "prob_fake": float(probs[0]),
        "prob_real": float(probs[1]),
    }

import re
from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Paths / Config
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]   # repo_root/fake-news-detector-distilbert
MODEL_PATH = ROOT_DIR / "models" / "best"

MAX_LENGTH = 256
LABELS = ["Fake", "Real"]

# Remove Reuters tags (same idea as preprocessing)
_reuters_pat = re.compile(r'[\(\[]\s*Reuters\s*[\)\]]|^\s*Reuters\s*-\s*', re.IGNORECASE)

def clean_text(text: str) -> str:
    text = (text or "").strip()
    return _reuters_pat.sub("", text).strip()

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def categorize(prob_real: float) -> str:
    # 0 <= 0.35 Likely False
    # 0.35 < p < 0.65 Ambiguous
    # 0.65 <= 1 Likely True
    if prob_real <= 0.35:
        return "Likely to be False"
    if prob_real < 0.65:
        return "Ambiguous Content"
    return "Likely to be True"

def build_input_text(title: str, body: str, source: str = "", date_str: str = "") -> str:
    parts = []
    if title.strip():
        parts.append(f"TITLE: {title.strip()}")
    if body.strip():
        parts.append(f"BODY: {body.strip()}")
    if source.strip():
        parts.append(f"SOURCE: {source.strip()}")
    if date_str.strip():
        parts.append(f"DATE: {date_str.strip()}")
    return "\n".join(parts).strip()

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model folder not found: {MODEL_PATH}\n"
            f"Expected a Hugging Face saved model at: models/best/\n"
            f"Tip: ensure models/best contains config.json + model.safetensors (or pytorch_model.bin) + tokenizer files."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

    device = get_device()
    model.to(device)
    model.eval()

    return tokenizer, model, device

def predict(title: str, body: str, source: str = "", date_str: str = "") -> dict:
    tokenizer, model, device = load_model()

    text = build_input_text(title, body, source, date_str)
    text = clean_text(text)

    if not text:
        raise ValueError("Empty text after cleaning. Provide title/body.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]  # shape [2]

    prob_fake = float(probs[0].item())
    prob_real = float(probs[1].item())
    pred_id = int(torch.argmax(probs).item())
    label = LABELS[pred_id]

    confidence_pct = float(probs[pred_id].item()) * 100.0

    return {
        "title": title,
        "body": body,
        "source": source,
        "date_str": date_str,
        "input_text": text,
        "label": label,
        "confidence_pct": confidence_pct,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "category": categorize(prob_real),
        "model_path": str(MODEL_PATH),
        "device": str(device),
    }
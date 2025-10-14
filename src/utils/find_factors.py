from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
from pathlib import Path

# Load FinBERT for sentiment
tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
clf = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()

def finbert_score(text):
    """
    Calculate sentiment score for a given text using FinBERT.
    Returns positive score - negative score (range: -1 to 1).
    """
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = clf(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]  # [neg, neu, pos]
    return probs[2] - probs[0]  # positive - negative

# Split all FOMC texts into sentences
sentences = []
for file in Path("dataset/opening_statements").glob("*.txt"):
    text = Path(file).read_text(encoding="utf-8", errors="ignore")
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences += [s.strip() for s in sents if len(s.split()) >= 5]

scores = [(s, finbert_score(s)) for s in sentences]
scores_sorted = sorted(scores, key=lambda x: x[1])

hawk_sents = [s for s, sc in scores_sorted[:20]]
dove_sents = [s for s, sc in scores_sorted[-20:]]

print("\nTop Hawkish sentences:")
print("\n".join(hawk_sents[:5]))
print("\nTop Dovish sentences:")
print("\n".join(dove_sents[:5]))

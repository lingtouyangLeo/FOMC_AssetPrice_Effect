# -*- coding: utf-8 -*-
"""
Factor Similarity Analysis for FOMC Communications (No Regression Version)
---------------------------------------------------------------------------
This script:
1) Reads FOMC transcripts (.txt) with YYYYMMDD in filenames
2) Embeds sentences with FinBERT (mean pooling + L2 normalization)
3) Computes similarities to hawkish/dovish anchors
4) Produces HawkScore = sim(hawk) - sim(dove) (>0 = more hawkish)
5) Regression section is left empty for future use.

Usage:
    python src/factor_similarity_analysis.py

Input:
    - Text files in: dataset/opening_statements/*.txt

Output:
    - CSV file: output/factor_similarity_scores.csv
    - Columns: Date, filename, sim_to_hawk, sim_to_dove, hawk_score

Interpretation:
    - hawk_score > 0: More hawkish (tighter policy stance)
    - hawk_score < 0: More dovish (looser policy stance)
    - hawk_score ≈ 0: Neutral stance

Dependencies:
    pip install transformers torch pandas numpy
"""

import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")


class FactorSimilarityAnalyzer:
    """Compute hawkish/dovish factor similarity scores using FinBERT embeddings."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        """
        Load FinBERT and prepare anchors.
        """
        print(f"[Init] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        print(f"[Init] Using device: {self.device}")

        # Hawkish and dovish anchors (extendable) - using extreme, contrasting statements
        self.factors = {
            "hawk": [
                "Recent indicators suggest that real GDP growth has picked up this quarter, with consumption spending remaining strong.",
                "The labor force participation rate has moved up over the past year, particularly for individuals aged 25 to 54 years.",
                "Indicators of economic activity and employment have strengthened since the beginning of the year.",
                "The economy is showing continued strength, suggesting further policy firming may be appropriate.",
            ],
            "dove": [
                "This action has no implications for our intended stance of monetary policy.",
                "We will continue to make our decisions meeting by meeting, based on incoming data.",
                "As we have said, we will provide advance notice before making any changes to our purchases.",
                "We intend to wrap up the review by late summer.",
            ],
        }

        # Cache anchor embeddings and centroids
        self.factor_vecs = {k: self.embed_texts(v) for k, v in self.factors.items()}
        self.factor_centroid = {
            k: vecs.mean(axis=0, keepdims=True) for k, vecs in self.factor_vecs.items()
        }  # [1, H], L2-normalized

    # ---------- Embedding utilities ----------

    @staticmethod
    def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings using the attention mask.
        """
        token_embeddings = model_output.last_hidden_state  # [B, T, H]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(dim=1)
        counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # [B, H]

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """
        Batch-embed texts -> mean-pooled, L2-normalized sentence embeddings.
        """
        all_vecs: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                vecs = self._mean_pooling(out, inputs["attention_mask"])   # [B, H]
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)     # L2 normalize
            all_vecs.append(vecs.cpu())
        return torch.cat(all_vecs, dim=0).numpy()

    def embed_sentence(self, sentence: str) -> np.ndarray:
        """Convenience single-sentence embed."""
        return self.embed_texts([sentence])[0]

    # ---------- Scoring ----------

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split text into sentences. Keep sentences with >=5 tokens to reduce noise.
        """
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)
        sents = [s.strip() for s in parts if len(s.strip().split()) >= 5]
        return sents

    def score_document(self, text: str) -> dict:
        """
        Compute sim_to_hawk, sim_to_dove, and hawk_score for a document.
        """
        sents = self.split_sentences(text)
        if not sents:
            return {"sim_to_hawk": 0.0, "sim_to_dove": 0.0, "hawk_score": 0.0}

        doc_vecs = self.embed_texts(sents)                     # [N, H], normalized
        hawk_cent = self.factor_centroid["hawk"]               # [1, H]
        dove_cent = self.factor_centroid["dove"]               # [1, H]

        # Cosine similarities via dot product (vectors already normalized)
        sim_hawk = float((doc_vecs @ hawk_cent.T).mean())
        sim_dove = float((doc_vecs @ dove_cent.T).mean())
        return {"sim_to_hawk": sim_hawk, "sim_to_dove": sim_dove, "hawk_score": sim_hawk - sim_dove}

    # ---------- IO ----------

    @staticmethod
    def extract_date_from_stem(stem: str) -> str:
        """
        Extract YYYYMMDD from filename stem and convert to YYYY-MM-DD.
        """
        m = re.search(r"(\d{8})", stem)
        if not m:
            return stem
        ymd = m.group(1)
        try:
            return datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return stem

    def process_folder(self, docs_dir: Path, out_csv: Path) -> pd.DataFrame:
        """
        Compute scores for all .txt files in docs_dir and save CSV.
        """
        files = sorted(docs_dir.glob("*.txt"))
        if not files:
            print(f"[Warn] No .txt files in: {docs_dir}")
            return pd.DataFrame()

        rows: list[dict] = []
        print(f"[Run] Found {len(files)} transcripts.")
        for i, f in enumerate(files, 1):
            date_iso = self.extract_date_from_stem(f.stem)
            print(f"  [{i}/{len(files)}] {f.name} -> Date={date_iso}")
            try:
                txt = Path(f).read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"    [Error] Read failed: {e}")
                continue
            if not txt.strip():
                print("    [Warn] Empty file. Skipped.")
                continue

            sc = self.score_document(txt)
            rows.append({"Date": date_iso, "filename": f.name, **sc})

        df = pd.DataFrame(rows).sort_values("Date")
        if not df.empty:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"[OK] Scores saved -> {out_csv}")
            print("[Summary]\n", df.describe())
        return df


# ---------- Placeholder for regression section ----------

def run_hac_regressions(scores_csv: Path, market_csv: Path, y_cols: list[str]) -> None:
    """
    Placeholder for future regression implementation.
    This function intentionally left empty.
    """
    print("\n[Info] Regression section is not implemented in this version.\n")
    pass


# ---------- Main function ----------

def main():
    # ---------------- CONFIG ----------------
    DOCS_DIR = Path("dataset/opening_statements")  # folder with .txt transcripts
    OUT_DIR = Path("output")
    SCORES_CSV = OUT_DIR / "factor_similarity_scores.csv"

    MODEL_NAME = "ProsusAI/finbert"  # FinBERT encoder
    DEVICE = None                    # None (auto), or "cpu"/"cuda"

    # Future regression config (currently unused)
    MARKET_CSV = Path("data/market.csv")
    Y_COLS = ["DGS2_Change", "DGS1_Change", "DGS10_Change", "DXY_Change"]
    RUN_REG = False  # regression disabled
    # ----------------------------------------

    print("=" * 60)
    print("FOMC Factor Similarity (FinBERT) - No Regression Version")
    print("=" * 60)
    print(f"[Paths] DOCS_DIR = {DOCS_DIR}")
    print(f"[Paths] OUT_DIR  = {OUT_DIR}")

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"docs dir not found: {DOCS_DIR}")

    analyzer = FactorSimilarityAnalyzer(model_name=MODEL_NAME, device=DEVICE)
    _ = analyzer.process_folder(DOCS_DIR, SCORES_CSV)

    # regression placeholder
    if RUN_REG:
        run_hac_regressions(SCORES_CSV, MARKET_CSV, Y_COLS)
    else:
        print("\n[Info] Regression step skipped.\n")


if __name__ == "__main__":
    main()

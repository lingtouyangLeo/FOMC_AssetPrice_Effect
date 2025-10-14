# -*- coding: utf-8 -*-
"""
Semantic Similarity Analysis for FOMC Communications
-----------------------------------------------------
This script uses semantic similarity to analyze the hawkish/dovish stance of FOMC statements.

Methodology:
1) Extract sentences from FOMC opening statements
2) Encode sentences using FinBERT (financial domain pre-trained BERT model)
3) Create semantic embeddings via mean pooling + L2 normalization
4) Define hawkish and dovish anchor sentences as reference points
5) Compute cosine similarity between document embeddings and anchor centroids
6) Calculate hawk_score = similarity_to_hawk - similarity_to_dove

The semantic similarity approach captures nuanced policy stances by:
- Understanding context and meaning beyond keyword matching
- Measuring semantic distance in high-dimensional embedding space
- Using financial domain knowledge from FinBERT pre-training

Usage:
    python src/factor_similarity_analysis.py

Input:
    - Text files in: dataset/opening_statements/*.txt

Output:
    - CSV file: output/factor_similarity_scores.csv
    - Columns: Date, filename, path, sim_to_hawk, sim_to_dove, hawk_score

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
    """
    Semantic Similarity Analyzer for FOMC Policy Stance
    
    Uses FinBERT embeddings and cosine similarity to measure the semantic distance
    between FOMC statements and predefined hawkish/dovish anchor sentences.
    
    The semantic approach captures policy nuances beyond keyword counting by:
    - Encoding contextual meaning in high-dimensional space
    - Measuring angular distance (cosine similarity) between embeddings
    - Leveraging financial domain knowledge from FinBERT
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        """
        Initialize the semantic similarity analyzer.
        
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
            device: Computing device (cuda/cpu), auto-detected if None
        """
        print(f"[Init] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        print(f"[Init] Using device: {self.device}")

        # Semantic anchors: extreme hawkish and dovish reference points
        # These serve as poles in the semantic space for measuring policy stance
        self.factors = {
            "hawk": [
                "Inflation is dangerously high and spiraling out of control.",
                "We must aggressively raise rates by 75 basis points or more.",
                "Restrictive monetary policy is essential to crush inflation.",
                "Interest rates need to stay high for an extended period.",
                "The economy is overheating with excessive demand.",
                "Inflation poses severe risks to economic stability.",
                "We prioritize price stability over growth concerns.",
                "Rate hikes must continue until inflation falls substantially.",
            ],
            "dove": [
                "Inflation is declining rapidly toward our 2% target.",
                "We must cut interest rates immediately to prevent recession.",
                "Accommodative monetary policy is needed to support jobs.",
                "Interest rates should be lowered to stimulate growth.",
                "The economy is dangerously weak with rising unemployment.",
                "Unemployment and weak growth are our primary concerns.",
                "We prioritize maximum employment over inflation worries.",
                "Rate cuts are necessary to support struggling businesses.",
            ],
        }

        # Cache anchor embeddings and centroids in semantic space
        self.factor_vecs = {k: self.embed_texts(v) for k, v in self.factors.items()}
        self.factor_centroid = {
            k: vecs.mean(axis=0, keepdims=True) for k, vecs in self.factor_vecs.items()
        }  # [1, H], L2-normalized centroids represent semantic "average" of each stance

    # ---------- Semantic Embedding Utilities ----------

    @staticmethod
    def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool token embeddings using the attention mask.
        
        This creates sentence-level embeddings from token-level representations,
        weighted by the attention mask to ignore padding tokens.
        
        Args:
            model_output: BERT model output with last_hidden_state
            attention_mask: Binary mask indicating real vs padding tokens
            
        Returns:
            Mean-pooled sentence embeddings [B, H]
        """
        token_embeddings = model_output.last_hidden_state  # [B, T, H]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(dim=1)
        counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # [B, H]

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """
        Batch-embed texts into semantic vector space.
        
        Process:
        1. Tokenize text into BERT input format
        2. Pass through FinBERT to get contextualized token embeddings
        3. Mean-pool tokens to create sentence-level embeddings
        4. L2-normalize to unit length (enables cosine similarity via dot product)
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process simultaneously
            
        Returns:
            Numpy array of shape [N, H] with L2-normalized embeddings
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
        """Convenience method to embed a single sentence into semantic space."""
        return self.embed_texts([sentence])[0]

    # ---------- Semantic Similarity Scoring ----------

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split text into sentences for semantic analysis.
        
        Filters sentences with >=5 tokens to reduce noise from fragments.
        
        Args:
            text: Input document text
            
        Returns:
            List of sentence strings
        """
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)
        sents = [s.strip() for s in parts if len(s.strip().split()) >= 5]
        return sents

    def score_document(self, text: str) -> dict:
        """
        Compute semantic similarity scores for a document.
        
        Process:
        1. Split document into sentences
        2. Embed each sentence into semantic space using FinBERT
        3. Compute cosine similarity to hawkish and dovish centroids
        4. Average similarities across all sentences
        5. Calculate hawk_score as the difference (positive = hawkish, negative = dovish)
        
        Cosine similarity measures the angular distance between embeddings:
        - 1.0 = identical semantic meaning
        - 0.0 = orthogonal (unrelated)
        - -1.0 = opposite meaning
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary with keys:
            - sim_to_hawk: Average cosine similarity to hawkish anchors
            - sim_to_dove: Average cosine similarity to dovish anchors
            - hawk_score: Difference (hawk - dove), indicating overall stance
        """
        sents = self.split_sentences(text)
        if not sents:
            return {"sim_to_hawk": 0.0, "sim_to_dove": 0.0, "hawk_score": 0.0}

        doc_vecs = self.embed_texts(sents)                     # [N, H], normalized
        hawk_cent = self.factor_centroid["hawk"]               # [1, H]
        dove_cent = self.factor_centroid["dove"]               # [1, H]

        # Cosine similarities via dot product (vectors already L2-normalized)
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
            rows.append({"Date": date_iso, "filename": f.name, "path": str(f), **sc})

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

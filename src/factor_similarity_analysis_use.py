# -*- coding: utf-8 -*-
"""
Semantic Similarity Analysis using Universal Sentence Encoder (USE)
--------------------------------------------------------------------
This script uses Google's Universal Sentence Encoder for semantic similarity analysis
of FOMC communications to measure hawkish/dovish policy stance.

Universal Sentence Encoder (USE):
- Developed by Google Research (Cer et al., 2018)
- Pre-trained on diverse web data for general-purpose semantic tasks
- Efficient: Direct sentence-to-vector encoding (no token-level processing)
- Optimized for semantic similarity and transfer learning tasks
- 512-dimensional embeddings

Method:
1) Load Universal Sentence Encoder from TensorFlow Hub
2) Encode FOMC sentences into semantic embeddings (512-dim vectors)
3) Encode hawkish/dovish anchor sentences
4) Compute cosine similarity between document and anchor centroids
5) Calculate hawk_score = similarity_to_hawk - similarity_to_dove

Usage:
    python src/factor_similarity_analysis_use.py

Input:
    - Text files in: dataset/opening_statements/*.txt

Output:
    - CSV file: output/factor_similarity_scores_use.csv
    - Columns: Date, filename, path, sim_to_hawk, sim_to_dove, hawk_score

Interpretation:
    - hawk_score > 0: More hawkish (semantically closer to hawkish anchors)
    - hawk_score < 0: More dovish (semantically closer to dovish anchors)
    - hawk_score ≈ 0: Neutral stance

Dependencies:
    pip install tensorflow tensorflow-hub pandas numpy
"""

import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

warnings.filterwarnings("ignore")
# Suppress TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')


class USEFactorSimilarityAnalyzer:
    """
    Semantic similarity analyzer using Universal Sentence Encoder (USE).
    
    USE provides efficient, high-quality sentence embeddings optimized for
    semantic similarity tasks without requiring fine-tuning.
    """

    def __init__(self, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Load Universal Sentence Encoder from TensorFlow Hub.
        
        Args:
            model_url: TensorFlow Hub URL for USE model
                      Default: universal-sentence-encoder/4 (latest version)
        """
        print(f"[Init] Loading Universal Sentence Encoder from TensorFlow Hub...")
        print(f"[Init] Model URL: {model_url}")
        self.model = hub.load(model_url)
        print(f"[Init] ✓ USE model loaded successfully")

        # Hawkish and dovish semantic anchors
        # Using extreme, contrasting statements for better separation
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

        # Compute and cache anchor embeddings
        print("[Init] Computing anchor embeddings...")
        self.factor_embeddings = {k: self.embed_texts(v) for k, v in self.factors.items()}
        self.factor_centroid = {
            k: vecs.mean(axis=0, keepdims=True) for k, vecs in self.factor_embeddings.items()
        }
        # Normalize centroids for cosine similarity
        self.factor_centroid = {
            k: v / np.linalg.norm(v, axis=1, keepdims=True) 
            for k, v in self.factor_centroid.items()
        }
        
        # Calculate anchor separation
        hawk_cent = self.factor_centroid["hawk"].flatten()
        dove_cent = self.factor_centroid["dove"].flatten()
        separation = float(np.dot(hawk_cent, dove_cent))
        print(f"[Init] Anchor separation (cosine similarity): {separation:.4f}")
        print(f"[Init] ✓ Initialization complete")

    # ---------- Embedding Utilities ----------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into semantic embeddings using Universal Sentence Encoder.
        
        USE directly maps sentences to 512-dimensional embeddings optimized
        for semantic similarity tasks. The embeddings are already normalized.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of shape [N, 512] with sentence embeddings
        """
        if not texts:
            return np.array([])
        
        # USE processes sentences in batch efficiently
        embeddings = self.model(texts)
        
        # Convert to numpy and normalize for cosine similarity
        embeddings_np = embeddings.numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        
        return normalized_embeddings

    # ---------- Scoring ----------

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
        Compute semantic similarity scores for a document using USE.
        
        Process:
        1. Split document into sentences
        2. Encode each sentence using Universal Sentence Encoder
        3. Compute cosine similarity to hawkish and dovish centroids
        4. Average similarities across all sentences
        5. Calculate hawk_score = sim_to_hawk - sim_to_dove
        
        Cosine similarity measures semantic closeness:
        - 1.0 = semantically identical
        - 0.0 = semantically unrelated
        - -1.0 = semantically opposite
        
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

        # Encode sentences using USE
        doc_vecs = self.embed_texts(sents)  # [N, 512], normalized
        
        hawk_cent = self.factor_centroid["hawk"]  # [1, 512]
        dove_cent = self.factor_centroid["dove"]  # [1, 512]

        # Cosine similarities via dot product (vectors already normalized)
        sim_hawk = float((doc_vecs @ hawk_cent.T).mean())
        sim_dove = float((doc_vecs @ dove_cent.T).mean())
        
        return {
            "sim_to_hawk": sim_hawk,
            "sim_to_dove": sim_dove,
            "hawk_score": sim_hawk - sim_dove
        }

    # ---------- I/O ----------

    @staticmethod
    def extract_date_from_stem(stem: str) -> str:
        """
        Extract YYYYMMDD from filename stem and convert to YYYY-MM-DD.
        
        Args:
            stem: Filename stem (without extension)
            
        Returns:
            Formatted date string YYYY-MM-DD
        """
        m = re.search(r"(\d{8})", stem)
        if not m:
            return stem
        raw = m.group(1)
        try:
            dt = datetime.strptime(raw, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return stem

    def process_folder(self, docs_dir: Path, output_csv: Path) -> pd.DataFrame:
        """
        Process all .txt files in docs_dir and compute similarity scores.
        
        Args:
            docs_dir: Directory containing .txt files
            output_csv: Path to save the resulting CSV
            
        Returns:
            DataFrame with columns: Date, filename, path, sim_to_hawk, sim_to_dove, hawk_score
        """
        txt_files = sorted(docs_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files in {docs_dir}")

        print(f"\n[Run] Found {len(txt_files)} documents in {docs_dir}")
        print("[Run] Processing documents with Universal Sentence Encoder...\n")

        results = []
        for i, fpath in enumerate(txt_files, start=1):
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            scores = self.score_document(text)
            date_str = self.extract_date_from_stem(fpath.stem)

            row = {
                "Date": date_str,
                "filename": fpath.name,
                "path": str(fpath),
                "sim_to_hawk": scores["sim_to_hawk"],
                "sim_to_dove": scores["sim_to_dove"],
                "hawk_score": scores["hawk_score"],
            }
            results.append(row)

            print(f"[{i:2d}/{len(txt_files)}] {fpath.name:40s} -> "
                  f"Date={date_str} | "
                  f"Hawk={scores['sim_to_hawk']:+.6f} | "
                  f"Dove={scores['sim_to_dove']:+.6f} | "
                  f"Score={scores['hawk_score']:+.6f}")

        df = pd.DataFrame(results).sort_values("Date").reset_index(drop=True)
        
        # Create output directory if needed
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, float_format="%.6f")
        
        print(f"\n[OK] Results saved to: {output_csv}")
        print(f"[OK] Processed {len(df)} documents")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Hawk Score Range: [{df['hawk_score'].min():.6f}, {df['hawk_score'].max():.6f}]")
        print(f"Hawk Score Mean:  {df['hawk_score'].mean():.6f}")
        print(f"Hawk Score Std:   {df['hawk_score'].std():.6f}")
        print("\nMost Hawkish Statements:")
        print(df.nlargest(3, 'hawk_score')[['Date', 'filename', 'hawk_score']].to_string(index=False))
        print("\nMost Dovish Statements:")
        print(df.nsmallest(3, 'hawk_score')[['Date', 'filename', 'hawk_score']].to_string(index=False))
        
        return df


# ---------- Placeholder for future regression ----------

def run_hac_regressions(df: pd.DataFrame, market_csv: Path | None = None):
    """
    Placeholder for HAC regression of hawk_score on asset price changes.
    This can be extended later with market data (DGS2, DGS1, DGS10, DXY, etc.)
    """
    print("\n[Regression] Regression functionality not implemented yet.")
    print("[Regression] Placeholder for future HAC analysis with market data.")


# ---------- Main ----------

def main():
    """Main entry point."""
    print("=" * 60)
    print("Universal Sentence Encoder (USE) - Semantic Similarity Analysis")
    print("=" * 60)
    
    # Paths
    DOCS_DIR = Path("dataset/opening_statements")
    OUTPUT_CSV = Path("output/factor_similarity_scores_use.csv")
    
    print(f"[Paths] DOCS_DIR = {DOCS_DIR}")
    print(f"[Paths] OUTPUT_CSV = {OUTPUT_CSV}")
    
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"docs dir not found: {DOCS_DIR}")
    
    # Initialize analyzer with USE
    analyzer = USEFactorSimilarityAnalyzer()
    
    # Process all documents
    df = analyzer.process_folder(DOCS_DIR, OUTPUT_CSV)
    
    # Placeholder for regression
    run_hac_regressions(df)
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

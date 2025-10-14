# -*- coding: utf-8 -*-
"""
Five-Factor Semantic Similarity Analysis for FOMC Communications
-----------------------------------------------------------------
This script expands the analysis from 2 factors (hawk/dove) to 5 factors that
better capture different dimensions of monetary policy communications.

Five Factors:
1. Rate/Tightening (短端路径) - Interest rate direction signals
2. Inflation Upward Pressure (通胀) - Inflation concern level
3. Forward Guidance/Path Language (前瞻指引) - Future policy path signals
4. Balance Sheet/QT (期限溢价通道) - Quantitative tightening/easing
5. Growth/Labor Softening (增长/就业走弱) - Economic growth and labor market weakness

Method:
- Uses Universal Sentence Encoder (USE) for semantic embeddings
- Each factor has 4-8 anchor sentences representing strong signals
- Computes cosine similarity between document and factor centroids
- Outputs 5 separate scores for each dimension

Usage:
    python src/five_factor_analysis.py

Input:
    - Text files in: dataset/opening_statements/*.txt

Output:
    - CSV file: output/five_factor_scores.csv
    - Columns: Date, filename, rate_score, inf_score, guidance_score, qt_score, growth_soft_score

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
tf.get_logger().setLevel('ERROR')


class FiveFactorAnalyzer:
    """
    Five-factor semantic similarity analyzer using Universal Sentence Encoder.
    
    Captures five distinct dimensions of FOMC policy communications:
    1. Rate/Tightening signals
    2. Inflation pressure concerns
    3. Forward guidance on policy path
    4. Balance sheet operations (QT/QE)
    5. Growth/labor market softening
    """

    def __init__(self, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Load Universal Sentence Encoder and prepare five-factor anchors.
        
        Args:
            model_url: TensorFlow Hub URL for USE model
        """
        print(f"[Init] Loading Universal Sentence Encoder from TensorFlow Hub...")
        self.model = hub.load(model_url)
        print(f"[Init] ✓ USE model loaded successfully")

        # Define five factors with anchor sentences
        self.factors = {
            # Factor 1: Rate/Tightening (短端路径)
            # Higher scores = more rate hikes expected, tighter policy
            "rate_tightening": [
                "The Committee will raise the federal funds rate.",
                "We will maintain a restrictive monetary policy stance.",
                "Further tightening may be appropriate.",
                "Interest rates need to rise significantly.",
                "We must keep rates higher for longer.",
                "Additional rate increases are likely needed.",
                "Policy needs to be sufficiently restrictive.",
                "We are committed to bringing rates to appropriate levels.",
            ],
            
            # Factor 2: Inflation Upward Pressure (通胀)
            # Higher scores = more inflation concern
            "inflation_pressure": [
                "Inflation is elevated and remains above target.",
                "Upside risks to inflation have increased.",
                "Inflation remains unacceptably high.",
                "Price pressures continue to be elevated.",
                "We are highly attentive to inflation risks.",
                "Inflation has not shown signs of declining.",
                "Core inflation remains stubbornly high.",
                "Inflation expectations are at risk of becoming unanchored.",
            ],
            
            # Factor 3: Forward Guidance/Path Language (前瞻指引)
            # Higher scores = longer restrictive policy expected
            "forward_guidance": [
                "We expect policy to remain restrictive for some time.",
                "We do not expect it appropriate to cut rates until inflation is clearly declining.",
                "We will proceed carefully and assess the totality of data.",
                "Policy will need to stay restrictive until we are confident.",
                "It will take time for inflation to return to target.",
                "We are prepared to maintain this stance as long as needed.",
                "The path forward will depend on incoming data.",
                "We will not prematurely ease policy.",
            ],
            
            # Factor 4: Balance Sheet/QT (期限溢价通道)
            # Higher scores = more QT, balance sheet reduction
            "balance_sheet_qt": [
                "The Committee will continue to reduce its holdings of Treasury securities.",
                "The balance sheet will decline as securities mature.",
                "Runoff will continue at a significant pace.",
                "We are reducing our balance sheet holdings substantially.",
                "Quantitative tightening will proceed as planned.",
                "We will allow our holdings to decline further.",
                "The reduction in the balance sheet is proceeding smoothly.",
                "Our securities holdings will continue to shrink.",
            ],
            
            # Factor 5: Growth/Labor Softening (增长/就业走弱)
            # Higher scores = more concern about economic weakness
            "growth_softening": [
                "Economic activity has slowed.",
                "The labor market has cooled significantly.",
                "Job gains have moderated substantially.",
                "Risks to the outlook are tilted to the downside.",
                "Economic growth is below trend.",
                "Employment growth has slowed notably.",
                "There are signs of weakening in economic activity.",
                "The labor market is no longer overheated.",
            ],
        }

        # Compute and cache factor embeddings
        print("[Init] Computing factor anchor embeddings...")
        self.factor_embeddings = {}
        self.factor_centroids = {}
        
        for factor_name, sentences in self.factors.items():
            embeddings = self.embed_texts(sentences)
            self.factor_embeddings[factor_name] = embeddings
            centroid = embeddings.mean(axis=0, keepdims=True)
            # Normalize centroid
            self.factor_centroids[factor_name] = centroid / np.linalg.norm(centroid)
        
        # Calculate and display anchor separations
        print("\n[Init] Factor centroid separations (cosine similarity):")
        factor_names = list(self.factors.keys())
        for i, name1 in enumerate(factor_names):
            for name2 in factor_names[i+1:]:
                cent1 = self.factor_centroids[name1].flatten()
                cent2 = self.factor_centroids[name2].flatten()
                sim = float(np.dot(cent1, cent2))
                print(f"  {name1} <-> {name2}: {sim:.4f}")
        
        print(f"\n[Init] ✓ Initialization complete")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into semantic embeddings using Universal Sentence Encoder.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Numpy array of shape [N, 512] with normalized sentence embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model(texts)
        embeddings_np = embeddings.numpy()
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        
        return normalized_embeddings

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split text into sentences for semantic analysis.
        
        Args:
            text: Input document text
            
        Returns:
            List of sentence strings (filtered: >= 5 tokens)
        """
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return []
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)
        sents = [s.strip() for s in parts if len(s.strip().split()) >= 5]
        return sents

    def score_document(self, text: str) -> dict:
        """
        Compute five-factor scores for a document.
        
        Process:
        1. Split document into sentences
        2. Encode each sentence using USE
        3. Compute cosine similarity to each factor centroid
        4. Average similarities across all sentences for each factor
        
        Args:
            text: Input document text
            
        Returns:
            Dictionary with keys:
            - rate_score: Rate/tightening signal strength
            - inf_score: Inflation pressure level
            - guidance_score: Forward guidance restrictiveness
            - qt_score: Balance sheet reduction signal
            - growth_soft_score: Growth/labor softening concern
        """
        sents = self.split_sentences(text)
        if not sents:
            return {
                "rate_score": 0.0,
                "inf_score": 0.0,
                "guidance_score": 0.0,
                "qt_score": 0.0,
                "growth_soft_score": 0.0
            }

        # Encode document sentences
        doc_vecs = self.embed_texts(sents)  # [N, 512], normalized
        
        # Compute similarity to each factor
        scores = {}
        scores["rate_score"] = float((doc_vecs @ self.factor_centroids["rate_tightening"].T).mean())
        scores["inf_score"] = float((doc_vecs @ self.factor_centroids["inflation_pressure"].T).mean())
        scores["guidance_score"] = float((doc_vecs @ self.factor_centroids["forward_guidance"].T).mean())
        scores["qt_score"] = float((doc_vecs @ self.factor_centroids["balance_sheet_qt"].T).mean())
        scores["growth_soft_score"] = float((doc_vecs @ self.factor_centroids["growth_softening"].T).mean())
        
        return scores

    @staticmethod
    def extract_date_from_stem(stem: str) -> str:
        """
        Extract YYYYMMDD from filename stem and convert to YYYY-MM-DD.
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
        Process all .txt files in docs_dir and compute five-factor scores.
        
        Args:
            docs_dir: Directory containing .txt files
            output_csv: Path to save the resulting CSV
            
        Returns:
            DataFrame with columns: Date, filename, path, rate_score, inf_score, 
                                   guidance_score, qt_score, growth_soft_score
        """
        txt_files = sorted(docs_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files in {docs_dir}")

        print(f"\n[Run] Found {len(txt_files)} documents in {docs_dir}")
        print("[Run] Processing documents with five-factor analysis...\n")

        results = []
        for i, fpath in enumerate(txt_files, start=1):
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            scores = self.score_document(text)
            date_str = self.extract_date_from_stem(fpath.stem)

            row = {
                "Date": date_str,
                "filename": fpath.name,
                "path": str(fpath),
                "rate_score": scores["rate_score"],
                "inf_score": scores["inf_score"],
                "guidance_score": scores["guidance_score"],
                "qt_score": scores["qt_score"],
                "growth_soft_score": scores["growth_soft_score"],
            }
            results.append(row)

            print(f"[{i:2d}/{len(txt_files)}] {fpath.name:40s} -> Date={date_str}")
            print(f"           Rate={scores['rate_score']:+.4f} | Inf={scores['inf_score']:+.4f} | "
                  f"Guide={scores['guidance_score']:+.4f} | QT={scores['qt_score']:+.4f} | "
                  f"GrowthSoft={scores['growth_soft_score']:+.4f}")

        df = pd.DataFrame(results).sort_values("Date").reset_index(drop=True)
        
        # Create output directory if needed
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, float_format="%.6f")
        
        print(f"\n[OK] Results saved to: {output_csv}")
        print(f"[OK] Processed {len(df)} documents")
        
        # Print summary statistics for each factor
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS BY FACTOR")
        print("=" * 80)
        
        factor_cols = ["rate_score", "inf_score", "guidance_score", "qt_score", "growth_soft_score"]
        factor_labels = ["Rate/Tightening", "Inflation Pressure", "Forward Guidance", 
                        "Balance Sheet/QT", "Growth/Labor Softening"]
        
        for col, label in zip(factor_cols, factor_labels):
            print(f"\n{label}:")
            print(f"  Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
            print(f"  Mean:  {df[col].mean():.4f}")
            print(f"  Std:   {df[col].std():.4f}")
            
            # Top 3 highest scores
            top3 = df.nlargest(3, col)[['Date', col]]
            print(f"  Top 3: {', '.join([f'{row.Date}({row[col]:.3f})' for _, row in top3.iterrows()])}")
        
        # Correlation matrix
        print("\n" + "=" * 80)
        print("FACTOR CORRELATION MATRIX")
        print("=" * 80)
        corr_matrix = df[factor_cols].corr()
        print(corr_matrix.round(3))
        
        return df


def main():
    """Main entry point."""
    print("=" * 80)
    print("Five-Factor Semantic Similarity Analysis for FOMC Communications")
    print("=" * 80)
    
    # Paths
    DOCS_DIR = Path("dataset/opening_statements")
    OUTPUT_CSV = Path("output/five_factor_scores.csv")
    
    print(f"[Paths] DOCS_DIR = {DOCS_DIR}")
    print(f"[Paths] OUTPUT_CSV = {OUTPUT_CSV}")
    
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"docs dir not found: {DOCS_DIR}")
    
    # Initialize analyzer
    analyzer = FiveFactorAnalyzer()
    
    # Process all documents
    df = analyzer.process_folder(DOCS_DIR, OUTPUT_CSV)
    
    print("\n" + "=" * 80)
    print("✓ Five-factor analysis complete!")
    print("=" * 80)
    print("\nInterpretation Guide:")
    print("  rate_score:         Higher = More rate hike signals, tighter policy")
    print("  inf_score:          Higher = More inflation concern")
    print("  guidance_score:     Higher = Longer restrictive policy expected")
    print("  qt_score:           Higher = More QT/balance sheet reduction")
    print("  growth_soft_score:  Higher = More economic/labor weakness concern")


if __name__ == "__main__":
    main()

"""
Factor Similarity Analysis for FOMC Statements

This script implements the Factor Similarity method:
1. Convert document sentences to vectors using FinBERT
2. Convert factor key sentences to vectors using FinBERT
3. Calculate average cosine similarity between document and factor vectors
4. Compute scores for each document (date)
5. Regress scores on changes in asset prices

Key factors:
- Inflation will rise
- Interest rates will rise
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FactorSimilarityAnalyzer:
    """
    Analyzer for computing factor similarity scores from FOMC statements.
    """
    
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize the analyzer with FinBERT model.
        
        Args:
            model_name (str): Name of the FinBERT model to use
        """
        print(f"Loading FinBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Define factor key sentences
        self.factors = {
            'inflation': "Inflation will rise",
            'interest_rates': "Interest rates will rise"
        }
    
    def encode_sentence(self, sentence):
        """
        Convert a sentence to a vector using FinBERT.
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            np.ndarray: Sentence embedding vector
        """
        # Tokenize and encode
        inputs = self.tokenizer(sentence, return_tensors='pt', 
                                padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings[0]
    
    def split_into_sentences(self, text):
        """
        Split text into sentences.
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        # Simple sentence splitting by period, exclamation, or question mark
        sentences = re.split(r'[.!?]+', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def compute_document_factor_score(self, document_text, factor_name):
        """
        Compute the average cosine similarity between document sentences and a factor.
        
        Args:
            document_text (str): Full document text
            factor_name (str): Name of the factor
            
        Returns:
            float: Average cosine similarity score
        """
        # Split document into sentences
        doc_sentences = self.split_into_sentences(document_text)
        
        if not doc_sentences:
            return 0.0
        
        # Get factor key sentence
        factor_sentence = self.factors[factor_name]
        
        # Encode factor sentence
        factor_vector = self.encode_sentence(factor_sentence)
        
        # Encode all document sentences and compute cosine similarity
        similarities = []
        for sentence in doc_sentences:
            if len(sentence.split()) < 3:  # Skip very short sentences
                continue
            
            try:
                sentence_vector = self.encode_sentence(sentence)
                # Compute cosine similarity
                sim = cosine_similarity(
                    sentence_vector.reshape(1, -1),
                    factor_vector.reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
            except Exception as e:
                print(f"  Warning: Could not encode sentence: {sentence[:50]}... Error: {e}")
                continue
        
        # Return average similarity
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
    
    def process_all_documents(self, input_folder, output_folder):
        """
        Process all documents and compute factor similarity scores.
        
        Args:
            input_folder (str): Path to folder containing text files
            output_folder (str): Path to save output CSV
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all text files
        text_files = sorted(list(input_path.glob("*.txt")))
        
        if not text_files:
            print(f"No text files found in {input_folder}")
            return
        
        print(f"\nFound {len(text_files)} documents to process")
        print("=" * 60)
        
        results = []
        
        # Process each document
        for i, text_file in enumerate(text_files, 1):
            # Extract date from filename (e.g., FOMCpresconf20201105.txt -> 2020-11-05)
            filename = text_file.stem
            date_str = filename.replace('FOMCpresconf', '')
            
            try:
                # Parse date: YYYYMMDD -> YYYY-MM-DD
                date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            except:
                print(f"Warning: Could not parse date from {filename}, using filename")
                date = date_str
            
            print(f"\n[{i}/{len(text_files)}] Processing: {text_file.name}")
            print(f"  Date: {date}")
            
            # Read document
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    document_text = f.read()
            except Exception as e:
                print(f"  Error reading file: {e}")
                continue
            
            if not document_text.strip():
                print(f"  Warning: Empty document, skipping")
                continue
            
            # Compute scores for each factor
            scores = {'Date': date}
            
            for factor_name in self.factors.keys():
                print(f"  Computing {factor_name} score...", end=' ')
                score = self.compute_document_factor_score(document_text, factor_name)
                scores[f'{factor_name}_score'] = score
                print(f"{score:.6f}")
            
            results.append(scores)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('Date')
        
        # Save to CSV
        output_file = output_path / 'factor_similarity_scores.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 60)
        print(f"✓ Processing complete!")
        print(f"✓ Results saved to: {output_file}")
        print(f"\nSummary Statistics:")
        print(df.describe())
        
        return df


def main():
    """
    Main function to run the factor similarity analysis.
    """
    # Define paths
    script_dir = Path(__file__).parent.parent
    input_folder = script_dir / "dataset" / "opening_statements"
    output_folder = script_dir / "output"
    
    print("=" * 60)
    print("FOMC Factor Similarity Analysis")
    print("=" * 60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    # Check if input folder exists
    if not input_folder.exists():
        print(f"\nError: Input folder does not exist: {input_folder}")
        print("Please run extract_opening_statement.py first to generate the text files.")
        return
    
    # Initialize analyzer
    try:
        analyzer = FactorSimilarityAnalyzer()
    except Exception as e:
        print(f"\nError loading FinBERT model: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers torch pandas numpy scikit-learn")
        return
    
    # Process all documents
    try:
        df = analyzer.process_all_documents(input_folder, output_folder)
        
        # Display first few rows
        print("\nFirst 10 rows of results:")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

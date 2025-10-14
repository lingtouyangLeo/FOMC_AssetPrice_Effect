# -*- coding: utf-8 -*-
"""
Compare FinBERT vs Universal Sentence Encoder Results
------------------------------------------------------
This script compares the hawk_score results from both methods
to understand their similarities and differences.

Usage:
    python src/compare_methods.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_results():
    """Load results from both methods."""
    finbert_csv = Path("output/factor_similarity_scores.csv")
    use_csv = Path("output/factor_similarity_scores_use.csv")
    
    if not finbert_csv.exists():
        print(f"⚠️  FinBERT results not found: {finbert_csv}")
        print("   Run: python src/factor_similarity_analysis.py")
        df_finbert = None
    else:
        df_finbert = pd.read_csv(finbert_csv)
        print(f"✓ Loaded FinBERT results: {len(df_finbert)} documents")
    
    if not use_csv.exists():
        print(f"⚠️  USE results not found: {use_csv}")
        print("   Run: python src/factor_similarity_analysis_use.py")
        df_use = None
    else:
        df_use = pd.read_csv(use_csv)
        print(f"✓ Loaded USE results: {len(df_use)} documents")
    
    return df_finbert, df_use

def compare_methods(df_finbert, df_use):
    """Compare the two methods."""
    if df_finbert is None or df_use is None:
        print("\n⚠️  Cannot compare - missing results from one or both methods.")
        return
    
    # Merge on Date
    merged = pd.merge(
        df_finbert[['Date', 'hawk_score']],
        df_use[['Date', 'hawk_score']],
        on='Date',
        suffixes=('_finbert', '_use')
    )
    
    print("\n" + "=" * 70)
    print("COMPARISON: FinBERT vs Universal Sentence Encoder")
    print("=" * 70)
    
    # Correlation
    correlation = merged['hawk_score_finbert'].corr(merged['hawk_score_use'])
    print(f"\nPearson Correlation: {correlation:.4f}")
    
    # Summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    
    stats = pd.DataFrame({
        'FinBERT': merged['hawk_score_finbert'].describe(),
        'USE': merged['hawk_score_use'].describe()
    })
    print(stats.round(6))
    
    # Agreement on direction
    finbert_positive = (merged['hawk_score_finbert'] > 0).sum()
    use_positive = (merged['hawk_score_use'] > 0).sum()
    both_positive = ((merged['hawk_score_finbert'] > 0) & (merged['hawk_score_use'] > 0)).sum()
    both_negative = ((merged['hawk_score_finbert'] < 0) & (merged['hawk_score_use'] < 0)).sum()
    
    print("\n" + "-" * 70)
    print("Direction Agreement:")
    print("-" * 70)
    print(f"FinBERT hawkish (>0):  {finbert_positive}/{len(merged)}")
    print(f"USE hawkish (>0):      {use_positive}/{len(merged)}")
    print(f"Both hawkish:          {both_positive}/{len(merged)}")
    print(f"Both dovish:           {both_negative}/{len(merged)}")
    print(f"Agreement rate:        {(both_positive + both_negative)/len(merged)*100:.1f}%")
    
    # Top 5 most hawkish/dovish by each method
    print("\n" + "-" * 70)
    print("Top 5 Most Hawkish - FinBERT:")
    print("-" * 70)
    top_hawk_finbert = merged.nlargest(5, 'hawk_score_finbert')[['Date', 'hawk_score_finbert', 'hawk_score_use']]
    print(top_hawk_finbert.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("Top 5 Most Hawkish - USE:")
    print("-" * 70)
    top_hawk_use = merged.nlargest(5, 'hawk_score_use')[['Date', 'hawk_score_finbert', 'hawk_score_use']]
    print(top_hawk_use.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("Top 5 Most Dovish - FinBERT:")
    print("-" * 70)
    top_dove_finbert = merged.nsmallest(5, 'hawk_score_finbert')[['Date', 'hawk_score_finbert', 'hawk_score_use']]
    print(top_dove_finbert.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("Top 5 Most Dovish - USE:")
    print("-" * 70)
    top_dove_use = merged.nsmallest(5, 'hawk_score_use')[['Date', 'hawk_score_finbert', 'hawk_score_use']]
    print(top_dove_use.to_string(index=False))
    
    # Largest disagreements
    merged['disagreement'] = abs(merged['hawk_score_finbert'] - merged['hawk_score_use'])
    print("\n" + "-" * 70)
    print("Largest Disagreements:")
    print("-" * 70)
    disagreements = merged.nlargest(5, 'disagreement')[['Date', 'hawk_score_finbert', 'hawk_score_use', 'disagreement']]
    print(disagreements.to_string(index=False))
    
    # Create visualization
    create_comparison_plot(merged)
    
    return merged

def create_comparison_plot(merged):
    """Create a scatter plot comparing the two methods."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(merged['hawk_score_finbert'], merged['hawk_score_use'], alpha=0.6, s=50)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(merged['hawk_score_finbert'].min(), merged['hawk_score_use'].min())
    max_val = max(merged['hawk_score_finbert'].max(), merged['hawk_score_use'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect Agreement')
    
    plt.xlabel('FinBERT Hawk Score', fontsize=11)
    plt.ylabel('USE Hawk Score', fontsize=11)
    plt.title('FinBERT vs Universal Sentence Encoder\nHawk Score Comparison', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Time series
    plt.subplot(1, 2, 2)
    merged_sorted = merged.sort_values('Date')
    x_indices = range(len(merged_sorted))
    
    plt.plot(x_indices, merged_sorted['hawk_score_finbert'], 'o-', label='FinBERT', alpha=0.7, linewidth=1.5)
    plt.plot(x_indices, merged_sorted['hawk_score_use'], 's-', label='USE', alpha=0.7, linewidth=1.5)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.xlabel('Document (chronological)', fontsize=11)
    plt.ylabel('Hawk Score', fontsize=11)
    plt.title('Hawk Score Over Time\nFinBERT vs USE', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "method_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_file}")
    plt.close()

def main():
    """Main entry point."""
    print("=" * 70)
    print("COMPARING SEMANTIC SIMILARITY METHODS")
    print("=" * 70)
    print()
    
    df_finbert, df_use = load_results()
    
    if df_finbert is not None and df_use is not None:
        merged = compare_methods(df_finbert, df_use)
        print("\n" + "=" * 70)
        print("✓ Comparison complete!")
        print("=" * 70)
    else:
        print("\n❌ Cannot perform comparison - missing data.")
        print("   Please run both analysis scripts first.")

if __name__ == "__main__":
    main()

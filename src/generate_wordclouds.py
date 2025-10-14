# -*- coding: utf-8 -*-
"""
Word Cloud Generator for FOMC Opening Statements
-------------------------------------------------
This script generates word clouds for each FOMC opening statement to visualize
the most frequently used terms and their relative importance.

Usage:
    python src/generate_wordclouds.py

Output:
    - Individual word clouds: output/wordclouds/wordcloud_YYYY-MM-DD.png
    - Period comparison clouds: output/wordclouds/periods/wordcloud_period_*.png

Dependencies:
    pip install wordcloud matplotlib pillow
"""

from pathlib import Path
from datetime import datetime
import re

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class FOMCWordCloudGenerator:
    """Generate word clouds for FOMC opening statements."""
    
    def __init__(self):
        """Initialize the word cloud generator with custom stopwords."""
        # Standard stopwords
        self.stopwords = set(STOPWORDS)
        
        # Add FOMC-specific common words to stopwords
        fomc_stopwords = {
            'will', 'percent', 'committee', 'fomc', 'federal', 'reserve',
            'year', 'month', 'powell', 'chair', 'said', 'says', 'noted',
            'meeting', 'policy', 'statement', 'press', 'conference',
            'us', 'u', 's', 'one', 'two', 'also', 'well', 'much', 'many',
            'first', 'second', 'third', 'however', 'moreover', 'furthermore',
        }
        self.stopwords.update(fomc_stopwords)
        
        # Word cloud settings
        self.wordcloud_config = {
            'width': 1200,
            'height': 600,
            'background_color': 'white',
            'colormap': 'RdYlGn_r',  # Red (hawkish) to Green (dovish)
            'max_words': 100,
            'relative_scaling': 0.5,
            'min_font_size': 10,
        }
    
    @staticmethod
    def extract_date_from_stem(stem: str) -> str:
        """Extract YYYYMMDD from filename and convert to YYYY-MM-DD."""
        m = re.search(r"(\d{8})", stem)
        if not m:
            return stem
        ymd = m.group(1)
        try:
            return datetime.strptime(ymd, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return stem
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for word cloud generation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers and percentages
        text = re.sub(r'\d+\.?\d*\s*percent', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters but keep spaces and hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_wordcloud(self, text: str, title: str, output_path: Path) -> None:
        """Generate and save a word cloud image."""
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        if not clean_text or len(clean_text.split()) < 10:
            print(f"  [Warn] Insufficient text for word cloud: {title}")
            return
        
        # Generate word cloud
        try:
            wordcloud = WordCloud(
                stopwords=self.stopwords,
                **self.wordcloud_config
            ).generate(clean_text)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'FOMC Opening Statement Word Cloud\n{title}', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  [OK] Saved: {output_path.name}")
            
        except Exception as e:
            print(f"  [Error] Failed to generate word cloud: {e}")
    
    def generate_all_wordclouds(self, docs_dir: Path, output_dir: Path) -> None:
        """Generate word clouds for all opening statements."""
        files = sorted(docs_dir.glob("*.txt"))
        
        if not files:
            print(f"[Warn] No .txt files in: {docs_dir}")
            return
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Run] Found {len(files)} opening statements.")
        print(f"[Run] Generating word clouds...")
        
        for i, f in enumerate(files, 1):
            date_iso = self.extract_date_from_stem(f.stem)
            print(f"\n[{i}/{len(files)}] {f.name} -> Date={date_iso}")
            
            # Read document
            try:
                text = f.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"  [Error] Read failed: {e}")
                continue
            
            if not text.strip():
                print(f"  [Warn] Empty file. Skipped.")
                continue
            
            # Generate output filename
            output_filename = f"wordcloud_{date_iso}.png"
            output_path = output_dir / output_filename
            
            # Generate word cloud
            self.generate_wordcloud(text, date_iso, output_path)
        
        print(f"\n[OK] All word clouds saved to: {output_dir}")
    
    def generate_comparative_wordcloud(self, docs_dir: Path, output_dir: Path,
                                      date_ranges: dict[str, tuple[str, str]]) -> None:
        """
        Generate comparative word clouds for different time periods.
        
        Args:
            docs_dir: Directory containing text files
            output_dir: Directory to save output images
            date_ranges: Dict mapping period names to (start_date, end_date) tuples
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted(docs_dir.glob("*.txt"))
        
        for period_name, (start_date, end_date) in date_ranges.items():
            print(f"\n[Run] Generating word cloud for period: {period_name} ({start_date} to {end_date})")
            
            # Collect all text from the period
            period_texts = []
            
            for f in files:
                date_iso = self.extract_date_from_stem(f.stem)
                
                if start_date <= date_iso <= end_date:
                    try:
                        text = f.read_text(encoding='utf-8', errors='ignore')
                        if text.strip():
                            period_texts.append(text)
                            print(f"  [+] Included: {f.name}")
                    except Exception as e:
                        print(f"  [Error] Could not read {f.name}: {e}")
            
            if not period_texts:
                print(f"  [Warn] No documents found for period {period_name}")
                continue
            
            # Combine all texts
            combined_text = " ".join(period_texts)
            
            # Generate word cloud
            title = f"{period_name}\n({start_date} to {end_date})\n{len(period_texts)} statements"
            output_filename = f"wordcloud_period_{period_name.replace(' ', '_')}.png"
            output_path = output_dir / output_filename
            
            self.generate_wordcloud(combined_text, title, output_path)


def main():
    # ---------------- CONFIG ----------------
    DOCS_DIR = Path("dataset/opening_statements")
    OUTPUT_DIR = Path("output/wordclouds")
    OUTPUT_PERIODS_DIR = Path("output/wordclouds/periods")
    
    # Define time periods for comparative analysis
    PERIODS = {
        "2020-2021_Pandemic": ("2020-01-01", "2021-12-31"),
        "2022_Rising_Inflation": ("2022-01-01", "2022-12-31"),
        "2023_Peak_Rates": ("2023-01-01", "2023-12-31"),
        "2024_Dovish_Turn": ("2024-01-01", "2024-12-31"),
        "2025_Recent": ("2025-01-01", "2025-12-31"),
    }
    # ----------------------------------------
    
    print("=" * 60)
    print("FOMC Word Cloud Generator")
    print("=" * 60)
    print(f"[Paths] DOCS_DIR = {DOCS_DIR}")
    print(f"[Paths] OUTPUT_DIR = {OUTPUT_DIR}")
    
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"docs dir not found: {DOCS_DIR}")
    
    generator = FOMCWordCloudGenerator()
    
    # Generate individual word clouds
    print("\n" + "=" * 60)
    print("Generating individual word clouds...")
    print("=" * 60)
    generator.generate_all_wordclouds(DOCS_DIR, OUTPUT_DIR)
    
    # Generate comparative word clouds by period
    print("\n" + "=" * 60)
    print("Generating period comparison word clouds...")
    print("=" * 60)
    generator.generate_comparative_wordcloud(DOCS_DIR, OUTPUT_PERIODS_DIR, PERIODS)
    
    print("\n" + "=" * 60)
    print("✓ All word clouds generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# FOMC Asset Price Effect Analysis 🏦

**Semantic analysis of Federal Reserve communications using NLP to measure monetary policy stance.**

This project applies FinBERT and Universal Sentence Encoder to analyze FOMC press conference transcripts, extracting policy signals across 5 dimensions for asset price prediction.

## Quick Start

```bash
# 1. Install dependencies
pip install tensorflow tensorflow-hub pandas numpy matplotlib scikit-learn

# 2. Extract opening statements from PDFs
python src/utils/extract_opening_statement.py

# 3. Run five-factor analysis (⭐ RECOMMENDED)
python src/five_factor_analysis.py

# 4. Visualize results
python src/utils/visualize_five_factors.py
```

**Main Output**: `output/five_factor_scores_with_hawkindex.csv` + visualization plots

## Project Structure

```
FOMC_AssetPrice_Effect/
├── src/                                        # Source code
│   ├── five_factor_analysis.py                 # ⭐ 5-factor + Hawkishness Index
│   ├── factor_similarity_analysis.py           # 2-factor analysis (FinBERT)
│   ├── factor_similarity_analysis_use.py       # 2-factor analysis (USE)
│   ├── MarketData_events.ipynb                 # 📊 Market data & event study analysis
│   └── utils/                                  # Utility scripts
│       ├── extract_opening_statement.py        # Extract opening statements from PDFs
│       ├── visualize_five_factors.py           # Visualize five-factor analysis results
│       ├── generate_wordclouds.py              # Generate word cloud visualizations
│       ├── compare_methods.py                  # Compare FinBERT vs USE results
│       └── find_factors.py                     # Find hawkish/dovish sentences using FinBERT
├── dataset/                                    # Data files
│   ├── transcripts/                            # Original PDF transcripts
│   ├── opening_statements/                     # Extracted opening statements (TXT files, 40 docs)
│   ├── events.csv                              # FOMC event dates
│   ├── event_windows_0_1.csv                   # Event window data
│   ├── wrds_fomc_markets_daily.csv             # Market data from WRDS
│   └── five_factor_regression_summary.csv      # Regression results
├── output/                                     # Analysis outputs
│   ├── five_factor_scores_with_hawkindex.csv   # ⭐ Main output (14 columns)
│   ├── hawkishness_index_trend.png             # Overall index time series
│   ├── five_factor_timeseries.png              # 5 factors over time
│   ├── five_factor_correlation.png             # Factor correlation heatmap
│   ├── five_factor_radar.png                   # Recent vs historical comparison
│   ├── method_comparison.png                   # FinBERT vs USE comparison
│   ├── factor_similarity_scores.csv            # 2-factor scores (FinBERT)
│   ├── factor_similarity_scores_use.csv        # 2-factor scores (USE)
│   └── wordclouds/                             # Word cloud images
├── FOMC_Communications_Report.pdf              # 📄 Project report
├── .gitignore                                  # Git ignore rules
├── LICENSE                                     # MIT License
└── README.md
```


## Setup

### Requirements

Before running the scripts, you need to install the required Python packages:

**For PDF extraction:**
```bash
pip install PyPDF2
```

**For factor similarity analysis (FinBERT):**
```bash
pip install transformers torch pandas numpy scikit-learn
```

**For factor similarity analysis (Universal Sentence Encoder):**
```bash
pip install tensorflow tensorflow-hub pandas numpy
```

**For word cloud visualization:**
```bash
pip install wordcloud matplotlib
```

Or install all at once (FinBERT version):
```bash
pip install PyPDF2 transformers torch pandas numpy scikit-learn wordcloud matplotlib
```

Or install all at once (USE version):
```bash
pip install PyPDF2 tensorflow tensorflow-hub pandas numpy wordcloud matplotlib
```

## Usage

### 1. Extract Opening Statements from PDF Transcripts

This script processes FOMC press conference PDF files and extracts Chair Powell's opening statements.

**Features:**
- Extracts only the opening statement (before the first "Thank you")
- Removes page numbers, headers, and date information
- Fixes incorrectly split words from PDF extraction
- Connects sentences across lines and pages
- Saves as clean text files

**Run the script:**

```bash
python src/utils/extract_opening_statement.py
```

The script will:
- Read all PDF files from `dataset/transcripts/`
- Extract the opening statements
- Save the results to `dataset/opening_statements/`

**Input:** PDF files in `dataset/transcripts/`  
**Output:** TXT files in `dataset/opening_statements/`

### 2. Factor Similarity Analysis

This script implements the Factor Similarity method to analyze FOMC statements using hawkish and dovish sentiment anchors:

**Method:**
1. Convert document sentences to vectors using FinBERT (ProsusAI/finbert)
2. Convert hawkish and dovish anchor sentences to vectors
3. Calculate cosine similarity between document and anchor centroids
4. Compute hawk_score = similarity_to_hawk - similarity_to_dove

**Anchor Factors:**
- **Hawkish Anchors:** Statements indicating tighter monetary policy (e.g., "Inflation remains dangerously high", "We must aggressively raise interest rates")
- **Dovish Anchors:** Statements indicating looser monetary policy (e.g., "Inflation is declining rapidly", "We need to prevent a recession")

**Interpretation:**
- **Positive hawk_score:** More hawkish (tighter policy stance)
- **Negative hawk_score:** More dovish (looser policy stance)
- **Score near 0:** Neutral stance

**Run the script:**

```bash
python src/factor_similarity_analysis.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Use FinBERT to encode sentences into normalized embeddings
- Calculate similarity to hawkish and dovish anchors
- Save results to `output/factor_similarity_scores.csv`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** CSV file in `output/factor_similarity_scores.csv`

**Output Format:**
```
Date,filename,path,sim_to_hawk,sim_to_dove,hawk_score
2020-11-05,FOMCpresconf20201105.txt,dataset/opening_statements/FOMCpresconf20201105.txt,0.5234,0.5456,-0.0222
2020-12-16,FOMCpresconf20201216.txt,dataset/opening_statements/FOMCpresconf20201216.txt,0.5189,0.5378,-0.0189
...
```

**Note:** The first run will download the FinBERT model (~440MB), which may take some time depending on your internet connection.

### 2b. Semantic Similarity Analysis using Universal Sentence Encoder (USE) [Alternative]

**Alternative implementation** using Google's Universal Sentence Encoder instead of FinBERT.

**What is Universal Sentence Encoder (USE)?**
- Developed by Google Research (Cer et al., 2018)
- Pre-trained on diverse web data for general-purpose semantic similarity tasks
- More efficient than BERT-based models (direct sentence-to-vector encoding)
- Produces 512-dimensional sentence embeddings
- Optimized specifically for semantic similarity and transfer learning

**Advantages of USE:**
- ✅ Faster inference (no token-level processing required)
- ✅ Simpler architecture (direct sentence encoding)
- ✅ Strong performance on semantic similarity benchmarks
- ✅ Pre-trained on broader domain (not finance-specific)

**Run the USE version:**

```bash
python src/factor_similarity_analysis_use.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Use Universal Sentence Encoder to encode sentences (512-dim embeddings)
- Calculate cosine similarity to hawkish and dovish anchors
- Save results to `output/factor_similarity_scores_use.csv`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** CSV file in `output/factor_similarity_scores_use.csv`

**Note:** The first run will download the USE model (~1GB) from TensorFlow Hub.

**Comparison: FinBERT vs USE**
| Feature | FinBERT | Universal Sentence Encoder |
|---------|---------|---------------------------|
| Domain | Finance-specific | General-purpose |
| Model Size | ~440MB | ~1GB |
| Embedding Dim | 768 | 512 |
| Speed | Slower (token-level) | Faster (sentence-level) |
| Best For | Financial text analysis | General semantic similarity |

### 2c. Compare Methods

Compare the results from FinBERT and Universal Sentence Encoder:

```bash
python src/compare_methods.py
```

This script will:
- Load results from both methods
- Calculate correlation between hawk_scores
- Show agreement on hawkish/dovish direction
- Identify largest disagreements
- Generate comparison visualizations

**Output:** 
- Console: Statistical comparison and correlation analysis
- File: `output/method_comparison.png` (scatter plot and time series)

### 2d. Five-Factor Analysis [Extended]

**Extended analysis** that expands from 2 factors (hawk/dove) to 5 granular factors capturing different dimensions of monetary policy.

**Five Factors:**

1. **Rate/Tightening** - Interest rate direction signals
   - Example anchors: "The Committee will raise the federal funds rate", "We will maintain a restrictive monetary policy stance"
   
2. **Inflation Upward Pressure** - Inflation concern level
   - Example anchors: "Inflation is elevated and remains above target", "Upside risks to inflation have increased"
   
3. **Forward Guidance/Path Language** - Future policy path signals
   - Example anchors: "We expect policy to remain restrictive for some time", "We do not expect it appropriate to cut until..."
   
4. **Balance Sheet/QT** - Quantitative tightening/easing
   - Example anchors: "The Committee will continue to reduce its holdings", "The balance sheet will decline"
   
5. **Growth/Labor Softening** - Economic growth and labor market weakness
   - Example anchors: "Economic activity has slowed", "The labor market has cooled"

**Why Five Factors?**
- More granular analysis than binary hawk/dove
- Captures different policy transmission channels
- Better for asset price impact analysis (yield curve effects differ by factor)
- Aligns with how markets interpret Fed communications

**Run the script:**

```bash
python src/five_factor_analysis.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Use Universal Sentence Encoder to encode sentences
- Compute cosine similarity to each of the 5 factor anchors
- Generate 5 independent scores per document
- Save results to `output/five_factor_scores.csv`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** CSV file in `output/five_factor_scores.csv`

**Output Format:**
```csv
Date,filename,path,rate_score,inf_score,guidance_score,qt_score,growth_soft_score
2020-11-05,FOMCpresconf20201105.txt,...,0.0862,0.1223,0.1167,0.1289,0.1023
```

**Interpretation:**
- **rate_score**: Higher = More rate hike signals, tighter policy
- **inf_score**: Higher = More inflation concern
- **guidance_score**: Higher = Longer restrictive policy expected
- **qt_score**: Higher = More QT/balance sheet reduction
- **growth_soft_score**: Higher = More economic/labor weakness concern

**Factor Correlations:**
The factors are designed to be somewhat independent but naturally show some correlation:
- Rate & Inflation: 0.925 (high correlation - both indicate tightening)
- Forward Guidance & Growth Softening: 0.500 (moderate - balancing act)
- Balance Sheet/QT & Growth Softening: 0.712 (balanced policy consideration)

### 3. Generate Word Clouds

This script generates word cloud visualizations for each FOMC opening statement and creates comparative word clouds by time period:

**Features:**
- Individual word clouds for all opening statements
- Period comparison word clouds (2020-2021 Pandemic, 2022 Rising Inflation, 2023 Peak Rates, 2024 Dovish Turn, 2025 Recent)
- Custom FOMC-specific stopword filtering
- Red-Yellow-Green colormap reflecting sentiment

**Run the script:**

```bash
python src/utils/generate_wordclouds.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Generate individual word clouds for each statement
- Generate comparative word clouds for different time periods
- Save images to `output/wordclouds/` and `output/wordclouds/periods/`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** PNG files in `output/wordclouds/` (individual) and `output/wordclouds/periods/` (comparisons)

## Analysis Pipeline

You can choose between different semantic similarity methods and levels of granularity:

**Option A: Using FinBERT (finance-specific, 2-factor)**
```bash
# Step 1: Extract opening statements from PDFs
python src/utils/extract_opening_statement.py

# Step 2: Compute factor similarity scores with FinBERT
python src/factor_similarity_analysis.py

# Step 3: Generate word cloud visualizations
python src/utils/generate_wordclouds.py
```

**Option B: Using Universal Sentence Encoder (general-purpose, 2-factor)**
```bash
# Step 1: Extract opening statements from PDFs
python src/utils/extract_opening_statement.py

# Step 2: Compute factor similarity scores with USE
python src/factor_similarity_analysis_use.py

# Step 3: Generate word cloud visualizations
python src/utils/generate_wordclouds.py
```

**Option C: Five-Factor Extended Analysis (5 dimensions)**
```bash
# Step 1: Extract opening statements from PDFs
python src/utils/extract_opening_statement.py

# Step 2: Compute five-factor scores
python src/five_factor_analysis.py

# Step 3: Generate word cloud visualizations
python src/utils/generate_wordclouds.py

# Step 4 (Optional): Visualize five-factor results
python src/utils/visualize_five_factors.py

# Step 5 (Optional): Compare 2-factor methods
python src/compare_methods.py
```

## Compare both methods:
```bash
python src/compare_methods.py
```
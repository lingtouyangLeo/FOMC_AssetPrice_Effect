# FOMC_AssetPrice_Effect

This project analyzes the effect of FOMC (Federal Open Market Committee) press conferences on asset prices.

## Project Structure

```
FOMC_AssetPrice_Effect/
├── src/                          # Source code
│   ├── extract_opening_statement.py  # Extract opening statements from PDF transcripts
│   └── factor_similarity_analysis.py # Compute factor similarity scores
├── dataset/                      # Data files
│   ├── transcripts/              # Original PDF transcripts
│   └── opening_statements/       # Extracted opening statements (TXT files)
├── output/                       # Analysis outputs
│   └── factor_similarity_scores.csv  # Factor similarity scores for each date
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
python src/extract_opening_statement.py
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

### 3. Generate Word Clouds

This script generates word cloud visualizations for each FOMC opening statement and creates comparative word clouds by time period:

**Features:**
- Individual word clouds for all opening statements
- Period comparison word clouds (2020-2021 Pandemic, 2022 Rising Inflation, 2023 Peak Rates, 2024 Dovish Turn, 2025 Recent)
- Custom FOMC-specific stopword filtering
- Red-Yellow-Green colormap reflecting sentiment

**Run the script:**

```bash
python src/generate_wordclouds.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Generate individual word clouds for each statement
- Generate comparative word clouds for different time periods
- Save images to `output/wordclouds/` and `output/wordclouds/periods/`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** PNG files in `output/wordclouds/` (individual) and `output/wordclouds/periods/` (comparisons)

## Analysis Pipeline

You can choose between two semantic similarity methods:

**Option A: Using FinBERT (finance-specific)**
```bash
# Step 1: Extract opening statements from PDFs
python src/extract_opening_statement.py

# Step 2: Compute factor similarity scores with FinBERT
python src/factor_similarity_analysis.py

# Step 3: Generate word cloud visualizations
python src/generate_wordclouds.py
```

**Option B: Using Universal Sentence Encoder (general-purpose)**
```bash
# Step 1: Extract opening statements from PDFs
python src/extract_opening_statement.py

# Step 2: Compute factor similarity scores with USE
python src/factor_similarity_analysis_use.py

# Step 3: Generate word cloud visualizations
python src/generate_wordclouds.py
```

## Compare both methods:
```bash
python src/compare_methods.py
```
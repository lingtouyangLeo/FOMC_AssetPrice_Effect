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

**For factor similarity analysis:**
```bash
pip install transformers torch pandas numpy scikit-learn
```

Or install all at once:
```bash
pip install PyPDF2 transformers torch pandas numpy scikit-learn
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

This script implements the Factor Similarity method to analyze FOMC statements:

**Method:**
1. Convert document sentences to vectors using FinBERT
2. Convert factor key sentences to vectors using FinBERT
3. Calculate average cosine similarity between document and factor vectors
4. Compute scores for each document (date)

**Key Factors:**
- **Inflation:** "Inflation will rise"
- **Interest Rates:** "Interest rates will rise"

**Run the script:**

```bash
python src/factor_similarity_analysis.py
```

The script will:
- Read all text files from `dataset/opening_statements/`
- Use FinBERT to encode sentences into vectors
- Calculate cosine similarity scores for each factor
- Save results to `output/factor_similarity_scores.csv`

**Input:** TXT files in `dataset/opening_statements/`  
**Output:** CSV file in `output/factor_similarity_scores.csv`

**Output Format:**
```
Date,inflation_score,interest_rates_score
2020-11-05,0.496361,0.500582
2020-12-16,0.503241,0.50943
...
```

**Note:** The first run will download the FinBERT model (~440MB), which may take some time depending on your internet connection.

## Analysis Pipeline

Run the complete analysis pipeline in order:

```bash
# Step 1: Extract opening statements from PDFs
python src/extract_opening_statement.py

# Step 2: Compute factor similarity scores
python src/factor_similarity_analysis.py
```

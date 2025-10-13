# FOMC_AssetPrice_Effect

This project analyzes the effect of FOMC (Federal Open Market Committee) press conferences on asset prices.

## Project Structure

```
FOMC_AssetPrice_Effect/
├── src/                          # Source code
│   └── extract_opening_statement.py  # Extract opening statements from PDF transcripts
├── dataset/                      # Data files
│   ├── transcripts/              # Original PDF transcripts
│   └── opening_statements/       # Extracted opening statements (TXT files)
├── output/                       # Analysis outputs
└── README.md
```

## Setup

### Requirements

Before running the scripts, you need to install the required Python packages:

```bash
pip install PyPDF2
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

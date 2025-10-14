"""
Extract Opening Statement from FOMC PDF files and save as text files.

This script processes PDF files containing FOMC minutes/transcripts,
extracts only the opening statement section, and saves them as .txt files.

Usage:
    python src/extract_opening_statement.py

Input:
    - PDF files in: dataset/transcripts/*.pdf

Output:
    - Text files in: dataset/opening_statements/*.txt

Dependencies:
    pip install PyPDF2
"""

import os
import re
from pathlib import Path
import PyPDF2


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def extract_opening_statement(text):
    """
    Extract Chair Powell's opening statement only (before first "thank you").
    
    This function:
    1. Removes header information (date, page numbers, transcript title)
    2. Extracts only CHAIR POWELL's statements
    3. Stops at the first "thank you" (end of opening statement)
    
    Args:
        text (str): Full text extracted from PDF
        
    Returns:
        str: Chair Powell's opening statement text
    """
    # Remove header information (first few lines with date, page, transcript info)
    # Pattern to match header lines
    header_patterns = [
        r'^.*?Chair Powell\'s Press Conference.*?$',
        r'^.*?Page \d+ of \d+.*?$',
        r'^.*?Transcript of.*?$',
        r'^\s*\d{4}\s*$',  # Year line
        r'^[A-Z][a-z]+ \d+,?\s*\d{4}.*?$',  # Date lines
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_header = True
    
    for line in lines:
        # Skip header lines at the beginning
        if skip_header:
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_header = True
                    break
            
            # Look for "CHAIR POWELL" to mark start of actual content
            if 'CHAIR POWELL' in line.upper():
                skip_header = False
                cleaned_lines.append(line)
            elif not is_header and line.strip():
                skip_header = False
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Find where CHAIR POWELL starts speaking (first occurrence)
    # Look for the pattern at the beginning of the actual speech
    chair_powell_match = re.search(r'CHAIR POWELL\s*\.', text, re.IGNORECASE)
    if chair_powell_match:
        start_pos = chair_powell_match.start()
    else:
        # If not found, look for just "Good afternoon" as backup
        good_afternoon = re.search(r'\bGood afternoon\b', text, re.IGNORECASE)
        start_pos = good_afternoon.start() if good_afternoon else 0
    
    # Find the first "thank you" which marks the end of opening statement
    # Look for "Thank you" or "thank you" that appears after the opening
    # Search from at least 500 characters after the start to skip any header "thank you"
    search_start = min(start_pos + 500, len(text))
    thank_you_pattern = r'\.\s*Thank you\b'
    thank_you_match = re.search(thank_you_pattern, text[search_start:], re.IGNORECASE)
    
    if thank_you_match:
        # End right before "thank you" (after the period)
        end_pos = search_start + thank_you_match.start() + 1  # Include the period before "Thank you"
    else:
        # If no "thank you" found, try to find question section markers
        end_patterns = [
            r'MICHELLE SMITH\.',
            r'\n[A-Z][A-Z\s]+:\s',  # Any person speaking (NAME:)
        ]
        end_pos = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text[start_pos + 100:], re.IGNORECASE)
            if match:
                end_pos = start_pos + 100 + match.start()
                break
    
    # Extract only CHAIR POWELL's statements
    opening_statement = text[start_pos:end_pos].strip()
    
    # Extract only Powell's words (remove the "CHAIR POWELL." label at start)
    if opening_statement.upper().startswith('CHAIR POWELL'):
        # Remove "CHAIR POWELL." or "CHAIR POWELL ."
        opening_statement = re.sub(r'^CHAIR POWELL\s*\.\s*', '', opening_statement, flags=re.IGNORECASE).strip()
    
    return opening_statement


def clean_text(text):
    """
    Clean the extracted text by removing excessive whitespace and formatting issues.
    Connect sentences across lines and pages.
    Fix words that are incorrectly split with spaces.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text with connected sentences
    """
    # Remove page information lines (e.g., "Page 2 of 24", "November 5, 2020 Chair Powell's Press Conference FINAL")
    text = re.sub(r'^.*?Page \d+ of \d+.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*?Chair Powell\'s Press Conference.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^.*?FINAL\s*$', '', text, flags=re.MULTILINE)
    
    # Remove date headers that appear in the middle
    text = re.sub(r'^[A-Z][a-z]+ \d+,?\s*\d{4}\s+Chair.*?$', '', text, flags=re.MULTILINE)
    
    # Remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # First remove all line breaks and extra whitespace to make text continuous
    text = re.sub(r'\s+', ' ', text)
    
    # Fix incorrectly split words (e.g., "a fternoon", "w ant", "c olleagues")
    # Match single letter + space + 2+ letters that form part of a word
    # This will fix splits at any position
    text = re.sub(r'\b([a-z])\s+([a-z]{2,})\b', r'\1\2', text)
    
    # Add line breaks after sentence-ending punctuation for readability
    # But keep sentences connected (no empty lines)
    text = re.sub(r'([.!?])\s+', r'\1 ', text)
    
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def process_pdf_folder(input_folder, output_folder):
    """
    Process all PDF files in a folder and extract opening statements.
    
    Args:
        input_folder (str): Path to folder containing PDF files
        output_folder (str): Path to folder where .txt files will be saved
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_file)
        
        if not full_text:
            print(f"  Warning: No text extracted from {pdf_file.name}")
            continue
        
        # Extract opening statement
        opening_statement = extract_opening_statement(full_text)
        
        # Clean the text
        cleaned_text = clean_text(opening_statement)
        
        # Save to .txt file
        output_filename = pdf_file.stem + ".txt"
        output_file = output_path / output_filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"  Saved: {output_filename} ({len(cleaned_text)} characters)")
        except Exception as e:
            print(f"  Error saving {output_filename}: {e}")
    
    print(f"\n✓ Processing complete! Output saved to {output_folder}")


def main():
    """
    Main function to run the extraction process.
    """
    # Define input and output folders
    # Adjust these paths according to your project structure
    script_dir = Path(__file__).parent.parent
    input_folder = script_dir / "dataset" / "transcripts"
    output_folder = script_dir / "dataset" / "opening_statements"
    
    print("=" * 60)
    print("FOMC Opening Statement Extractor")
    print("=" * 60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 60)
    
    # Check if input folder exists
    if not input_folder.exists():
        print(f"\nError: Input folder does not exist: {input_folder}")
        print("Please create the folder and add PDF files to process.")
        return
    
    # Process all PDFs
    process_pdf_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()

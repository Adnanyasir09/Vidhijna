# merge_pdfs.py

import os
from PyPDF2 import PdfMerger

# ----------------------------
# Project root helper
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def normalize_path(path: str) -> str:
    """Ensure the path is absolute and relative to PROJECT_ROOT if needed."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path

# ----------------------------
# Paths from environment variables or defaults
# ----------------------------
COMMERCIAL_CASES_DIR = normalize_path(
    os.getenv("COMMERCIAL_CASES_DIR", os.path.join(PROJECT_ROOT, "commercial_cases"))
)
OUTPUT_FILE = os.getenv("MERGED_CASES_PDF", "merged_output_cases.pdf")
OUTPUT_PATH = os.path.join(COMMERCIAL_CASES_DIR, OUTPUT_FILE)

# ----------------------------
# Merge function
# ----------------------------
def merge_pdfs_in_directory(directory: str, output_path: str) -> str | None:
    """
    Merges all PDF files in a specified directory into a single PDF.

    Args:
        directory (str): The directory containing the PDF files.
        output_path (str): Full path to the output merged PDF file.

    Returns:
        str: Path to the merged PDF file, or None if failed.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return None

    merger = PdfMerger()

    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Error: No PDF files found in the directory.")
        return None

    # Sort files numerically if filenames contain numbers
    pdf_files.sort(
        key=lambda f: (
            int("".join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else f
        )
    )

    # Merge PDFs
    for pdf in pdf_files:
        pdf_path = os.path.join(directory, pdf)
        try:
            merger.append(pdf_path)
            print(f"Added: {pdf_path}")
        except Exception as e:
            print(f"Error adding {pdf_path}: {e}")

    # Write merged PDF to output file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merger.write(output_path)
        merger.close()
        print(f"Merged PDF saved as: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving merged PDF: {e}")
        return None

# ----------------------------
# Run merger
# ----------------------------
merge_pdfs_in_directory(COMMERCIAL_CASES_DIR, OUTPUT_PATH)

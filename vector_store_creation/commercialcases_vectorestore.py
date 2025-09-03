# create_cases_vectorstore.py

import os
import math
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# ----------------------------
# Helper: Project root and path normalization
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def normalize_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path

# ----------------------------
# Paths (can also use env variables)
# ----------------------------
COMMERCIAL_LAWS_PDF = normalize_path(
    os.getenv("COMMERCIAL_LAWS_PDF", os.path.join(PROJECT_ROOT, "commercial_cases", "merged_output_cases.pdf"))
)
CASES_FAISS_PATH = normalize_path(
    os.getenv("CASES_FAISS_PATH", os.path.join(PROJECT_ROOT, "cases_index"))
)

# Ensure directories exist
os.makedirs(os.path.dirname(COMMERCIAL_LAWS_PDF), exist_ok=True)
os.makedirs(CASES_FAISS_PATH, exist_ok=True)

# ----------------------------
# Load the PDF
# ----------------------------
loader = PDFPlumberLoader(COMMERCIAL_LAWS_PDF)
print("LOADER INITIALISED")

docs = loader.load()
if not docs:
    print("Error: No pages loaded from the PDF. Please check the file path and format.")
    exit()

print(f"Total pages loaded: {len(docs)}")

# ----------------------------
# Split documents into chunks
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
if not documents:
    print("Error: Document splitting failed. No text chunks were generated.")
    exit()

print(f"Total text chunks created: {len(documents)}")

# ----------------------------
# Initialize Ollama embeddings
# ----------------------------
embeddings = OllamaEmbeddings(model="all-minilm:33m")

# ----------------------------
# Batch-wise embedding generation
# ----------------------------
batch_size = 100
vectorstore_db = None
total_batches = math.ceil(len(documents) / batch_size)
print("Generating embeddings batch-wise...")

for i in range(0, len(documents), batch_size):
    batch = documents[i: i + batch_size]
    if not batch:
        continue
    try:
        if vectorstore_db is None:
            vectorstore_db = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore_db.add_documents(batch)
    except Exception as e:
        print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
        continue
    print(f"Processed batch {i // batch_size + 1} of {total_batches}")

# ----------------------------
# Save FAISS vector store
# ----------------------------
if vectorstore_db:
    vectorstore_db.save_local(CASES_FAISS_PATH)
    print(f"Vector store saved to '{CASES_FAISS_PATH}'")
else:
    print("Error: No documents were processed. FAISS vector store was not created.")

# create_laws_vectorstore.py

import os
import math
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# ----------------------------
# Project root and path normalization
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def normalize_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path

# ----------------------------
# Paths (from env or defaults)
# ----------------------------
COMMERCIAL_LAWS_PDF = normalize_path(
    os.getenv("COMMERCIAL_LAWS_PDF", os.path.join(PROJECT_ROOT, "commercial_laws", "merged_output_laws.pdf"))
)
LAWS_FAISS_PATH = normalize_path(
    os.getenv("LAWS_FAISS_PATH", os.path.join(PROJECT_ROOT, "commercial_laws_index"))
)

# Ensure directories exist
os.makedirs(os.path.dirname(COMMERCIAL_LAWS_PDF), exist_ok=True)
os.makedirs(LAWS_FAISS_PATH, exist_ok=True)

# ----------------------------
# Load PDF
# ----------------------------
loader = PDFPlumberLoader(COMMERCIAL_LAWS_PDF)
print("LOADER INITIALISED")
docs = loader.load()
if not docs:
    print("Error: No pages loaded from the PDF.")
    exit()
print(f"Total pages loaded: {len(docs)}")

# ----------------------------
# Custom Chapter and Section Splitter
# ----------------------------
def chapter_section_splitter(docs):
    """
    Splits the document into chapters and sections based on headings.
    Assumes chapters and sections are marked by "Chapter X" or "Section Y".
    """
    chapters = []
    current_chapter = []
    for doc in docs:
        text = doc.page_content
        if "Chapter" in text and current_chapter:
            chapters.append("\n".join(current_chapter))
            current_chapter = []
        current_chapter.append(text)
    if current_chapter:
        chapters.append("\n".join(current_chapter))
    return chapters

print("Splitting into chapters and sections...")
chapters = chapter_section_splitter(docs)
print(f"Total chapters detected: {len(chapters)}")

# ----------------------------
# Hierarchical splitting: chapters → sections → chunks
# ----------------------------
def hierarchical_splitter(chapters, chunk_size=1000, chunk_overlap=200):
    split_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for chapter in chapters:
        sections = text_splitter.split_text(chapter)
        split_documents.extend(sections)
    return split_documents

print("Performing hierarchical splitting...")
split_documents = hierarchical_splitter(chapters)
print(f"Total text chunks after hierarchical splitting: {len(split_documents)}")

# Convert to LangChain Documents
documents = [Document(page_content=text) for text in split_documents]

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
    vectorstore_db.save_local(LAWS_FAISS_PATH)
    print(f"Vector store saved to '{LAWS_FAISS_PATH}'")
else:
    print("Error: No documents were processed. FAISS vector store was not created.")

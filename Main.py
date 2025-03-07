import os
import faiss
import ollama
import pickle
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ====== CONFIG ======
DOCS_FOLDER = "./docs"
INDEX_PATH = "./index/faiss_index.bin"
CHUNKS_PATH = "./index/chunks.pkl"
MODEL_NAME = "gemma2"  # Change to another local LLM if needed

# ====== SETUP EMBEDDINGS ======
model = SentenceTransformer("all-MiniLM-L6-v2")

# ====== LOAD & CHUNK DOCUMENTS ======
def load_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def load_docs(folder):
    """Load all documents from a folder."""
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            docs.append(load_pdf(path))
        elif file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_documents(docs):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text("\n".join(docs))

# ====== BUILD FAISS INDEX ======
def build_index(chunks):
    """Create and save FAISS index."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print("📁 FAISS index saved.")

# ====== SEARCH DOCUMENTS ======
def search_docs(query, top_k=3):
    """Retrieve the most relevant chunks for a query."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# ====== GENERATE RESPONSE ======
def generate_answer(question, context):
    """Use Ollama to generate an AI response based on retrieved context."""
    prompt = f"Answer based on this context:\n\n{context}\n\nQuestion: {question}"
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ====== MAIN SCRIPT ======
if __name__ == "__main__":
    # Check if index exists
    if not os.path.exists(INDEX_PATH):
        print("🔍 Index not found. Processing documents...")
        docs = load_docs(DOCS_FOLDER)
        chunks = chunk_documents(docs)
        build_index(chunks)
    else:
        print("✅ FAISS index found. Ready to answer questions!")

    # Chat loop
    while True:
        query = input("\n🔹 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        
        results = search_docs(query)
        context = "\n".join(results)
        answer = generate_answer(query, context)
        
        print("\n🤖 AI Answer:", answer)

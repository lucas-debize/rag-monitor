import os
import glob
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

DOCUMENTS_DIR = "/app/data/documents"
CHROMA_DIR = "/app/data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(directory):
    documents = []

    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    for pdf_path in pdf_files:
        print(f"Chargement PDF : {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

    md_files = glob.glob(os.path.join(directory, "*.md"))
    for md_path in md_files:
        print(f"Chargement Markdown : {md_path}")
        loader = UnstructuredMarkdownLoader(md_path)
        documents.extend(loader.load())

    print(f"Total documents chargés : {len(documents)}")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks générés : {len(chunks)}")
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )

def build_vectorstore(chunks):
    print(f"Initialisation des embeddings : {EMBEDDING_MODEL}")
    print(f"Stockage dans ChromaDB : {CHROMA_DIR}")
    vectorstore = get_vectorstore()
    if chunks:
        vectorstore.add_documents(chunks)
    print(f"Vectorstore : {vectorstore._collection.count()} vecteurs au total")
    return vectorstore


def main():
    if not os.path.exists(DOCUMENTS_DIR):
        raise FileNotFoundError(f"Le dossier {DOCUMENTS_DIR} n'existe pas")

    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise ValueError(f"Aucun document trouvé dans {DOCUMENTS_DIR}")

    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)

    print("\nTest de recherche de similarité...")
    results = vectorstore.similarity_search("test", k=2)
    print(f"Résultats trouvés : {len(results)}")
    for i, res in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(res.page_content[:200])

    print("\n=== ÉTAPE 2 VALIDÉE ===")
    print("Pipeline d'ingestion fonctionnelle")


if __name__ == "__main__":
    main()

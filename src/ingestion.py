import os
import glob
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCUMENTS_DIR = "/app/data/documents"
CHROMA_DIR = "/app/data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_chroma_client = None
_embeddings = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        client=get_chroma_client(),
    )

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
    print("\nAffichage des chunks générés :")

    for index, chunk in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"Chunk {index}/{len(chunks)}")
        print(f"Source : {chunk.metadata.get('source', 'unknown')}")
        print(f"Page : {chunk.metadata.get('page', 'unknown')}")
        print("-" * 80)
        print(chunk.page_content)

    return chunks

def build_vectorstore(chunks):
    print(f"Initialisation des embeddings : {EMBEDDING_MODEL}")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    print(f"Stockage dans ChromaDB : {CHROMA_DIR}")
    vectorstore = get_vectorstore()

    try:
        existing_ids = vectorstore._collection.get()["ids"]
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
            print(f"Ancienne collection vidée : {len(existing_ids)} vecteurs supprimés")
    except Exception as e:
        print(f"Avertissement nettoyage collection : {e}")

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
    results = vectorstore.similarity_search("test", k=5)

    print(f"Résultats trouvés : {len(results)}")

    for i, res in enumerate(results, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source : {res.metadata.get('source', 'unknown')}")
        print(f"Page : {res.metadata.get('page', 'unknown')}")
        print(res.page_content[:200])

    print("\n=== ÉTAPE 2 VALIDÉE ===")
    print("Pipeline d'ingestion fonctionnelle")

if __name__ == "__main__":
    main()

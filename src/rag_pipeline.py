import os
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

CHROMA_PATH = "/app/data/chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")

PROMPT_TEMPLATE = """Tu es un assistant qui répond exclusivement à partir du contexte fourni ci-dessous.

Règles strictes :
- Si la réponse n'est pas explicitement présente dans le contexte, réponds exactement : "Je ne sais pas, l'information n'est pas dans les documents fournis."
- Ne fais aucune supposition et n'utilise aucune connaissance externe.
- Cite systématiquement les sources utilisées sous la forme [Source: nom_du_fichier].
- Réponds de manière concise et factuelle.

Contexte :
{context}

Question : {question}

Réponse :"""


def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "inconnu").split("/")[-1]
        page = d.metadata.get("page", None)
        header = f"[Source: {source}" + (f", page {page}]" if page is not None else "]")
        formatted.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_chain():
    print("Chargement des embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Connexion à ChromaDB : {CHROMA_PATH}")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(f"Connexion à Ollama : {OLLAMA_URL} (modèle : {MODEL_NAME})")
    llm = OllamaLLM(base_url=OLLAMA_URL, model=MODEL_NAME, temperature=0.1)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(chain, retriever, question):
    print(f"\n{'='*60}")
    print(f"Question : {question}")
    print(f"{'='*60}")

    answer = chain.invoke(question)
    print(f"\nRéponse :\n{answer}")

    docs = retriever.invoke(question)
    print(f"\nSources récupérées ({len(docs)}) :")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "inconnu").split("/")[-1]
        page = d.metadata.get("page", "?")
        preview = d.page_content[:100].replace("\n", " ")
        print(f"  {i}. {src} (page {page}) — {preview}...")


def main():
    chain, retriever = build_chain()

    questions = [
        "De quoi parle ce document ?",
        "Quel est le sujet du sprint 6 ?",
        "Quelle est la capitale de l'Australie ?",
    ]

    for q in questions:
        answer_question(chain, retriever, q)

    print("\n=== ÉTAPE 3 VALIDÉE ===")
    print("Pipeline RAG fonctionnelle (retrieval + prompt + LLM + sources)")


if __name__ == "__main__":
    main()

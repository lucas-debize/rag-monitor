import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from src.prompts import get_prompt
from src.mlflow_tracker import init_mlflow, start_run, log_question, log_aggregates, end_run, Timer

CHROMA_PATH = "/app/data/chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client

def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "inconnu").split("/")[-1]
        page = d.metadata.get("page", None)
        header = f"[Source: {source}" + (f", page {page}]" if page is not None else "]")
        formatted.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

def clean_answer(answer: str):
    normalized = answer.lower()

    refusal_patterns = [
        "je ne sais pas",
        "pas dans les documents",
        "information n'est pas dans",
        "ne contient pas",
        "pas mentionné",
        "pas présent",
    ]

    if any(p in normalized for p in refusal_patterns):
        return "Je ne sais pas, l'information n'est pas dans les documents fournis."

    return answer

def build_chain(prompt_version: str = None):
    prompt_version = prompt_version or PROMPT_VERSION
    prompt_def = get_prompt(prompt_version)

    print("Chargement des embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Connexion à ChromaDB : {CHROMA_PATH}")
    chroma_client = get_chroma_client()
    vectorstore = Chroma(
        client=chroma_client,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Connexion à Ollama : {OLLAMA_URL} (modèle : {MODEL_NAME})")
    llm = OllamaLLM(
        base_url=OLLAMA_URL,
        model=MODEL_NAME,
        temperature=0,
        seed=42,
        num_ctx=4096,
    )

    prompt = PromptTemplate.from_template(prompt_def["template"])

    base_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if prompt_version == "v3":
        chain = base_chain | clean_answer
    else:
        chain = base_chain

    return chain, retriever, prompt_def

def extract_sources(docs):
    sources = []
    for d in docs:
        src = d.metadata.get("source", "inconnu").split("/")[-1]
        page = d.metadata.get("page", "?")
        sources.append(f"{src}#p{page}")
    return sources

def answer_question(chain, retriever, question, index=0):
    print(f"\n{'='*60}\nQuestion : {question}\n{'='*60}")
    with Timer() as t:
        answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = extract_sources(docs)

    print(f"\nRéponse :\n{answer}")
    print(f"\nLatence : {t.elapsed:.2f}s")
    print(f"Sources récupérées ({len(sources)}) :")
    for i, s in enumerate(sources, 1):
        print(f"  {i}. {s}")

    log_question(index, question, answer, t.elapsed, len(sources), sources)
    return {"answer": answer, "latency": t.elapsed, "sources": sources}

def run_pipeline(prompt_version: str, questions: list):
    chain, retriever, prompt_def = build_chain(prompt_version)

    params = {
        "prompt_version": prompt_version,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "num_questions": len(questions),
    }

    init_mlflow()
    start_run(
        run_name=f"rag-{prompt_version}",
        params=params,
        prompt_template=prompt_def["template"],
        prompt_version=prompt_version,
    )

    latencies, lengths, nums = [], [], []
    try:
        for i, q in enumerate(questions):
            res = answer_question(chain, retriever, q, index=i)
            latencies.append(res["latency"])
            lengths.append(len(res["answer"]))
            nums.append(len(res["sources"]))
        log_aggregates(latencies, lengths, nums)
    finally:
        end_run()

    return {"latencies": latencies, "lengths": lengths, "nums_sources": nums}

def main():
    questions = [
        "De quoi parle ce document ?",
        "Quel est le sujet du sprint 6 ?",
        "Quelle est la capitale de l'Australie ?",
    ]
    run_pipeline(PROMPT_VERSION, questions)
    print("\n=== PIPELINE RAG EXÉCUTÉE ET TRACKÉE DANS MLFLOW ===")
 
if __name__ == "__main__":
    main()

import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "mistral")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_judge():
    llm = ChatOpenAI(
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        model=JUDGE_MODEL,
        temperature=0.0,
        timeout=300,
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def load_testset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Test set chargé : {len(data)} questions depuis {path}")
    return data


def build_evaluation_dataset(samples):
    return Dataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])


def run_ragas(samples):
    dataset = build_evaluation_dataset(samples)
    judge_llm, judge_emb = build_judge()
    run_config = RunConfig(timeout=1800, max_workers=1, max_retries=3)

    print(f"Évaluation RAGAS sur {len(samples)} échantillons (juge : {JUDGE_MODEL} via OpenAI-compat)...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    scores = {
        "faithfulness": float(df["faithfulness"].mean(skipna=True)),
        "answer_relevancy": float(df["answer_relevancy"].mean(skipna=True)),
        "context_precision": float(df["context_precision"].mean(skipna=True)),
    }
    return scores, df

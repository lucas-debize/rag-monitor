import os
import json
import re
import pandas as pd
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
REFUSAL_SENTENCE = "Je ne sais pas, l'information n'est pas dans les documents fournis."

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

def normalize_text(value):
    value = value.lower()
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def normalize_source_name(source):
    source = source.strip()
    source = source.split("#")[0]
    source = source.split(",")[0]
    source = source.split("/")[-1]
    return source.strip()

def remove_citations_from_answer(answer):
    answer = re.sub(r"\[Source:\s*[^\]]+\]", "", answer)
    answer = re.sub(r"\[Sources:\s*[^\]]+\]", "", answer)
    answer = re.sub(r"Sources?\s*:\s*.*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip()

def extract_cited_sources(answer):
    return re.findall(r"\[Source:\s*([^\]]+)\]", answer)

def is_refusal(answer):
    normalized = normalize_text(answer)
    expected = normalize_text(REFUSAL_SENTENCE)
    refusal_patterns = [
        expected,
        "je ne sais pas",
        "l'information n'est pas dans les documents",
        "information n'est pas dans les documents fournis",
    ]
    return any(pattern in normalized for pattern in refusal_patterns)

def contains_expected_keywords(answer, expected_keywords):
    if not expected_keywords:
        return 0.0

    normalized_answer = normalize_text(answer)
    matches = 0

    for keyword in expected_keywords:
        if normalize_text(keyword) in normalized_answer:
            matches += 1

    return matches / len(expected_keywords)

def compute_answer_correctness(sample):
    category = sample.get("category", "unknown")
    answer = remove_citations_from_answer(sample.get("answer", ""))
    expected_keywords = sample.get("expected_keywords", [])

    if category == "out_of_scope":
        return 1.0 if is_refusal(answer) else 0.0

    if category == "factual_in_doc":
        if is_refusal(answer):
            return 0.0
        return contains_expected_keywords(answer, expected_keywords)

    return 0.0

def compute_context_recall(sample):
    category = sample.get("category", "unknown")
    expected_sources = [normalize_source_name(s) for s in sample.get("expected_sources", [])]
    retrieved_sources = [normalize_source_name(s) for s in sample.get("sources", [])]

    if category == "out_of_scope":
        return 1.0

    if not expected_sources:
        return 0.0

    for expected_source in expected_sources:
        if expected_source in retrieved_sources:
            return 1.0

    return 0.0

def compute_citation_score(sample):
    category = sample.get("category", "unknown")
    answer = sample.get("answer", "")
    cited_sources = [normalize_source_name(s) for s in extract_cited_sources(answer)]
    retrieved_sources = [normalize_source_name(s) for s in sample.get("sources", [])]
    expected_sources = [normalize_source_name(s) for s in sample.get("expected_sources", [])]

    if category == "out_of_scope":
        return 1.0 if len(cited_sources) == 0 else 0.0

    if category == "factual_in_doc":
        if len(cited_sources) == 0:
            return 0.0

        for cited_source in cited_sources:
            if cited_source.lower() == "aucune":
                return 0.0
            if cited_source in retrieved_sources and cited_source in expected_sources:
                return 1.0

        return 0.0

    return 0.0

def compute_refusal_score(sample):
    category = sample.get("category", "unknown")
    answer = sample.get("answer", "")

    if category == "out_of_scope":
        return 1.0 if is_refusal(answer) else 0.0

    if category == "factual_in_doc":
        return 0.0 if is_refusal(answer) else 1.0

    return 0.0

def compute_hallucination_score(sample):
    category = sample.get("category", "unknown")
    answer = sample.get("answer", "")
    forbidden_keywords = sample.get("forbidden_keywords", [])

    if category == "out_of_scope":
        return 1.0 if is_refusal(answer) else 0.0

    normalized_answer = normalize_text(remove_citations_from_answer(answer))

    for keyword in forbidden_keywords:
        if normalize_text(keyword) in normalized_answer:
            return 0.0

    if is_refusal(answer):
        return 0.0

    return 1.0

def build_evaluation_dataframe(samples):
    rows = []

    for sample in samples:
        answer_correctness = compute_answer_correctness(sample)
        context_recall = compute_context_recall(sample)
        citation_score = compute_citation_score(sample)
        refusal_score = compute_refusal_score(sample)
        hallucination_score = compute_hallucination_score(sample)

        rag_monitor_score = (
            answer_correctness * 0.30
            + context_recall * 0.15
            + citation_score * 0.30
            + refusal_score * 0.15
            + hallucination_score * 0.10
        )

        rows.append({
            "question": sample.get("question", ""),
            "category": sample.get("category", "unknown"),
            "answer": sample.get("answer", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "sources": sample.get("sources", []),
            "expected_sources": sample.get("expected_sources", []),
            "expected_keywords": sample.get("expected_keywords", []),
            "answer_correctness": answer_correctness,
            "context_recall": context_recall,
            "citation_score": citation_score,
            "refusal_score": refusal_score,
            "hallucination_score": hallucination_score,
            "rag_monitor_score": rag_monitor_score,
        })

    return pd.DataFrame(rows)

def safe_mean(df, column):
    if df.empty:
        return 0.0
    return float(df[column].mean())

def compute_business_scores(df):
    factual_df = df[df["category"] == "factual_in_doc"]
    out_df = df[df["category"] == "out_of_scope"]

    factual_answer_correctness = safe_mean(factual_df, "answer_correctness")
    factual_context_recall = safe_mean(factual_df, "context_recall")
    factual_citation_score = safe_mean(factual_df, "citation_score")
    factual_hallucination_score = safe_mean(factual_df, "hallucination_score")

    out_of_scope_refusal_score = safe_mean(out_df, "refusal_score")
    out_of_scope_citation_score = safe_mean(out_df, "citation_score")
    out_of_scope_hallucination_score = safe_mean(out_df, "hallucination_score")

    rag_monitor_score = (
        factual_answer_correctness * 0.30
        + factual_context_recall * 0.15
        + factual_citation_score * 0.25
        + factual_hallucination_score * 0.10
        + out_of_scope_refusal_score * 0.10
        + out_of_scope_hallucination_score * 0.10
    )

    scores = {
        "answer_correctness": factual_answer_correctness,
        "context_recall": factual_context_recall,
        "citation_score": factual_citation_score,
        "refusal_score": out_of_scope_refusal_score,
        "hallucination_score": out_of_scope_hallucination_score,
        "rag_monitor_score": float(rag_monitor_score),
        "factual_answer_correctness": factual_answer_correctness,
        "factual_context_recall": factual_context_recall,
        "factual_citation_score": factual_citation_score,
        "factual_hallucination_score": factual_hallucination_score,
        "out_of_scope_refusal_score": out_of_scope_refusal_score,
        "out_of_scope_citation_score": out_of_scope_citation_score,
        "out_of_scope_hallucination_score": out_of_scope_hallucination_score,
    }

    return scores

def build_ragas_dataset(samples):
    return Dataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in samples
    ])

def run_optional_ragas(samples, df):
    judge_llm, judge_emb = build_judge()
    run_config = RunConfig(timeout=1800, max_workers=1, max_retries=3)

    print(f"Évaluation RAGAS secondaire sur {len(samples)} échantillons")

    result = evaluate(
        dataset=build_ragas_dataset(samples),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
    )

    ragas_df = result.to_pandas()

    df["faithfulness"] = ragas_df["faithfulness"]
    df["answer_relevancy"] = ragas_df["answer_relevancy"]
    df["context_precision"] = ragas_df["context_precision"]

    scores = {
        "faithfulness": float(df["faithfulness"].mean(skipna=True)),
        "answer_relevancy": float(df["answer_relevancy"].mean(skipna=True)),
        "context_precision": float(df["context_precision"].mean(skipna=True)),
    }

    return scores, df

def run_evaluation(samples, with_ragas=False):
    df = build_evaluation_dataframe(samples)
    scores = compute_business_scores(df)

    if with_ragas:
        ragas_scores, df = run_optional_ragas(samples, df)
        scores.update(ragas_scores)

    return scores, df

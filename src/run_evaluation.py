import os
import sys
import json
import argparse
import mlflow
import math

from src.rag_pipeline import build_chain, extract_sources, format_docs
from src.evaluator import load_testset, run_evaluation
from src.mlflow_tracker import init_mlflow
from src.ingestion import run_ingestion

TESTSET_PATH = os.getenv("TESTSET_PATH", "/app/data/testset/testset.json")
OUTPUT_DIR = "/app/data/eval_results"
RAG_MONITOR_THRESHOLD = float(os.getenv("RAG_MONITOR_THRESHOLD", "0.75"))
ANSWER_CORRECTNESS_THRESHOLD = float(os.getenv("ANSWER_CORRECTNESS_THRESHOLD", "0.50"))
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.6"))
CITATION_SCORE_THRESHOLD = float(os.getenv("CITATION_SCORE_THRESHOLD", "0.6"))
REFUSAL_SCORE_THRESHOLD = float(os.getenv("REFUSAL_SCORE_THRESHOLD", "0.6"))

BUSINESS_KEYS = {
    "answer_correctness",
    "context_recall",
    "citation_score",
    "refusal_score",
    "hallucination_score",
    "rag_monitor_score",
    "factual_answer_correctness",
    "factual_context_recall",
    "factual_citation_score",
    "factual_hallucination_score",
    "out_of_scope_refusal_score",
    "out_of_scope_citation_score",
    "out_of_scope_hallucination_score",
}

def generate_predictions(prompt_version, testset):
    chain, retriever, prompt_def = build_chain(prompt_version)
    samples = []

    for i, item in enumerate(testset):
        q = item["question"]
        print(f"\n[{i + 1}/{len(testset)}] Q: {q}")

        answer = chain.invoke(q)
        docs = retriever.invoke(q)
        contexts = [format_docs([d]) for d in docs]
        sources = extract_sources(docs)

        print(f"  → Réponse: {answer[:160]}...")

        samples.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "sources": sources,
            "category": item.get("category", "unknown"),
            "expected_keywords": item.get("expected_keywords", []),
            "expected_sources": item.get("expected_sources", []),
            "forbidden_keywords": item.get("forbidden_keywords", []),
        })

    return samples, prompt_def

def save_results(prompt_version, eval_mode, scores, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}_{eval_mode}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}_{eval_mode}.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés : {csv_path}")

    return csv_path, json_path

def evaluate_version(prompt_version, testset, with_ragas):
    eval_mode = "ragas" if with_ragas else "base"

    print(f"\n{'#' * 60}\n# Évaluation RAG Monitor — prompt {prompt_version} — mode {eval_mode}\n{'#' * 60}")

    samples, prompt_def = generate_predictions(prompt_version, testset)
    scores, df = run_evaluation(samples, with_ragas=with_ragas)

    df["eval_mode"] = eval_mode
    scores["eval_mode"] = eval_mode

    print("\n--- Scores MÉTIER (déterministes, sans IA juge) ---")
    for k, v in scores.items():
        if isinstance(v, (int, float)) and k in BUSINESS_KEYS:
            print(f"  {k:35s}: {v:.4f}")

    if with_ragas:
        print("\n--- Scores RAGAS (IA juge — indicatifs) ---")
        for k, v in scores.items():
            if isinstance(v, (int, float)) and k.startswith("ragas_"):
                print(f"  {k:35s}: {v:.4f}")

    csv_path, json_path = save_results(prompt_version, eval_mode, scores, df)

    init_mlflow()

    metrics_to_log = {
        k: v
        for k, v in scores.items()
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
    }

    with mlflow.start_run(run_name=f"eval-{prompt_version}-{eval_mode}"):
        mlflow.log_params({
            "prompt_version": prompt_version,
            "eval_mode": eval_mode,
            "judge_model": os.getenv("JUDGE_MODEL", "mistral:7b-instruct-v0.3-q4_0"),
            "testset_size": len(testset),
            "with_ragas": with_ragas,
        })
        mlflow.set_tag("eval_mode", eval_mode)
        mlflow.set_tag("prompt_version", prompt_version)
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(json_path)
        mlflow.log_text(prompt_def["template"], f"prompt_{prompt_version}.txt")

    return scores

def is_valid_number(value):
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))

def check_scores(summary, with_ragas):
    version = "v3"

    if version not in summary:
        print("\n❌ ÉCHEC : la version v3 doit être évaluée pour vérifier les seuils")
        sys.exit(1)

    scores = summary[version]

    use_faithfulness = with_ragas and is_valid_number(scores.get("ragas_faithfulness"))

    if with_ragas and not use_faithfulness:
        print("\n⚠️  ragas_faithfulness invalide ou indisponible : bascule sur factual_answer_correctness")

    deterministic_keys = [
        "rag_monitor_score",
        "factual_citation_score",
        "out_of_scope_refusal_score",
    ]

    if not use_faithfulness:
        deterministic_keys.append("factual_answer_correctness")

    for metric in deterministic_keys:
        value = scores.get(metric)
        if not is_valid_number(value):
            print(f"\n❌ ÉCHEC : score invalide ou NaN détecté pour {version} sur {metric}")
            sys.exit(2)

    if scores["rag_monitor_score"] < RAG_MONITOR_THRESHOLD:
        print(f"\n❌ ÉCHEC : {version} rag_monitor_score={scores['rag_monitor_score']:.3f} < {RAG_MONITOR_THRESHOLD}")
        sys.exit(1)

    if use_faithfulness:
        if scores["ragas_faithfulness"] < FAITHFULNESS_THRESHOLD:
            print(f"\n❌ ÉCHEC : {version} ragas_faithfulness={scores['ragas_faithfulness']:.3f} < {FAITHFULNESS_THRESHOLD}")
            sys.exit(1)
    else:
        if scores["factual_answer_correctness"] < ANSWER_CORRECTNESS_THRESHOLD:
            print(f"\n❌ ÉCHEC : {version} factual_answer_correctness={scores['factual_answer_correctness']:.3f} < {ANSWER_CORRECTNESS_THRESHOLD}")
            sys.exit(1)

    if scores["factual_citation_score"] < CITATION_SCORE_THRESHOLD:
        print(f"\n❌ ÉCHEC : {version} factual_citation_score={scores['factual_citation_score']:.3f} < {CITATION_SCORE_THRESHOLD}")
        sys.exit(1)

    if scores["out_of_scope_refusal_score"] < REFUSAL_SCORE_THRESHOLD:
        print(f"\n❌ ÉCHEC : {version} out_of_scope_refusal_score={scores['out_of_scope_refusal_score']:.3f} < {REFUSAL_SCORE_THRESHOLD}")
        sys.exit(1)

    print("\n✅ La version v3 respecte les seuils RAG Monitor")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--check-threshold", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--without-ragas", action="store_true")
    args = parser.parse_args()

    with_ragas = not args.without_ragas

    run_ingestion()

    testset = load_testset(TESTSET_PATH)

    if args.max_samples:
        testset = testset[:args.max_samples]
        print(f"Test set limité à {len(testset)} questions")

    eval_mode = "ragas" if with_ragas else "base"
    summary = {}

    for version in args.versions:
        summary[version] = evaluate_version(version, testset, with_ragas)

    print(f"\n{'=' * 105}\nRÉCAPITULATIF\n{'=' * 105}")

    if with_ragas:
        second_header = "Faithfulness"
        second_key = "ragas_faithfulness"
    else:
        second_header = "AnswerCorr"
        second_key = "factual_answer_correctness"

    print(
        f"{'Version':<12}"
        f"{'RAG Score':<14}"
        f"{second_header:<16}"
        f"{'Context':<12}"
        f"{'Citations':<12}"
        f"{'OOS Refusal':<14}"
    )

    for version, scores in summary.items():
        second_value = scores.get(second_key, float("nan"))
        print(
            f"{version:<12}"
            f"{scores['rag_monitor_score']:<14.4f}"
            f"{second_value:<16.4f}"
            f"{scores['factual_context_recall']:<12.4f}"
            f"{scores['factual_citation_score']:<12.4f}"
            f"{scores['out_of_scope_refusal_score']:<14.4f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{eval_mode}.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nRésumé : {summary_path}")

    if args.check_threshold:
        check_scores(summary, with_ragas)

    print("\n=== ÉVALUATION MÉTIER RAG MONITOR TERMINÉE ===")

if __name__ == "__main__":
    main()

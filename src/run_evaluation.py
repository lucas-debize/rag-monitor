import os
import sys
import json
import argparse
import mlflow
import math

from src.rag_pipeline import build_chain, extract_sources, format_docs
from src.evaluator import load_testset, run_evaluation
from src.mlflow_tracker import init_mlflow

TESTSET_PATH = os.getenv("TESTSET_PATH", "/app/data/testset/testset.json")
OUTPUT_DIR = "/app/data/eval_results"
RAG_MONITOR_THRESHOLD = float(os.getenv("RAG_MONITOR_THRESHOLD", "0.75"))
ANSWER_CORRECTNESS_THRESHOLD = float(os.getenv("ANSWER_CORRECTNESS_THRESHOLD", "0.50"))
CITATION_SCORE_THRESHOLD = float(os.getenv("CITATION_SCORE_THRESHOLD", "0.75"))
REFUSAL_SCORE_THRESHOLD = float(os.getenv("REFUSAL_SCORE_THRESHOLD", "0.75"))

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

def save_results(prompt_version, scores, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés : {csv_path}")

    return csv_path, json_path

def evaluate_version(prompt_version, testset, with_ragas):
    print(f"\n{'#' * 60}\n# Évaluation métier RAG Monitor — prompt {prompt_version}\n{'#' * 60}")

    samples, prompt_def = generate_predictions(prompt_version, testset)
    scores, df = run_evaluation(samples, with_ragas=with_ragas)

    print("\n--- Scores RAG Monitor ---")
    for k, v in scores.items():
        print(f"  {k:35s}: {v:.4f}")

    csv_path, json_path = save_results(prompt_version, scores, df)

    init_mlflow()

    with mlflow.start_run(run_name=f"eval-{prompt_version}"):
        mlflow.log_params({
            "prompt_version": prompt_version,
            "judge_model": os.getenv("JUDGE_MODEL", "mistral"),
            "testset_size": len(testset),
            "with_ragas": with_ragas,
        })
        mlflow.log_metrics(scores)
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(json_path)
        mlflow.log_text(prompt_def["template"], f"prompt_{prompt_version}.txt")

    return scores

def check_scores(summary):
    version = "v3"

    if version not in summary:
        print("\n❌ ÉCHEC : la version v3 doit être évaluée pour vérifier les seuils")
        sys.exit(1)

    scores = summary[version]

    for metric, value in scores.items():
        if isinstance(value, float) and math.isnan(value):
            print(f"\n❌ ÉCHEC : score NaN détecté pour {version} sur {metric}")
            sys.exit(2)

    if scores["rag_monitor_score"] < RAG_MONITOR_THRESHOLD:
        print(f"\n❌ ÉCHEC : {version} rag_monitor_score={scores['rag_monitor_score']:.3f} < {RAG_MONITOR_THRESHOLD}")
        sys.exit(1)

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
    parser.add_argument("--with-ragas", action="store_true")
    args = parser.parse_args()

    testset = load_testset(TESTSET_PATH)

    if args.max_samples:
        testset = testset[:args.max_samples]
        print(f"Test set limité à {len(testset)} questions")

    summary = {}

    for version in args.versions:
        summary[version] = evaluate_version(version, testset, args.with_ragas)

    print(f"\n{'=' * 105}\nRÉCAPITULATIF\n{'=' * 105}")
    print(
        f"{'Version':<10}"
        f"{'RAG Score':<15}"
        f"{'Factual OK':<15}"
        f"{'Context':<15}"
        f"{'Citations':<15}"
        f"{'OOS Refusal':<15}"
        f"{'OOS Halluc.':<15}"
    )

    for version, scores in summary.items():
        print(
            f"{version:<10}"
            f"{scores['rag_monitor_score']:<15.4f}"
            f"{scores['factual_answer_correctness']:<15.4f}"
            f"{scores['factual_context_recall']:<15.4f}"
            f"{scores['factual_citation_score']:<15.4f}"
            f"{scores['out_of_scope_refusal_score']:<15.4f}"
            f"{scores['out_of_scope_hallucination_score']:<15.4f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nRésumé : {summary_path}")

    if args.check_threshold:
        check_scores(summary)

    print("\n=== ÉVALUATION MÉTIER RAG MONITOR TERMINÉE ===")

if __name__ == "__main__":
    main()

import os
import sys
import json
import argparse
import mlflow
import math

from src.rag_pipeline import build_chain, extract_sources
from src.evaluator import load_testset, run_ragas
from src.mlflow_tracker import init_mlflow

TESTSET_PATH = os.getenv("TESTSET_PATH", "/app/data/testset/testset.json")
OUTPUT_DIR = "/app/data/eval_results"
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.75"))

def generate_predictions(prompt_version, testset):
    chain, retriever, prompt_def = build_chain(prompt_version)
    samples = []
    for i, item in enumerate(testset):
        q = item["question"]
        print(f"\n[{i+1}/{len(testset)}] Q: {q}")
        answer = chain.invoke(q)
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        sources = extract_sources(docs)
        print(f"  → Réponse: {answer[:120]}...")
        samples.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "sources": sources,
            "category": item.get("category", "unknown"),
        })
    return samples, prompt_def

def save_results(prompt_version, scores, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"eval_{prompt_version}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"\nRésultats sauvegardés : {csv_path}")
    return csv_path, json_path

def evaluate_version(prompt_version, testset):
    print(f"\n{'#'*60}\n# Évaluation RAGAS — prompt {prompt_version}\n{'#'*60}")

    samples, prompt_def = generate_predictions(prompt_version, testset)
    scores, df = run_ragas(samples)

    print("\n--- Scores RAGAS ---")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")

    csv_path, json_path = save_results(prompt_version, scores, df)

    init_mlflow()
    with mlflow.start_run(run_name=f"eval-{prompt_version}"):
        mlflow.log_params({
            "prompt_version": prompt_version,
            "judge_model": os.getenv("JUDGE_MODEL", "mistral"),
            "testset_size": len(testset),
        })
        mlflow.log_metrics(scores)
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(json_path)
        mlflow.log_text(prompt_def["template"], f"prompt_{prompt_version}.txt")

    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--check-threshold", action="store_true",
                        help="Échoue si faithfulness < seuil (pour CI/CD)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limite le nombre de questions évaluées")
    args = parser.parse_args()

    testset = load_testset(TESTSET_PATH)
    if args.max_samples:
        testset = testset[:args.max_samples]
        print(f"Test set limité à {len(testset)} questions")

    summary = {}
    for v in args.versions:
        summary[v] = evaluate_version(v, testset)

    print(f"\n{'='*60}\nRÉCAPITULATIF\n{'='*60}")
    print(f"{'Version':<10}{'Faithfulness':<18}{'Relevancy':<18}{'CtxPrecision':<18}")
    for v, s in summary.items():
        print(f"{v:<10}{s['faithfulness']:<18.4f}{s['answer_relevancy']:<18.4f}{s['context_precision']:<18.4f}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRésumé : {summary_path}")

    for v, s in summary.items():
        if any(math.isnan(val) for val in s.values()):
            print(f"\n❌ ÉCHEC : scores NaN détectés pour {v} — l'évaluation RAGAS a planté (vérifier la connexion au juge LLM)")
            sys.exit(2)

    if args.check_threshold:
        for v, s in summary.items():
            if s["faithfulness"] < FAITHFULNESS_THRESHOLD:
                print(f"\n❌ ÉCHEC : {v} faithfulness={s['faithfulness']:.3f} < {FAITHFULNESS_THRESHOLD}")
                sys.exit(1)
        print(f"\n✅ Toutes les versions respectent le seuil ({FAITHFULNESS_THRESHOLD})")

    print("\n=== ÉTAPE 5 VALIDÉE ===")

if __name__ == "__main__":
    main()

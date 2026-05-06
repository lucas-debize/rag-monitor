from src.rag_pipeline import run_pipeline
from src.prompts import list_versions

QUESTIONS = [
    "De quoi parle ce document ?",
    "Quel est le sujet du sprint 6 ?",
    "Quelle est la deadline de livraison ?",
    "Quels sont les challenges à relever ?",
    "Quelle est la capitale de l'Australie ?",
]

VERSIONS_TO_COMPARE = ["v1", "v2", "v3"]


def main():
    available = list_versions()
    versions = [v for v in VERSIONS_TO_COMPARE if v in available]

    print(f"Comparaison des versions de prompts : {versions}")
    print(f"Nombre de questions : {len(QUESTIONS)}\n")

    summary = {}
    for v in versions:
        print(f"\n{'#'*60}\n# Exécution avec prompt {v}\n{'#'*60}")
        result = run_pipeline(v, QUESTIONS)
        avg_latency = sum(result["latencies"]) / len(result["latencies"]) if result["latencies"] else 0
        avg_length = sum(result["lengths"]) / len(result["lengths"]) if result["lengths"] else 0
        summary[v] = {"avg_latency": avg_latency, "avg_length": avg_length}

    print(f"\n{'='*60}\nRÉSUMÉ DE LA COMPARAISON\n{'='*60}")
    for v, stats in summary.items():
        print(f"  {v} -> latence moy: {stats['avg_latency']:.2f}s | longueur moy: {stats['avg_length']:.0f} chars")

    print("\nUI MLflow disponible sur http://localhost:5000")
    print("\n=== ÉTAPE 4 VALIDÉE ===")
    print("Tracking + versioning de prompts opérationnel via MLflow")


if __name__ == "__main__":
    main()

import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    ColumnSummaryMetric,
    TextDescriptorsDriftMetric,
    EmbeddingsDriftMetric,
)
from evidently.descriptors import TextLength, SentenceCount, NonLetterCharacterPercentage
from evidently import ColumnMapping
from src.metrics_logger import METRICS_FILE

DRIFT_REPORT_PATH = "/app/data/drift_report.html"

REFERENCE_QUESTIONS = [
    "De quoi parle ce document ?",
    "Quel est le sujet du sprint 6 ?",
    "Quelle est la capitale de l'Australie ?",
    "Qu'est-ce qu'un backlog ?",
    "Comment configurer Docker ?",
    "Qu'est-ce que RAG ?",
    "Comment réduire les hallucinations ?",
    "Qui est le responsable du projet RAG Monitor ?",
    "Quelles sont les technologies utilisées ?",
    "Quel est le but de la release 2.0 ?",
]

def _embed_questions(texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(list(texts), show_progress_bar=False)
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])],
    )
    embedding_df.reset_index(drop=True, inplace=True)
    return embedding_df

def generate_drift_report() -> bool:
    if not os.path.exists(METRICS_FILE):
        print("Aucune donnée utilisateur disponible pour calculer la dérive.")
        return False

    current_df = pd.read_csv(METRICS_FILE)
    if current_df.empty or len(current_df) < 3:
        print("Pas assez de requêtes utilisateur pour calculer la dérive (minimum 3 requêtes requises).")
        return False

    current_texts = current_df["question"].dropna().reset_index(drop=True)
    reference_texts = pd.Series(REFERENCE_QUESTIONS)

    current_data = pd.DataFrame({"text": current_texts})
    ref_data = pd.DataFrame({"text": reference_texts})

    current_embeddings = _embed_questions(current_texts)
    reference_embeddings = _embed_questions(reference_texts)

    current_full = pd.concat([current_data, current_embeddings], axis=1)
    reference_full = pd.concat([ref_data, reference_embeddings], axis=1)

    embedding_columns = [c for c in current_full.columns if c.startswith("emb_")]

    column_mapping = ColumnMapping()
    column_mapping.text_features = ["text"]
    column_mapping.embeddings = {"question_embeddings": embedding_columns}

    text_report = Report(
        metrics=[
            ColumnSummaryMetric(column_name="text"),
            TextDescriptorsDriftMetric(
                column_name="text",
                descriptors={
                    "Longueur du texte": TextLength(),
                    "Nombre de phrases": SentenceCount(),
                    "Caractères non alphabétiques %": NonLetterCharacterPercentage(),
                },
            ),
            EmbeddingsDriftMetric("question_embeddings"),
        ]
    )

    text_report.run(
        reference_data=reference_full,
        current_data=current_full,
        column_mapping=column_mapping,
    )

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    text_report.save_html(DRIFT_REPORT_PATH)
    print(f"Rapport de dérive généré avec succès dans : {DRIFT_REPORT_PATH}")
    return True

def get_semantic_drift() -> dict:
    if not os.path.exists(METRICS_FILE):
        return {"drift_detected": False, "score": None}

    current_df = pd.read_csv(METRICS_FILE)
    if len(current_df) < 5:
        return {"drift_detected": False, "score": None}

    current_texts = current_df["question"].dropna().reset_index(drop=True)
    reference_texts = pd.Series(REFERENCE_QUESTIONS)

    current_embeddings = _embed_questions(current_texts)
    reference_embeddings = _embed_questions(reference_texts)

    embedding_columns = [c for c in current_embeddings.columns]

    column_mapping = ColumnMapping()
    column_mapping.embeddings = {"question_embeddings": embedding_columns}

    report = Report(metrics=[EmbeddingsDriftMetric("question_embeddings")])
    report.run(
        reference_data=reference_embeddings,
        current_data=current_embeddings,
        column_mapping=column_mapping,
    )

    result = report.as_dict()
    metric_result = result["metrics"][0]["result"]
    drift_detected = bool(metric_result.get("drift_detected", False))
    drift_score = metric_result.get("drift_score")

    return {"drift_detected": drift_detected, "score": drift_score}

def check_drift_status() -> dict:
    if not os.path.exists(METRICS_FILE):
        return {"drift_detected": False, "message": "Pas de données d'historique."}

    current_df = pd.read_csv(METRICS_FILE)
    if len(current_df) < 5:
        return {"drift_detected": False, "message": "Collecte de données en cours (min. 5 requêtes requises)."}

    refusals = current_df.tail(10)["answer"].str.contains("Je ne sais pas", case=False, na=False).sum()
    refusal_rate = refusals / min(10, len(current_df))

    semantic = get_semantic_drift()
    semantic_msg = ""
    if semantic["score"] is not None:
        semantic_msg = f" Dérive sémantique des questions : {'OUI' if semantic['drift_detected'] else 'non'} (score {semantic['score']:.2f})."

    if refusal_rate >= 0.5 or semantic["drift_detected"]:
        return {
            "drift_detected": True,
            "level": "ERROR",
            "message": f"Alerte dérive ! {refusal_rate*100:.0f}% des dernières requêtes ont échoué.{semantic_msg}",
        }
    elif refusal_rate >= 0.3:
        return {
            "drift_detected": True,
            "level": "WARNING",
            "message": f"Attention : {refusal_rate*100:.0f}% des dernières requêtes n'ont pas trouvé de réponse.{semantic_msg}",
        }

    return {
        "drift_detected": False,
        "message": f"Aucune dérive significative détectée. Le comportement utilisateur est stable.{semantic_msg}",
    }

import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnSummaryMetric, TextDescriptorsDriftMetric, EmbeddingsDriftMetric
from evidently.descriptors import TextLength, SentenceCount, NonLetterCharacterPercentage
from evidently import ColumnMapping
from src.metrics_logger import METRICS_FILE

DRIFT_REPORT_PATH = "/app/data/drift_report.html"
MIN_REPORT_QUERIES = 3
MIN_STATUS_QUERIES = 5
MIN_EMBEDDING_TOTAL_SAMPLES = 31
WARNING_REFUSAL_RATE = 0.3
ERROR_REFUSAL_RATE = 0.5

REFERENCE_QUESTIONS = [
    "Quel est le sujet de ce sprint ?",
    "Quel engine est utilisé dans ce projet ?",
    "Qui est l'auteur du document ?",
    "Pourquoi le projet est-il un défi dès le départ ?",
    "Quel est le premier défi à relever ?",
    "Que se passe-t-il quand on appuie sur Play ?",
    "Quel problème ont les cristaux visuellement ?",
    "Quel problème ont les sons des cristaux ?",
    "Qu'est-ce qui ne fonctionne pas avec l'interaction ?",
    "Pourquoi le puzzle est-il un problème ?",
    "Quel est le FPS actuel dans la scène vide ?",
    "Quel FPS est considéré comme acceptable ?",
    "Que faut-il ajouter au jeu une fois corrigé ?",
    "Quelles sont les causes de penalty points ?",
    "Quelle est la date limite de rendu ?",
    "Comment doit-on livrer le build ?",
    "À quelle adresse email envoyer le projet ?",
    "Qui faut-il ajouter comme collaborateur sur GitHub ?",
]

def _load_metrics(prompt_version=None, window_size=None) -> pd.DataFrame:
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame()

    metrics_df = pd.read_csv(METRICS_FILE)

    if metrics_df.empty:
        return pd.DataFrame()

    if "timestamp" in metrics_df.columns:
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"], errors="coerce")
        metrics_df = metrics_df.sort_values("timestamp")

    if prompt_version and prompt_version != "Toutes" and "prompt_version" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["prompt_version"] == prompt_version]

    if window_size:
        metrics_df = metrics_df.tail(int(window_size))

    return metrics_df.reset_index(drop=True)

def _load_current_questions(prompt_version=None, window_size=None) -> pd.Series:
    metrics_df = _load_metrics(prompt_version=prompt_version, window_size=window_size)

    if metrics_df.empty or "question" not in metrics_df.columns:
        return pd.Series(dtype=str)

    return metrics_df["question"].dropna().astype(str).reset_index(drop=True)

def _can_use_embeddings(current_texts: pd.Series) -> bool:
    total_samples = len(REFERENCE_QUESTIONS) + len(current_texts)
    return total_samples >= MIN_EMBEDDING_TOTAL_SAMPLES

def _embed_questions(texts: pd.Series) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts.tolist(), show_progress_bar=False)

    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{index}" for index in range(embeddings.shape[1])],
    )

    embedding_df.reset_index(drop=True, inplace=True)
    return embedding_df

def _build_text_datasets(current_texts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, ColumnMapping]:
    reference_texts = pd.Series(REFERENCE_QUESTIONS, dtype=str)

    current_data = pd.DataFrame({"question": current_texts})
    reference_data = pd.DataFrame({"question": reference_texts})

    column_mapping = ColumnMapping()
    column_mapping.text_features = ["question"]

    return reference_data, current_data, column_mapping

def _build_embedding_datasets(current_texts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, ColumnMapping]:
    reference_texts = pd.Series(REFERENCE_QUESTIONS, dtype=str)

    current_data = pd.DataFrame({"question": current_texts})
    reference_data = pd.DataFrame({"question": reference_texts})

    current_embeddings = _embed_questions(current_texts)
    reference_embeddings = _embed_questions(reference_texts)

    current_full = pd.concat([current_data, current_embeddings], axis=1)
    reference_full = pd.concat([reference_data, reference_embeddings], axis=1)

    embedding_columns = [column for column in current_full.columns if column.startswith("emb_")]

    column_mapping = ColumnMapping()
    column_mapping.text_features = ["question"]
    column_mapping.embeddings = {"question_embeddings": embedding_columns}

    return reference_full, current_full, column_mapping

def generate_drift_report(prompt_version=None, window_size=None) -> bool:
    current_texts = _load_current_questions(prompt_version=prompt_version, window_size=window_size)

    if len(current_texts) < MIN_REPORT_QUERIES:
        print(f"Pas assez de requêtes utilisateur pour générer le rapport Evidently AI. Minimum requis : {MIN_REPORT_QUERIES}.")
        return False

    if _can_use_embeddings(current_texts):
        reference_data, current_data, column_mapping = _build_embedding_datasets(current_texts)
        metrics = [
            ColumnSummaryMetric(column_name="question"),
            TextDescriptorsDriftMetric(
                column_name="question",
                stattest_threshold=0.05,
                descriptors={
                    "Longueur (nb caractères)": TextLength(),
                    "Complexité (nb phrases)": SentenceCount(),
                    "Bruit (caractères spéciaux %)": NonLetterCharacterPercentage(),
                },
            ),
            EmbeddingsDriftMetric("question_embeddings"),
        ]
    else:
        reference_data, current_data, column_mapping = _build_text_datasets(current_texts)
        metrics = [
            ColumnSummaryMetric(column_name="question"),
            TextDescriptorsDriftMetric(
                column_name="question",
                stattest_threshold=0.05,
                descriptors={
                    "Longueur (nb caractères)": TextLength(),
                    "Complexité (nb phrases)": SentenceCount(),
                    "Bruit (caractères spéciaux %)": NonLetterCharacterPercentage(),
                },
            ),
        ]


    report = Report(metrics=metrics)

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    report.save_html(DRIFT_REPORT_PATH)

    print(f"Rapport Evidently AI généré : {DRIFT_REPORT_PATH}")
    return True

def get_semantic_drift(prompt_version=None, window_size=None) -> dict:
    current_texts = _load_current_questions(prompt_version=prompt_version, window_size=window_size)

    if len(current_texts) < MIN_STATUS_QUERIES:
        return {
            "available": False,
            "drift_detected": False,
            "score": None,
            "message": f"Minimum {MIN_STATUS_QUERIES} requêtes requises pour calculer la dérive sémantique.",
        }

    if not _can_use_embeddings(current_texts):
        required_current = MIN_EMBEDDING_TOTAL_SAMPLES - len(REFERENCE_QUESTIONS)
        return {
            "available": False,
            "drift_detected": False,
            "score": None,
            "message": f"Dérive sémantique indisponible (minimum {required_current} requêtes). Les autres indicateurs restent actifs.",
        }

    try:
        reference_data, current_data, column_mapping = _build_embedding_datasets(current_texts)

        report = Report(metrics=[EmbeddingsDriftMetric("question_embeddings")])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        result = report.as_dict()
        metric_result = result["metrics"][0]["result"]

        return {
            "available": True,
            "drift_detected": bool(metric_result.get("drift_detected", False)),
            "score": metric_result.get("drift_score"),
            "message": "Dérive sémantique calculée.",
        }
    except Exception as error:
        return {
            "available": False,
            "drift_detected": False,
            "score": None,
            "message": f"Dérive sémantique indisponible : {error}",
        }

def get_refusal_rate(metrics_df: pd.DataFrame) -> float:
    if metrics_df.empty or "answer" not in metrics_df.columns:
        return 0.0

    answers = metrics_df["answer"].fillna("").astype(str)

    refusal_count = answers.str.contains(
        "Je ne sais pas|aucun document pertinent|aucune source",
        case=False,
        regex=True,
    ).sum()

    return refusal_count / len(answers) if len(answers) > 0 else 0.0

def check_drift_status(prompt_version=None, window_size=None) -> dict:
    metrics_df = _load_metrics(prompt_version=prompt_version, window_size=window_size)

    if metrics_df.empty:
        return {
            "drift_detected": False,
            "level": "INFO",
            "message": "Pas encore de données d'historique pour analyser la dérive.",
        }

    if len(metrics_df) < MIN_STATUS_QUERIES:
        return {
            "drift_detected": False,
            "level": "INFO",
            "message": f"Collecte de données en cours. Minimum requis : {MIN_STATUS_QUERIES} requêtes.",
        }

    refusal_rate = get_refusal_rate(metrics_df)
    semantic = get_semantic_drift(prompt_version=prompt_version, window_size=window_size)

    semantic_message = ""

    if semantic["available"] and semantic["score"] is not None:
        semantic_label = "oui" if semantic["drift_detected"] else "non"
        semantic_message = f" Dérive sémantique : {semantic_label}, score : {semantic['score']:.3f}."
    elif semantic["message"]:
        semantic_message = f" {semantic['message']}"

    if refusal_rate >= ERROR_REFUSAL_RATE or semantic["drift_detected"]:
        return {
            "drift_detected": True,
            "level": "ERROR",
            "message": f"Alerte dérive détectée. {refusal_rate * 100:.0f}% des {len(metrics_df)} requêtes sélectionnées n'ont pas obtenu de réponse exploitable.{semantic_message}",
        }

    if refusal_rate >= WARNING_REFUSAL_RATE:
        return {
            "drift_detected": True,
            "level": "WARNING",
            "message": f"Risque de dérive. {refusal_rate * 100:.0f}% des {len(metrics_df)} requêtes sélectionnées n'ont pas obtenu de réponse exploitable.{semantic_message}",
        }

    return {
        "drift_detected": False,
        "level": "OK",
        "message": f"Aucune dérive significative détectée sur les {len(metrics_df)} requêtes sélectionnées. Taux de refus : {refusal_rate * 100:.0f}%.{semantic_message}",
    }

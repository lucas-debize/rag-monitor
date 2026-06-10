import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient
import shutil

from src.ingestion import CHROMA_DIR
from src.metrics_logger import log_query, load_metrics
from src.rag_pipeline import build_chain, extract_sources
from src.prompts import list_versions, get_prompt
from src.ingestion import chunk_documents, build_vectorstore, DOCUMENTS_DIR

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-monitor")

RAG_MONITOR_THRESHOLD = float(os.getenv("RAG_MONITOR_THRESHOLD", "0.80"))
ANSWER_CORRECTNESS_THRESHOLD = float(os.getenv("ANSWER_CORRECTNESS_THRESHOLD", "0.80"))
CITATION_SCORE_THRESHOLD = float(os.getenv("CITATION_SCORE_THRESHOLD", "0.80"))
REFUSAL_SCORE_THRESHOLD = float(os.getenv("REFUSAL_SCORE_THRESHOLD", "0.90"))

BUSINESS_METRICS = [
    "rag_monitor_score",
    "factual_answer_correctness",
    "factual_context_recall",
    "factual_citation_score",
    "factual_hallucination_score",
    "out_of_scope_refusal_score",
    "out_of_scope_citation_score",
    "out_of_scope_hallucination_score",
]

DISPLAY_METRICS = {
    "rag_monitor_score": "Score global",
    "factual_answer_correctness": "Exactitude",
    "factual_context_recall": "Rappel contexte",
    "factual_citation_score": "Citations",
    "factual_hallucination_score": "Anti-hallucination factuelle",
    "out_of_scope_refusal_score": "Refus hors-scope",
    "out_of_scope_citation_score": "Citations hors-scope",
    "out_of_scope_hallucination_score": "Anti-hallucination hors-scope",
}

st.set_page_config(page_title="RAG Monitor", page_icon="🤖", layout="wide")

@st.cache_resource(show_spinner=False)
def reset_vectorstore_on_startup():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    if os.path.exists(DOCUMENTS_DIR):
        for file in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    from src.ingestion import get_vectorstore

    vectorstore = get_vectorstore()

    try:
        existing_ids = vectorstore._collection.get()["ids"]
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
    except Exception as e:
        print(f"Avertissement reset startup : {e}")

    return True

@st.cache_resource(show_spinner="Initialisation de la chaîne RAG...")
def get_chain(prompt_version: str):
    chain, retriever, prompt_def = build_chain(prompt_version)
    return chain, retriever, prompt_def


@st.cache_data(ttl=30, show_spinner=False)
def get_metrics():
    return load_metrics()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_runs():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        return pd.DataFrame()

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )

    rows = []

    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "prompt_version": run.data.params.get("prompt_version", run.data.tags.get("prompt_version", "n/a")),
            "testset_size": run.data.params.get("testset_size", ""),
            "with_ragas": run.data.params.get("with_ragas", ""),
        }

        for metric in BUSINESS_METRICS:
            row[metric] = run.data.metrics.get(metric)

        rows.append(row)

    return pd.DataFrame(rows)


def format_score(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def score_status(value, threshold):
    if value is None or pd.isna(value):
        return "unknown"
    return "ok" if float(value) >= threshold else "ko"


def render_score_alert(label, value, threshold):
    status = score_status(value, threshold)
    formatted = format_score(value)

    if status == "ok":
        st.success(f"{label} : {formatted} ≥ {threshold:.2f}")
    elif status == "ko":
        st.error(f"{label} : {formatted} < {threshold:.2f}")
    else:
        st.warning(f"{label} : non disponible")


def render_sources(docs, sources, latency):
    with st.expander(f"📚 Sources utilisées ({len(sources)}) — {latency:.2f}s"):
        if not docs:
            st.info("Aucune source récupérée.")
            return

        for index, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "inconnu").split("/")[-1]
            page = doc.metadata.get("page", "")
            page_label = f"page {page}" if page != "" else "page inconnue"

            st.markdown(f"**{index}. {source} — {page_label}**")
            st.text(doc.page_content[:700])


def ingest_uploaded_files(uploaded_files):
    if os.path.exists(DOCUMENTS_DIR):
        for file in os.listdir(DOCUMENTS_DIR):
            os.remove(os.path.join(DOCUMENTS_DIR, file))
    else:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    saved_paths = []

    for uploaded_file in uploaded_files:
        target_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)

        with open(target_path, "wb") as output:
            output.write(uploaded_file.getbuffer())

        saved_paths.append(target_path)

    from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader

    documents = []

    for path in saved_paths:
        if path.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif path.endswith(".md"):
            documents.extend(UnstructuredMarkdownLoader(path).load())

    chunks = chunk_documents(documents)
    build_vectorstore(chunks)

    get_chain.clear()
    get_metrics.clear()

    return len(documents), len(chunks)

def page_chat():
    st.title("💬 Chat documentaire")
    st.caption("Pose une question sur les documents indexés. Le modèle répond uniquement à partir du contexte récupéré.")

    with st.sidebar:
        st.header("Configuration RAG")

        versions = list_versions()
        default_index = versions.index("v3") if "v3" in versions else len(versions) - 1
        prompt_version = st.selectbox("Version du prompt", versions, index=default_index)

        prompt_def = get_prompt(prompt_version)

        st.info(prompt_def["description"])

        with st.expander("Voir le template du prompt"):
            st.code(prompt_def["template"])

        st.divider()

        st.header("Documents")

        uploaded_files = st.file_uploader(
            "Ajouter des fichiers PDF ou Markdown",
            type=["pdf", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Ingérer les documents"):
            with st.spinner("Ingestion en cours..."):
                documents_count, chunks_count = ingest_uploaded_files(uploaded_files)

            st.success(f"{len(uploaded_files)} fichier(s), {documents_count} document(s), {chunks_count} chunk(s) ingéré(s).")

        st.divider()

        if st.button("Effacer l'historique du chat"):
            st.session_state.history = []
            st.rerun()

    if "history" not in st.session_state:
        st.session_state.history = []

    chain, retriever, _ = get_chain(prompt_version)

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])

        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            render_sources(entry["docs"], entry["sources"], entry["latency"])

    question = st.chat_input("Pose ta question...")

    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des passages pertinents et génération de la réponse..."):
            start = time.perf_counter()
            docs = retriever.invoke(question)

            if not docs:
                answer = "Je ne sais pas répondre à cette question car aucun document pertinent n'a été trouvé."
                latency = time.perf_counter() - start
                sources = []
            else:
                answer = chain.invoke(question)
                latency = time.perf_counter() - start
                sources = extract_sources(docs)
            latency = time.perf_counter() - start
            sources = extract_sources(docs)

        log_query(
            question=question,
            answer=answer,
            latency_s=latency,
            sources=sources,
            prompt_version=prompt_version,
        )

        get_metrics.clear()

        st.markdown(answer)
        render_sources(docs, sources, latency)

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "docs": docs,
            "latency": latency,
            "prompt_version": prompt_version,
        }
    )


def page_dashboard():
    st.title("📊 Dashboard monitoring")
    st.caption("Suivi des requêtes utilisateur, performances du RAG et résultats d'évaluation MLflow.")

    metrics_df = get_metrics()
    runs_df = fetch_runs()

    tab_realtime, tab_quality, tab_history = st.tabs(
        [
            "Temps réel",
            "Qualité RAG",
            "Historique",
        ]
    )

    with tab_realtime:
        st.subheader("Métriques temps réel")

        if metrics_df.empty:
            st.info("Aucune requête enregistrée. Pose une question dans l'onglet Chat pour alimenter le dashboard.")
        else:
            with st.sidebar:
                st.header("Filtres dashboard")

                versions = ["Toutes"] + sorted(metrics_df["prompt_version"].dropna().unique().tolist())
                selected_version = st.selectbox("Version du prompt", versions, index=0)

                last_n = st.slider("Nombre de requêtes affichées", 5, 500, 50, step=5)

            filtered_df = metrics_df.copy()

            if selected_version != "Toutes":
                filtered_df = filtered_df[filtered_df["prompt_version"] == selected_version]

            filtered_df = filtered_df.sort_values("timestamp").tail(last_n)

            if filtered_df.empty:
                st.warning("Aucune donnée disponible pour ce filtre.")
            else:
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Requêtes", len(filtered_df))
                c2.metric("Latence moyenne", f"{filtered_df['latency_s'].mean():.2f}s")
                c3.metric("Longueur moyenne", f"{filtered_df['answer_length'].mean():.0f} caractères")
                c4.metric("Sources moyennes", f"{filtered_df['num_sources'].mean():.2f}")

                st.divider()

                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Latence par requête")
                    fig_latency = px.line(
                        filtered_df,
                        x="timestamp",
                        y="latency_s",
                        color="prompt_version",
                        markers=True,
                    )
                    st.plotly_chart(fig_latency, use_container_width=True)

                with col_right:
                    st.subheader("Longueur des réponses")
                    fig_length = px.histogram(
                        filtered_df,
                        x="answer_length",
                        color="prompt_version",
                        nbins=20,
                    )
                    st.plotly_chart(fig_length, use_container_width=True)

                st.subheader("Sources récupérées")
                fig_sources = px.histogram(
                    filtered_df,
                    x="num_sources",
                    color="prompt_version",
                    nbins=10,
                )
                st.plotly_chart(fig_sources, use_container_width=True)

    with tab_quality:
        st.subheader("Qualité RAG via MLflow")

        if runs_df.empty:
            st.info("Aucun run MLflow trouvé pour l'expérience configurée.")
        else:
            eval_df = runs_df[runs_df["run_name"].astype(str).str.startswith("eval-", na=False)].copy()

            if eval_df.empty:
                st.warning("Aucun run d'évaluation trouvé. Lance une évaluation avec run_evaluation.py.")
            else:
                latest_eval = eval_df.sort_values("start_time", ascending=False).iloc[0]

                st.markdown("#### Dernière évaluation")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Prompt", latest_eval["prompt_version"])
                c2.metric("Score global", format_score(latest_eval["rag_monitor_score"]))
                c3.metric("Exactitude", format_score(latest_eval["factual_answer_correctness"]))
                c4.metric("Citations", format_score(latest_eval["factual_citation_score"]))

                render_score_alert("Score global RAG Monitor", latest_eval["rag_monitor_score"], RAG_MONITOR_THRESHOLD)
                render_score_alert("Exactitude factuelle", latest_eval["factual_answer_correctness"], ANSWER_CORRECTNESS_THRESHOLD)
                render_score_alert("Score de citation", latest_eval["factual_citation_score"], CITATION_SCORE_THRESHOLD)
                render_score_alert("Refus hors-scope", latest_eval["out_of_scope_refusal_score"], REFUSAL_SCORE_THRESHOLD)

                st.divider()

                st.markdown("#### Comparaison des prompts")

                display_columns = [
                    "start_time",
                    "run_name",
                    "prompt_version",
                    "status",
                    "testset_size",
                ] + BUSINESS_METRICS

                comparison_df = eval_df[display_columns].sort_values("start_time", ascending=False)

                renamed_df = comparison_df.rename(columns=DISPLAY_METRICS)

                st.dataframe(renamed_df, use_container_width=True)

                chart_df = eval_df.dropna(subset=["rag_monitor_score"])

                if not chart_df.empty:
                    st.markdown("#### Score global par version")
                    fig_score = px.bar(
                        chart_df.sort_values("start_time"),
                        x="prompt_version",
                        y="rag_monitor_score",
                        color="prompt_version",
                        hover_data=["run_name", "start_time"],
                    )
                    fig_score.add_hline(y=RAG_MONITOR_THRESHOLD, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_score, use_container_width=True)

                detail_metrics = [
                    "factual_answer_correctness",
                    "factual_context_recall",
                    "factual_citation_score",
                    "out_of_scope_refusal_score",
                    "out_of_scope_hallucination_score",
                ]

                detail_df = eval_df[["prompt_version"] + detail_metrics].dropna(how="all", subset=detail_metrics)

                if not detail_df.empty:
                    melted_df = detail_df.melt(
                        id_vars=["prompt_version"],
                        value_vars=detail_metrics,
                        var_name="metric",
                        value_name="score",
                    )
                    melted_df["metric"] = melted_df["metric"].map(DISPLAY_METRICS)

                    st.markdown("#### Détail des scores métier")
                    fig_detail = px.bar(
                        melted_df,
                        x="prompt_version",
                        y="score",
                        color="metric",
                        barmode="group",
                    )
                    st.plotly_chart(fig_detail, use_container_width=True)

    with tab_history:
        st.subheader("Dernières requêtes utilisateur")

        if metrics_df.empty:
            st.info("Aucune requête enregistrée.")
        else:
            history_df = metrics_df.sort_values("timestamp", ascending=False).copy()

            st.dataframe(
                history_df[
                    [
                        "timestamp",
                        "prompt_version",
                        "question",
                        "answer",
                        "latency_s",
                        "answer_length",
                        "num_sources",
                        "sources",
                    ]
                ],
                use_container_width=True,
            )


PAGES = {
    "💬 Chat": page_chat,
    "📊 Dashboard": page_dashboard,
}


def main():
    reset_vectorstore_on_startup()
    st.sidebar.title("RAG Monitor")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption(f"MLflow : {MLFLOW_URI}")
    PAGES[choice]()


if __name__ == "__main__":
    main()

import os
import time
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient

from src.rag_pipeline import build_chain, extract_sources
from src.prompts import list_versions, get_prompt
from src.ingestion import load_documents, chunk_documents, build_vectorstore, DOCUMENTS_DIR

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-monitor")

st.set_page_config(page_title="RAG Monitor", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Initialisation de la chaîne RAG...")
def get_chain(prompt_version: str):
    chain, retriever, prompt_def = build_chain(prompt_version)
    return chain, retriever, prompt_def


def page_chat():
    st.title("💬 Chat RAG")

    with st.sidebar:
        st.header("Configuration")
        versions = list_versions()
        prompt_version = st.selectbox("Version du prompt", versions, index=len(versions) - 1)

        st.subheader("Prompt courant")
        with st.expander("Voir le template"):
            st.code(get_prompt(prompt_version)["template"])

        st.subheader("Ajouter des documents")
        uploaded = st.file_uploader(
            "PDF ou Markdown",
            type=["pdf", "md"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingérer"):
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            new_paths = []
            for f in uploaded:
                target = os.path.join(DOCUMENTS_DIR, f.name)
                with open(target, "wb") as out:
                    out.write(f.getbuffer())
                new_paths.append(target)
            with st.spinner("Ingestion en cours..."):
                from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
                docs = []
                for p in new_paths:
                    if p.endswith(".pdf"):
                        docs.extend(PyPDFLoader(p).load())
                    elif p.endswith(".md"):
                        docs.extend(UnstructuredMarkdownLoader(p).load())
                chunks = chunk_documents(docs)
                build_vectorstore(chunks)
                st.cache_resource.clear()
            st.success(f"{len(uploaded)} document(s) ingéré(s). Cache rechargé.")


        if st.button("🗑️ Effacer l'historique"):
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
            with st.expander(f"📚 Sources ({len(entry['sources'])}) — {entry['latency']:.2f}s"):
                for i, doc in enumerate(entry["docs"]):
                    src = doc.metadata.get("source", "inconnu").split("/")[-1]
                    page = doc.metadata.get("page", "")
                    st.markdown(f"**{i+1}. {src}** {f'(page {page})' if page != '' else ''}")
                    st.text(doc.page_content[:400] + "...")

    question = st.chat_input("Pose ta question...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération..."):
                t0 = time.perf_counter()
                docs = retriever.invoke(question)
                answer = chain.invoke(question)
                latency = time.perf_counter() - t0
                sources = extract_sources(docs)
            st.markdown(answer)
            with st.expander(f"📚 Sources ({len(sources)}) — {latency:.2f}s"):
                for i, doc in enumerate(docs):
                    src = doc.metadata.get("source", "inconnu").split("/")[-1]
                    page = doc.metadata.get("page", "")
                    st.markdown(f"**{i+1}. {src}** {f'(page {page})' if page != '' else ''}")
                    st.text(doc.page_content[:400] + "...")

        st.session_state.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "docs": docs,
            "latency": latency,
            "prompt_version": prompt_version,
        })


@st.cache_data(ttl=30, show_spinner=False)
def fetch_runs():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=200)
    rows = []
    for r in runs:
        rows.append({
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "status": r.info.status,
            "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
            "prompt_version": r.data.tags.get("prompt_version", "n/a"),
            "avg_latency_s": r.data.metrics.get("avg_latency_s"),
            "avg_answer_length": r.data.metrics.get("avg_answer_length"),
            "avg_num_sources": r.data.metrics.get("avg_num_sources"),
            "faithfulness": r.data.metrics.get("faithfulness"),
            "answer_relevancy": r.data.metrics.get("answer_relevancy"),
            "context_precision": r.data.metrics.get("context_precision"),
        })
    return pd.DataFrame(rows)


def page_dashboard():
    st.title("📊 Dashboard — Monitoring & MLflow")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Rafraîchir"):
            st.cache_data.clear()

    df = fetch_runs()
    if df.empty:
        st.warning("Aucun run MLflow trouvé. Lance une requête dans le Chat ou exécute une évaluation.")
        return

    st.subheader("Vue d'ensemble")
    last = df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs total", len(df))
    m2.metric("Latence moy. (dernier run)", f"{last['avg_latency_s']:.2f}s" if pd.notna(last["avg_latency_s"]) else "—")
    m3.metric("Faithfulness (dernier)", f"{last['faithfulness']:.3f}" if pd.notna(last["faithfulness"]) else "—")
    m4.metric("Relevancy (dernier)", f"{last['answer_relevancy']:.3f}" if pd.notna(last["answer_relevancy"]) else "—")

    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.75"))
    if pd.notna(last["faithfulness"]) and last["faithfulness"] < threshold:
        st.error(f"⚠️ ALERTE : faithfulness ({last['faithfulness']:.3f}) sous le seuil ({threshold})")
    elif pd.notna(last["faithfulness"]):
        st.success(f"✅ Faithfulness OK ({last['faithfulness']:.3f} ≥ {threshold})")

    st.subheader("Comparaison par version de prompt")
    eval_df = df.dropna(subset=["faithfulness"])
    if not eval_df.empty:
        agg = eval_df.groupby("prompt_version")[["faithfulness", "answer_relevancy", "context_precision"]].mean().reset_index()
        melted = agg.melt(id_vars="prompt_version", var_name="metric", value_name="score")
        fig = px.bar(melted, x="prompt_version", y="score", color="metric", barmode="group", range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune métrique RAGAS disponible. Lance `run_evaluation.py`.")

    st.subheader("Évolution des latences")
    lat_df = df.dropna(subset=["avg_latency_s"]).sort_values("start_time")
    if not lat_df.empty:
        fig2 = px.line(lat_df, x="start_time", y="avg_latency_s", color="prompt_version", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tous les runs")
    st.dataframe(df, use_container_width=True)


PAGES = {
    "💬 Chat": page_chat,
    "📊 Dashboard": page_dashboard,
}


def main():
    st.sidebar.title("RAG Monitor")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"MLflow UI : [{MLFLOW_URI}]({MLFLOW_URI})")
    PAGES[choice]()


if __name__ == "__main__":
    main()

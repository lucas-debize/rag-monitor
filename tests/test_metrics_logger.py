import csv
import importlib


def test_log_query_creates_metrics_file(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    monkeypatch.setenv("METRICS_DIR", str(metrics_dir))

    import src.metrics_logger as metrics_logger

    importlib.reload(metrics_logger)

    metrics_logger.log_query(
        question="Quelle est la capitale de la France ?",
        answer="Paris est la capitale de la France.",
        latency_s=1.23456,
        sources=["document.pdf"],
        prompt_version="v3",
    )

    metrics_file = metrics_dir / "queries.csv"

    assert metrics_file.exists()

    with open(metrics_file, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["prompt_version"] == "v3"
    assert rows[0]["question"] == "Quelle est la capitale de la France ?"
    assert rows[0]["answer"] == "Paris est la capitale de la France."
    assert rows[0]["latency_s"] == "1.2346"
    assert rows[0]["answer_length"] == str(len("Paris est la capitale de la France."))
    assert rows[0]["num_sources"] == "1"
    assert rows[0]["sources"] == "document.pdf"


def test_load_metrics_returns_empty_dataframe_when_file_does_not_exist(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    monkeypatch.setenv("METRICS_DIR", str(metrics_dir))

    import src.metrics_logger as metrics_logger

    importlib.reload(metrics_logger)

    df = metrics_logger.load_metrics()

    assert df.empty
    assert list(df.columns) == metrics_logger.FIELDS


def test_load_metrics_reads_existing_metrics(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    monkeypatch.setenv("METRICS_DIR", str(metrics_dir))

    import src.metrics_logger as metrics_logger

    importlib.reload(metrics_logger)

    metrics_logger.log_query(
        question="Question test",
        answer="Réponse test",
        latency_s=0.5,
        sources=["source.md", "source.pdf"],
        prompt_version="v2",
    )

    df = metrics_logger.load_metrics()

    assert len(df) == 1
    assert df.iloc[0]["prompt_version"] == "v2"
    assert df.iloc[0]["question"] == "Question test"
    assert df.iloc[0]["answer"] == "Réponse test"
    assert df.iloc[0]["num_sources"] == 2
    assert df.iloc[0]["sources"] == "source.md|source.pdf"

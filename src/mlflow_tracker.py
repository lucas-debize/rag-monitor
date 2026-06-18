import os
import time
import mlflow

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://0.0.0.0:5000/")
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-monitor")


def init_mlflow(experiment_name: str = DEFAULT_EXPERIMENT, tracking_uri: str = DEFAULT_TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment : {experiment_name}")


def start_run(run_name: str, params: dict, prompt_template: str, prompt_version: str):
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    mlflow.set_tag("prompt_version", prompt_version)
    mlflow.log_text(prompt_template, artifact_file=f"prompts/{prompt_version}.txt")
    return run


def log_question(index: int, question: str, answer: str, latency_s: float, num_sources: int, sources: list):
    mlflow.log_metric("latency_s", latency_s, step=index)
    mlflow.log_metric("answer_length", len(answer), step=index)
    mlflow.log_metric("num_sources", num_sources, step=index)
    artifact = (
        f"Q: {question}\n\n"
        f"A: {answer}\n\n"
        f"Latency: {latency_s:.2f}s\n"
        f"Sources ({num_sources}): {', '.join(sources)}\n"
    )
    mlflow.log_text(artifact, artifact_file=f"qa/q{index:02d}.txt")


def log_aggregates(latencies: list, answer_lengths: list, num_sources_list: list):
    if latencies:
        mlflow.log_metric("avg_latency_s", sum(latencies) / len(latencies))
        mlflow.log_metric("max_latency_s", max(latencies))
    if answer_lengths:
        mlflow.log_metric("avg_answer_length", sum(answer_lengths) / len(answer_lengths))
    if num_sources_list:
        mlflow.log_metric("avg_num_sources", sum(num_sources_list) / len(num_sources_list))


def end_run():
    mlflow.end_run()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

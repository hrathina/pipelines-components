"""Evaluate speculative decoding acceptance and speedup."""

from kfp import dsl


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "requests>=2.32.0"],
    install_kfp_package=False,
)
def evaluate_speculator(
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    verifier_model: str,
    draft_model_path: str,
    evaluation_dataset_uri: str,
    evaluation_speculator_type: str,
    evaluation_max_samples: int = 80,
    evaluation_max_tokens: int = 256,
    evaluation_temperature: float = 0.0,
    evaluation_num_speculative_tokens: int = 5,
    evaluation_gpu_memory_utilization: float = 0.9,
    evaluation_port: int = 8000,
    evaluation_startup_timeout_seconds: int = 900,
    evaluation_request_timeout_seconds: int = 600,
    evaluation_sample_seed: int = 42,
    evaluation_model_mount_path: str = "/mnt/persistent",
    evaluation_dataset_mount_path: str = "/mnt/persistent",
) -> None:
    """Evaluate a draft model with verifier-only and speculative vLLM servers.

    The component starts a verifier-only server to establish a latency baseline,
    then restarts vLLM with the draft model configured for speculative decoding.
    The evaluation dataset must be a JSONL file referenced by
    ``evaluation_dataset_uri`` with a ``prompt``, ``messages``, or
    ``conversations`` field, or a string value on each line. At most
    ``evaluation_max_samples`` records are selected deterministically. The output Metrics artifact contains
    ``acceptance_rate``, ``acceptance_length``, and ``speedup`` for Model
    Registry integration.

    Args:
        output_metrics: KFP Metrics artifact for acceptance and speedup values.
        output_results: JSON artifact containing raw timing and acceptance results.
        verifier_model: Local or PVC-relative path to the verifier model.
        draft_model_path: Local path to the trained draft model on the mounted PVC.
        evaluation_dataset_uri: Local or PVC-relative path to the evaluation JSONL.
        evaluation_max_samples: Maximum number of prompts to evaluate.
        evaluation_max_tokens: Maximum completion tokens per prompt.
        evaluation_temperature: Sampling temperature sent to both servers.
        evaluation_num_speculative_tokens: Number of draft tokens proposed per step.
        evaluation_speculator_type: Speculator method used by the draft checkpoint.
        evaluation_gpu_memory_utilization: vLLM GPU memory utilization fraction.
        evaluation_port: Local port used by the temporary vLLM servers.
        evaluation_startup_timeout_seconds: Maximum time to wait for vLLM startup.
        evaluation_request_timeout_seconds: Maximum time allowed for one request.
        evaluation_sample_seed: Seed used for deterministic dataset sampling.
        evaluation_model_mount_path: Local mount prefix for a relative verifier path.
        evaluation_dataset_mount_path: Local mount prefix for a relative dataset path.
    """
    import json
    import os
    import random
    import signal
    import subprocess
    import time
    from pathlib import Path

    import requests

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    verifier_path = verifier_model
    if verifier_path.startswith("pvc://"):
        verifier_path = verifier_path.removeprefix("pvc://").split("/", 1)[-1]
    if not os.path.isabs(verifier_path):
        verifier_path = os.path.join(evaluation_model_mount_path, verifier_path)
    draft_path = draft_model_path
    if not os.path.isabs(draft_path):
        draft_path = os.path.join(evaluation_model_mount_path, draft_path)
    dataset_path = evaluation_dataset_uri
    if dataset_path.startswith("pvc://"):
        dataset_path = dataset_path.removeprefix("pvc://")
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(evaluation_dataset_mount_path, dataset_path)
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        candidates = sorted(dataset_path.glob("*.jsonl"))
        if not candidates:
            raise ValueError(f"No JSONL dataset found in {dataset_path}")
        dataset_path = candidates[0]

    prompts = []
    with dataset_path.open() as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, str):
                prompt = record
            else:
                prompt = record.get("prompt")
                if not prompt:
                    messages = record.get("messages") or record.get("conversations") or []
                    user_messages = [
                        message for message in messages if message.get("role", message.get("from")) in ("user", "human")
                    ]
                    if user_messages:
                        prompt = user_messages[-1].get("content", user_messages[-1].get("value"))
            if not prompt:
                raise ValueError("Each evaluation record must contain a prompt or user message")
            prompts.append(prompt)
    if not prompts:
        raise ValueError("The evaluation dataset contains no prompts")
    if evaluation_max_samples <= 0:
        raise ValueError("evaluation_max_samples must be greater than zero")
    if len(prompts) > evaluation_max_samples:
        prompts = random.Random(evaluation_sample_seed).sample(prompts, evaluation_max_samples)

    endpoint = f"http://127.0.0.1:{evaluation_port}"
    process = None

    def stop_server() -> None:
        nonlocal process
        if process is None:
            return
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process = None

    def start_server(speculative: bool) -> None:
        nonlocal process
        command = [
            "vllm",
            "serve",
            verifier_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(evaluation_port),
            "--gpu-memory-utilization",
            str(evaluation_gpu_memory_utilization),
        ]
        if speculative:
            methods = {"eagle3": "eagle3", "dflash": "dflash", "mtp": "mtp", "peagle": "eagle3"}
            method = methods.get(evaluation_speculator_type.lower())
            if method is None:
                raise ValueError(f"Unsupported evaluation_speculator_type: {evaluation_speculator_type}")
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(
                        {
                            "method": method,
                            "model": draft_path,
                            "num_speculative_tokens": evaluation_num_speculative_tokens,
                        }
                    ),
                ]
            )
        process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + evaluation_startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM did not become ready within {evaluation_startup_timeout_seconds} seconds")

    def run_requests() -> tuple[float, int]:
        started = time.perf_counter()
        output_tokens = 0
        for prompt in prompts:
            response = requests.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": verifier_path,
                    "prompt": prompt,
                    "max_tokens": evaluation_max_tokens,
                    "temperature": evaluation_temperature,
                },
                timeout=evaluation_request_timeout_seconds,
            )
            response.raise_for_status()
            usage = response.json().get("usage", {})
            output_tokens += int(usage.get("completion_tokens", 0))
        return time.perf_counter() - started, output_tokens

    def read_speculative_metrics() -> dict[str, float]:
        response = requests.get(f"{endpoint}/metrics", timeout=30)
        response.raise_for_status()
        counters = {}
        for line in response.text.splitlines():
            if line.startswith("#") or "vllm:spec_decode" not in line:
                continue
            metric, _, value = line.partition(" ")
            metric = metric.split("{", 1)[0].removesuffix("_total")
            try:
                counters[metric] = counters.get(metric, 0.0) + float(value)
            except ValueError:
                continue
        draft_tokens = counters.get("vllm:spec_decode_num_draft_tokens", 0.0)
        accepted_tokens = counters.get("vllm:spec_decode_num_accepted_tokens", 0.0)
        drafts = counters.get("vllm:spec_decode_num_drafts", 0.0)
        return {
            "num_drafts": drafts,
            "num_draft_tokens": draft_tokens,
            "num_accepted_tokens": accepted_tokens,
            "acceptance_rate": accepted_tokens / draft_tokens if draft_tokens else 0.0,
            "acceptance_length": 1.0 + accepted_tokens / drafts if drafts else 0.0,
        }

    try:
        start_server(speculative=False)
        baseline_seconds, baseline_tokens = run_requests()
        stop_server()

        start_server(speculative=True)
        speculative_seconds, speculative_tokens = run_requests()
        spec_metrics = read_speculative_metrics()
    finally:
        stop_server()

    results = {
        **spec_metrics,
        "baseline_seconds": baseline_seconds,
        "speculative_seconds": speculative_seconds,
        "baseline_output_tokens": baseline_tokens,
        "speculative_output_tokens": speculative_tokens,
        "speedup": baseline_seconds / speculative_seconds if speculative_seconds else 0.0,
        "requests": len(prompts),
    }
    with Path(output_results.path).open("w") as results_file:
        json.dump(results, results_file, indent=2)
    for name in ("acceptance_rate", "acceptance_length", "speedup"):
        output_metrics.log_metric(name, results[name])

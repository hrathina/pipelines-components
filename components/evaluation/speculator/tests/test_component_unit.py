"""Unit tests for the Speculator evaluation component."""

import inspect

from ..component import evaluate_speculator


def test_component_function_exists():
    """The component is exposed as a KFP Python function."""
    assert callable(evaluate_speculator)
    assert hasattr(evaluate_speculator, "python_func")


def test_component_has_expected_parameters():
    """The component exposes model, dataset, and benchmark parameters."""
    params = inspect.signature(evaluate_speculator.python_func).parameters
    for name in (
        "output_metrics",
        "output_results",
        "verifier_model",
        "draft_model_path",
        "evaluation_dataset_uri",
        "evaluation_max_samples",
        "evaluation_max_tokens",
        "evaluation_num_speculative_tokens",
        "evaluation_speculator_type",
        "evaluation_sample_seed",
        "evaluation_max_failure_rate",
    ):
        assert name in params


def test_component_defaults():
    """The component uses safe defaults for a small evaluation run."""
    params = inspect.signature(evaluate_speculator.python_func).parameters
    assert params["evaluation_max_samples"].default == 80
    assert params["evaluation_max_tokens"].default == 256
    assert params["evaluation_temperature"].default == 0.0
    assert params["evaluation_num_speculative_tokens"].default == 5
    assert params["evaluation_sample_seed"].default == 42
    assert params["evaluation_max_failure_rate"].default == 0.1


def test_component_validates_evaluation_bounds():
    """The evaluator validates token and speedup-related safeguards."""
    source = inspect.getsource(evaluate_speculator.python_func)
    assert "evaluation_max_tokens must be greater than zero" in source
    assert "sanity bound" in source


def test_component_documents_metrics_and_server_workflow():
    """The component documents its vLLM metrics and execution workflow."""
    docstring = evaluate_speculator.python_func.__doc__.lower()
    assert "acceptance_rate" in docstring
    assert "speedup" in docstring
    assert "vllm" in docstring
    assert "messages" in docstring
    assert "evaluation_max_samples" in docstring


def test_component_parses_prometheus_values_with_timestamps():
    """The evaluator accounts for optional Prometheus timestamps."""
    source = inspect.getsource(evaluate_speculator.python_func)
    assert "parts = line.split()" in source
    assert "value_str = parts[0], parts[1]" in source
    assert "vLLM speculative-decoding counters are missing" in source
    assert "Raw /metrics response preview" in source


def test_component_strips_pvc_claim_from_dataset_uri():
    """Dataset PVC URIs use the path after the PVC claim name."""
    source = inspect.getsource(evaluate_speculator.python_func)
    assert '.removeprefix("pvc://").split("/", 1)[-1]' in source

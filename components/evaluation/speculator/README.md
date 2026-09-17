# Speculator ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Evaluate a draft model with verifier-only and speculative vLLM servers.

The component starts a verifier-only server to establish a latency baseline, then restarts vLLM with the draft model configured for speculative decoding. The evaluation dataset must be a JSONL file referenced by ``evaluation_dataset_uri`` with a ``prompt``, ``messages``, or ``conversations`` field,
or a string value on each line. At most ``evaluation_max_samples`` records are selected deterministically. The output Metrics artifact contains ``acceptance_rate``, ``acceptance_length``, and ``speedup`` for Model Registry integration.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `output_metrics` | `dsl.Output[dsl.Metrics]` | `None` | KFP Metrics artifact for acceptance and speedup values. |
| `output_results` | `dsl.Output[dsl.Artifact]` | `None` | JSON artifact containing raw timing and acceptance results. |
| `verifier_model` | `str` | `None` | Local or PVC-relative path to the verifier model. |
| `draft_model_path` | `str` | `None` | Local path to the trained draft model on the mounted PVC. |
| `evaluation_dataset_uri` | `str` | `None` | Local path or PVC URI (``pvc://<claim>/<path>``) to the evaluation JSONL. |
| `evaluation_speculator_type` | `str` | `None` | Speculator method used by the draft checkpoint. |
| `evaluation_max_samples` | `int` | `80` | Maximum number of prompts to evaluate. |
| `evaluation_max_tokens` | `int` | `256` | Maximum completion tokens per prompt. |
| `evaluation_temperature` | `float` | `0.0` | Sampling temperature sent to both servers. |
| `evaluation_num_speculative_tokens` | `int` | `5` | Number of draft tokens proposed per step. |
| `evaluation_gpu_memory_utilization` | `float` | `0.9` | vLLM GPU memory utilization fraction. |
| `evaluation_port` | `int` | `8000` | Local port used by the temporary vLLM servers. |
| `evaluation_startup_timeout_seconds` | `int` | `900` | Maximum time to wait for vLLM startup. |
| `evaluation_request_timeout_seconds` | `int` | `600` | Maximum time allowed for one request. |
| `evaluation_sample_seed` | `int` | `42` | Seed used for deterministic dataset sampling. |
| `evaluation_model_mount_path` | `str` | `/mnt/persistent` | Local mount prefix for a relative verifier path. |
| `evaluation_dataset_mount_path` | `str` | `/mnt/persistent` | Local mount prefix for a relative dataset path. |
| `evaluation_max_failure_rate` | `float` | `0.1` | Maximum allowed request failure fraction. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `None` |  |

## Metadata 🗂️

- **Name**: speculator
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: vLLM, Version: >=0.9.0
    - Name: Speculators, Version: >=0.7.1
- **Tags**:
  - evaluation
  - llm
  - speculator
  - speculative-decoding
  - metrics
- **Last Verified**: 2026-09-14 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/evaluating_performance/)

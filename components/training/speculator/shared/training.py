"""TrainJob helpers for Speculator components."""

import logging
import time


def select_runtime(client, log: logging.Logger, runtime_name: str):
    """Return the named ClusterTrainingRuntime."""
    for runtime in client.list_runtimes():
        if getattr(runtime, "name", "") == runtime_name:
            log.info("Runtime: %s", runtime)
            return runtime
    raise RuntimeError(f"Runtime '{runtime_name}' not found")


def wait_for_training_job(client, job: str, log: logging.Logger) -> None:
    """Wait for a TrainJob to complete and raise on failure."""
    client.wait_for_job_status(name=job, status={"Running"}, timeout=900)
    try:
        train_job = client.get_job(name=job)
    except Exception as exc:
        log.warning("Could not retrieve TrainJob details: %s", exc)
        train_job = None
    steps = (getattr(train_job, "steps", None) or []) if train_job else []
    if any(step.name == "node-0" for step in steps):
        for attempt in range(3):
            try:
                for line in client.get_job_logs(name=job, step="node-0", follow=True):
                    print(line, flush=True)
                break
            except Exception as exc:
                if attempt == 2:
                    log.warning("Log streaming failed: %s", exc)
                else:
                    time.sleep(5)
    client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=1800)
    result = client.get_job(name=job)
    if getattr(result, "status", None) != "Complete":
        raise RuntimeError(f"Job ended with status: {getattr(result, 'status', None)}")

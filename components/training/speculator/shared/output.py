"""Model output helpers for Speculator components."""

import logging
import os
import shutil


def find_model_dir(root: str):
    """Find the newest model checkpoint below root."""
    if not os.path.isdir(root):
        return None
    candidates = []
    for directory, _, files in os.walk(root):
        if "config.json" in files:
            candidates.append((os.path.getmtime(os.path.join(directory, "config.json")), directory))
    if candidates:
        return max(candidates)[1]
    directories = [os.path.join(root, entry) for entry in os.listdir(root)]
    directories = [directory for directory in directories if os.path.isdir(directory)]
    return max(directories, key=os.path.getmtime, default=None)


def persist_model(ckpt_dir: str, pvc_path: str, model_name: str, output_model, log: logging.Logger) -> None:
    """Copy the trained model to the PVC and KFP output artifact."""
    latest = find_model_dir(ckpt_dir)
    if not latest:
        raise RuntimeError(f"No model found in {ckpt_dir}")
    pvc_output = os.path.join(pvc_path, "final_model")
    if os.path.exists(pvc_output):
        shutil.rmtree(pvc_output, ignore_errors=True)
    shutil.copytree(latest, pvc_output, dirs_exist_ok=True)
    shutil.copytree(latest, output_model.path, dirs_exist_ok=True)
    output_model.name = f"{model_name}-checkpoint"
    output_model.metadata["model_name"] = model_name
    output_model.metadata["artifact_path"] = output_model.path
    output_model.metadata["pvc_model_dir"] = pvc_output
    log.info("Model copied to %s", pvc_output)

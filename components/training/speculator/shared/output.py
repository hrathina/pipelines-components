"""Model output helpers for Speculator components."""

import logging
import os
import shutil


def find_model_dir(root: str):
    """Find the best model checkpoint produced by Speculator training."""
    if not os.path.isdir(root):
        return None
    best_dir = os.path.join(root, "checkpoint_best")
    if os.path.isfile(os.path.join(best_dir, "config.json")):
        return best_dir
    return None


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

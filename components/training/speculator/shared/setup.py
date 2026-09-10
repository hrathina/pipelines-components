"""Launcher setup helpers for Speculator components."""

import logging
import os
import sys


def create_logger(name: str = "train_speculator") -> logging.Logger:
    """Create a logger that writes to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def init_k8s(log: logging.Logger):
    """Create a Kubernetes API client from injected credentials."""
    from kubernetes import client as k8s

    server = os.environ.get("KUBERNETES_SERVER_URL", "").strip()
    token = os.environ.get("KUBERNETES_AUTH_TOKEN", "").strip()
    if not server or not token:
        raise RuntimeError("KUBERNETES_SERVER_URL and KUBERNETES_AUTH_TOKEN must be set")
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    if not os.path.isfile(ca_path):
        raise RuntimeError(f"In-cluster CA certificate not found at {ca_path}")
    config = k8s.Configuration()
    config.host = server
    config.verify_ssl = True
    config.ssl_ca_cert = ca_path
    config.api_key = {"authorization": f"Bearer {token}"}
    k8s.Configuration.set_default(config)
    return k8s.ApiClient(config)


def configure_env(csv: str, base: dict[str, str], log: logging.Logger) -> dict[str, str]:
    """Merge comma-separated environment overrides and export them."""
    env = dict(base)
    for item in filter(None, (csv or "").split(",")):
        key, separator, value = item.strip().partition("=")
        if not separator or not key:
            raise ValueError(f"Invalid environment setting: {item}")
        env[key.strip()] = value.strip()
    for key, value in env.items():
        os.environ[key] = value
    log.info("Env: %s", sorted(env))
    return env


def setup_hf_token(env: dict[str, str], model: str, log: logging.Logger) -> None:
    """Propagate the optional Hugging Face token."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        env["HF_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
    elif isinstance(model, str) and "/" in model and not os.path.exists(model):
        log.warning("HF_TOKEN is not set; only public models are accessible for '%s'", model)

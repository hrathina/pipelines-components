"""Tests for Speculator model output helpers."""

from ..output import find_model_dir


def test_find_model_dir_selects_checkpoint_best(tmp_path):
    """Verify checkpoint_best is the only published checkpoint source."""
    best_dir = tmp_path / "checkpoint_best"
    best_dir.mkdir()
    (best_dir / "config.json").write_text("{}")

    checkpoint_dir = tmp_path / "10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}")

    assert find_model_dir(str(tmp_path)) == str(best_dir)


def test_find_model_dir_returns_none_without_checkpoint_best(tmp_path):
    """Verify missing checkpoint_best is not replaced by another checkpoint."""
    checkpoint_dir = tmp_path / "10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}")

    assert find_model_dir(str(tmp_path)) is None

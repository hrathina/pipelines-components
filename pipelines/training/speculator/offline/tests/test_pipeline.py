"""Tests for the Speculator offline pipeline."""

from kfp import compiler

from ..pipeline import speculator_offline_pipeline


class TestSpeculatorOfflinePipeline:
    """Basic tests for the offline pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_offline_pipeline)

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_offline_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()

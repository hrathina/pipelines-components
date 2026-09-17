"""Tests for the Speculator online pipeline."""

from kfp import compiler

from ..pipeline import speculator_online_pipeline


class TestSpeculatorOnlinePipeline:
    """Basic tests for the online pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_online_pipeline)

    def test_pipeline_exposes_evaluation_inputs(self):
        """Test that the pipeline exposes evaluation controls."""
        inputs = speculator_online_pipeline.component_spec.inputs
        for name in ("evaluation_dataset_uri", "evaluation_max_samples", "evaluation_sample_seed"):
            assert name in inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_online_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()

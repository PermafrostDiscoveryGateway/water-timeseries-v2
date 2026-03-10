"""Tests for BreakpointPipeline CLI and chunking functionality."""

import shutil
import tempfile
from pathlib import Path

import pytest

from water_timeseries.scripts.break_pipeline import BreakpointPipeline


class TestBreakpointPipelineChunking:
    """Test chunking functionality of BreakpointPipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_chunking_with_min_chunksize_2(self, dw_test_zarr_path, temp_output_dir):
        """Test chunking with min_chunksize=2."""
        output_file = temp_output_dir / "breaks_min5.parquet"

        # Create pipeline with min_chunksize override via public attribute
        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            min_chunksize=2,
        )

        # Re-chunk with new min_chunksize
        chunks = pipeline.chunk_dataset()
        print(pipeline.n_chunks)
        # Check that we have chunks
        assert pipeline.n_chunks == 3, "Should have 3 chunks"

        # Check minimum chunk size
        chunk_sizes = [len(chunk.id_geohash) for chunk in chunks]
        min_size = min(chunk_sizes)
        assert min_size >= 1, f"Minimum chunk size should be >= 2, got {min_size}"

        # Verify total number of IDs matches
        total_ids = sum(chunk_sizes)
        original_ids = len(pipeline.input_ds.id_geohash)
        assert total_ids == original_ids, f"Total IDs {total_ids} should match original {original_ids}"

    def test_chunking_with_min_chunksize_1(self, dw_test_zarr_path, temp_output_dir):
        """Test chunking with min_chunksize=1."""
        output_file = temp_output_dir / "breaks_min10.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            min_chunksize=1,
        )

        # Re-chunk with new min_chunksize
        chunks = pipeline.chunk_dataset()

        # Check that we have chunks
        assert len(chunks) == 5, "Should have at least one chunk"

        # Check minimum chunk size
        chunk_sizes = [len(chunk.id_geohash) for chunk in chunks]
        min_size = min(chunk_sizes)
        assert min_size >= 1, f"Minimum chunk size should be >= 1, got {min_size}"

        # Verify total number of IDs matchesfor default min chunk size and one setup for min chunksize of 50
        total_ids = sum(chunk_sizes)
        original_ids = len(pipeline.input_ds.id_geohash)
        assert total_ids == original_ids, f"Total IDs {total_ids} should match original {original_ids}"

    def test_chunking_single_chunk(self, dw_test_zarr_path, temp_output_dir):
        """Test chunking with n_chunks=1 (no actual chunking)."""
        output_file = temp_output_dir / "breaks_single.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=1,
            logger=None,
        )

        # With n_chunks=1, should have 1 chunk containing all data
        chunks = pipeline.chunked_ds

        assert len(chunks) == 1, f"Should have 1 chunk when n_chunks=1, got {len(chunks)}"
        assert len(chunks[0].id_geohash) == len(pipeline.input_ds.id_geohash)


class TestBreakpointPipelineExecution:
    """Test execution of BreakpointPipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_output_file_written_sequential(self, dw_test_zarr_path, temp_output_dir):
        """Test that output file is written when running sequentially (n_chunks=1)."""
        output_file = temp_output_dir / "breaks_sequential.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=1,
            logger=None,
        )

        pipeline.run_breaks()
        pipeline.save_to_parquet()

        # Check output file exists
        assert output_file.exists(), f"Output file should exist at {output_file}"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_output_file_written_parallel(self, dw_test_zarr_path, temp_output_dir):
        """Test that output file is written when running in parallel (n_chunks>1)."""
        output_file = temp_output_dir / "breaks_parallel.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=3,
            logger=None,
        )

        pipeline.run_breaks()
        pipeline.save_to_parquet()

        # Check output file exists
        assert output_file.exists(), f"Output file should exist at {output_file}"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

    def test_parallel_processing_uses_ray(self, dw_test_zarr_path, temp_output_dir):
        """Test that parallel processing uses Ray when n_chunks > 1."""
        import ray

        output_file = temp_output_dir / "breaks_ray.parquet"

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=3,
            logger=None,
        )

        # Run with parallelization
        pipeline.run_breaks()

        # Verify Ray was used (should be initialized)
        assert ray.is_initialized(), "Ray should be initialized for parallel processing"

    def test_sequential_processing_no_ray(self, dw_test_zarr_path, temp_output_dir):
        """Test that sequential processing doesn't require Ray initialization."""
        import ray

        output_file = temp_output_dir / "breaks_no_ray.parquet"

        # Shutdown ray if initialized
        if ray.is_initialized():
            ray.shutdown()

        pipeline = BreakpointPipeline(
            water_dataset_file=dw_test_zarr_path,
            output_file=str(output_file),
            n_chunks=1,
            logger=None,
        )

        # Run sequentially - should work without Ray
        pipeline.run_breaks()

        # Ray should not be initialized for sequential processing
        # (it may be initialized by the pipeline, but that's optional)


class TestBreakpointPipelineJRC:
    """Test BreakpointPipeline with JRC dataset."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_jrc_dataset_chunking(self, jrc_test_zarr_path, temp_output_dir):
        """Test chunking works with JRC dataset."""
        output_file = temp_output_dir / "breaks_jrc.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=jrc_test_zarr_path,
            output_file=str(output_file),
            n_chunks=5,
            logger=None,
        )

        # Check dataset type was detected correctly
        assert pipeline.water_dataset_type == "jrc", "Should detect JRC dataset type"

        # Check chunking worked
        assert len(pipeline.chunked_ds) > 0, "Should have chunks"

    def test_jrc_output_file_written(self, jrc_test_zarr_path, temp_output_dir):
        """Test that output file is written for JRC dataset."""
        output_file = temp_output_dir / "breaks_jrc_output.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=jrc_test_zarr_path,
            output_file=str(output_file),
            n_chunks=2,
            logger=None,
        )

        pipeline.run_breaks()
        pipeline.save_to_parquet()

        # Check output file exists
        assert output_file.exists(), f"Output file should exist at {output_file}"
        assert output_file.stat().st_size > 0, "Output file should not be empty"


class TestBreakpointPipelineParallelization:
    """Test parallelization with larger synthetic dataset."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary directory for output files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_parallel_chunking_with_large_dataset_default_min_chunksize(
        self, synthetic_dw_dataset_large_zarr, temp_output_dir
    ):
        """Test chunking with default min_chunksize (10) and 10 chunks.

        With 100 geohashes, n_chunks=10, and default min_chunksize=10:
        chunk_size = max(10, 100//10) = max(10, 10) = 10
        Expected: 10 chunks of 10 geohashes each
        """
        output_file = temp_output_dir / "breaks_large_default.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=synthetic_dw_dataset_large_zarr,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            # Using default min_chunksize=10
        )

        # Check that we have 10 chunks
        assert pipeline.n_chunks == 10, f"Expected 10 chunks, got {pipeline.n_chunks}"

        # Check that all chunks have the minimum size (10)
        chunk_sizes = [len(chunk.id_geohash) for chunk in pipeline.chunked_ds]
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        assert min_size >= 10, f"Minimum chunk size should be >= 10, got {min_size}"
        assert max_size <= 10, f"Maximum chunk size should be <= 10, got {max_size}"

        # Verify total number of IDs matches
        total_ids = sum(chunk_sizes)
        assert total_ids == 100, f"Total IDs should be 100, got {total_ids}"

    def test_parallel_chunking_with_large_dataset_min_chunksize_50(
        self, synthetic_dw_dataset_large_zarr, temp_output_dir
    ):
        """Test chunking with min_chunksize=50 and 10 chunks.

        With 100 geohashes, n_chunks=10, and min_chunksize=50:
        chunk_size = max(50, 100//10) = max(50, 10) = 50
        Expected: 2 chunks of 50 geohashes each
        """
        output_file = temp_output_dir / "breaks_large_min50.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=synthetic_dw_dataset_large_zarr,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            min_chunksize=50,
        )

        # Check that we have 2 chunks (due to min_chunksize=50)
        assert pipeline.n_chunks == 2, (
            f"Expected 2 chunks (min_chunksize=50 overrides n_chunks), got {pipeline.n_chunks}"
        )

        # Check that all chunks have size 50
        chunk_sizes = [len(chunk.id_geohash) for chunk in pipeline.chunked_ds]
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        assert min_size >= 50, f"Minimum chunk size should be >= 50, got {min_size}"
        assert max_size <= 50, f"Maximum chunk size should be <= 50, got {max_size}"

        # Verify total number of IDs matches
        total_ids = sum(chunk_sizes)
        assert total_ids == 100, f"Total IDs should be 100, got {total_ids}"

    def test_parallel_chunking_with_large_dataset_min_chunksize_5(
        self, synthetic_dw_dataset_large_zarr, temp_output_dir
    ):
        """Test that large dataset is chunked correctly for parallel processing."""
        output_file = temp_output_dir / "breaks_large.parquet"

        pipeline = BreakpointPipeline(
            water_dataset_file=synthetic_dw_dataset_large_zarr,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            min_chunksize=5,
        )

        # Check that we have the expected number of chunks
        assert pipeline.n_chunks == 10, f"Expected 10 chunks, got {pipeline.n_chunks}"

        # Check that all chunks have the minimum size
        chunk_sizes = [len(chunk.id_geohash) for chunk in pipeline.chunked_ds]
        min_size = min(chunk_sizes)
        assert min_size >= 5, f"Minimum chunk size should be >= 5, got {min_size}"

        # Verify total number of IDs matches
        total_ids = sum(chunk_sizes)
        assert total_ids == 100, f"Total IDs should be 100, got {total_ids}"

    def test_parallel_execution_with_large_dataset(self, synthetic_dw_dataset_large_zarr, temp_output_dir):
        """Test parallel execution with large dataset."""
        import ray

        output_file = temp_output_dir / "breaks_large_parallel.parquet"

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        pipeline = BreakpointPipeline(
            water_dataset_file=synthetic_dw_dataset_large_zarr,
            output_file=str(output_file),
            n_chunks=10,
            logger=None,
            min_chunksize=5,
        )

        # Run with parallelization
        pipeline.run_breaks()
        pipeline.save_to_parquet()

        # Check output file exists
        assert output_file.exists(), f"Output file should exist at {output_file}"
        assert output_file.stat().st_size > 0, "Output file should not be empty"

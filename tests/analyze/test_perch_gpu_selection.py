"""Tests for the enhanced --use_perch command line option with GPU/CPU model selection."""
import warnings
from unittest.mock import MagicMock, patch

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.utils import detect_gpu_available, ensure_model_exists


class TestGPUDetection:
    """Tests for GPU detection functionality."""

    @patch("tensorflow.config.list_physical_devices")
    def test_detect_gpu_available_with_gpu(self, mock_list_devices):
        """Test GPU detection when GPU is available."""
        mock_device = MagicMock()
        mock_device.name = "/physical_device:GPU:0"
        mock_device.device_type = "GPU"
        mock_list_devices.return_value = [mock_device]

        result = detect_gpu_available()
        assert result is True

    @patch("tensorflow.config.list_physical_devices")
    def test_detect_gpu_available_without_gpu(self, mock_list_devices):
        """Test GPU detection when GPU is not available."""
        mock_list_devices.return_value = []

        result = detect_gpu_available()
        assert result is False

    @patch("tensorflow.config.list_physical_devices")
    def test_detect_gpu_available_with_exception(self, mock_list_devices):
        """Test GPU detection when an exception occurs."""
        mock_list_devices.side_effect = Exception("Test exception")

        result = detect_gpu_available()
        assert result is False


class TestEnsureModelExists:
    """Tests for ensure_model_exists with new Perch options."""

    @patch("birdnet_analyzer.utils.check_birdnet_files")
    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    def test_ensure_model_exists_disable(self, mock_ensure_perch, mock_check_birdnet):
        """Test that Perch is not loaded when disabled."""
        mock_check_birdnet.return_value = True  # Pretend BirdNET model exists
        ensure_model_exists(check_perch="disable")
        mock_ensure_perch.assert_not_called()

    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    @patch("birdnet_analyzer.utils.detect_gpu_available")
    def test_ensure_model_exists_auto_with_gpu(self, mock_detect_gpu, mock_ensure_perch):
        """Test auto mode with GPU available."""
        mock_detect_gpu.return_value = True

        ensure_model_exists(check_perch="auto")

        mock_ensure_perch.assert_called_once_with(use_cpu_model=False)

    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    @patch("birdnet_analyzer.utils.detect_gpu_available")
    def test_ensure_model_exists_auto_without_gpu(self, mock_detect_gpu, mock_ensure_perch):
        """Test auto mode without GPU available."""
        mock_detect_gpu.return_value = False

        ensure_model_exists(check_perch="auto")

        mock_ensure_perch.assert_called_once_with(use_cpu_model=True)

    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    @patch("birdnet_analyzer.utils.detect_gpu_available")
    def test_ensure_model_exists_gpu_force(self, mock_detect_gpu, mock_ensure_perch):
        """Test forcing GPU model."""
        mock_detect_gpu.return_value = True

        ensure_model_exists(check_perch="gpu")

        mock_ensure_perch.assert_called_once_with(use_cpu_model=False)

    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    @patch("birdnet_analyzer.utils.detect_gpu_available")
    def test_ensure_model_exists_gpu_force_without_gpu(self, mock_detect_gpu, mock_ensure_perch):
        """Test forcing GPU model when GPU is not available - should warn."""
        mock_detect_gpu.return_value = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ensure_model_exists(check_perch="gpu")

            # Verify a warning was issued
            assert len(w) == 1
            assert "GPU model requested but no GPU detected" in str(w[0].message)

        mock_ensure_perch.assert_called_once_with(use_cpu_model=False)

    @patch("birdnet_analyzer.utils.ensure_perch_exists")
    def test_ensure_model_exists_cpu_force(self, mock_ensure_perch):
        """Test forcing CPU model."""
        ensure_model_exists(check_perch="cpu")

        mock_ensure_perch.assert_called_once_with(use_cpu_model=True)

    def test_ensure_model_exists_invalid_option(self):
        """Test that invalid option raises ValueError."""
        with pytest.raises(ValueError, match="Invalid use_perch value"):
            ensure_model_exists(check_perch="invalid")


class TestEnsurePerchExists:
    """Tests for ensure_perch_exists function."""

    @patch("birdnet_analyzer.utils.check_perchv2_files")
    @patch("kagglehub.model_download")
    @patch("shutil.copytree")
    @patch("os.makedirs")
    def test_ensure_perch_exists_gpu_model(self, mock_makedirs, mock_copytree, mock_kagglehub, mock_check_files):
        """Test downloading GPU model."""
        from birdnet_analyzer.utils import ensure_perch_exists

        mock_check_files.return_value = False
        mock_kagglehub.return_value = "/fake/path/to/model"

        ensure_perch_exists(use_cpu_model=False)

        mock_kagglehub.assert_called_once_with(
            "google/bird-vocalization-classifier/tensorFlow2/perch_v2"
        )

    @patch("birdnet_analyzer.utils.check_perchv2_files")
    @patch("kagglehub.model_download")
    @patch("shutil.copytree")
    @patch("os.makedirs")
    def test_ensure_perch_exists_cpu_model(self, mock_makedirs, mock_copytree, mock_kagglehub, mock_check_files):
        """Test downloading CPU model."""
        from birdnet_analyzer.utils import ensure_perch_exists

        mock_check_files.return_value = False
        mock_kagglehub.return_value = "/fake/path/to/model"

        ensure_perch_exists(use_cpu_model=True)

        mock_kagglehub.assert_called_once_with(
            "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
        )

    @patch("birdnet_analyzer.utils.check_perchv2_files")
    def test_ensure_perch_exists_already_exists(self, mock_check_files):
        """Test that model is not re-downloaded if it exists."""
        from birdnet_analyzer.utils import ensure_perch_exists

        mock_check_files.return_value = True

        ensure_perch_exists(use_cpu_model=False)

        # Should not attempt to download since model exists
        mock_check_files.assert_called_once()


class TestAnalyzeWithPerchOptions:
    """Tests for analyze function with new Perch options."""

    @patch("birdnet_analyzer.analyze.core._set_params")
    @patch("birdnet_analyzer.analyze.utils.analyze_file")
    @patch("birdnet_analyzer.analyze.utils.save_analysis_params")
    def test_analyze_with_perch_disable(self, mock_save_params, mock_analyze_file, mock_set_params):
        """Test analyze with Perch disabled."""
        from birdnet_analyzer.analyze.core import analyze

        mock_set_params.return_value = [("test.wav", {})]
        mock_analyze_file.return_value = "test_results.txt"

        cfg.FILE_LIST = ["test.wav"]
        cfg.LABELS = ["Species1"]
        cfg.SPECIES_LIST = None
        cfg.CPU_THREADS = 1
        cfg.COMBINE_RESULTS = False
        cfg.OUTPUT_PATH = "/tmp"

        analyze("test.wav", "/tmp", use_perch="disable")

        _, kwargs = mock_set_params.call_args
        assert kwargs["use_perch"] == "disable"

    @patch("birdnet_analyzer.analyze.core._set_params")
    @patch("birdnet_analyzer.analyze.utils.analyze_file")
    @patch("birdnet_analyzer.analyze.utils.save_analysis_params")
    def test_analyze_with_perch_auto(self, mock_save_params, mock_analyze_file, mock_set_params):
        """Test analyze with Perch auto mode."""
        from birdnet_analyzer.analyze.core import analyze

        mock_set_params.return_value = [("test.wav", {})]
        mock_analyze_file.return_value = "test_results.txt"

        cfg.FILE_LIST = ["test.wav"]
        cfg.LABELS = ["Species1"]
        cfg.SPECIES_LIST = None
        cfg.CPU_THREADS = 1
        cfg.COMBINE_RESULTS = False
        cfg.OUTPUT_PATH = "/tmp"

        analyze("test.wav", "/tmp", use_perch="auto")

        _, kwargs = mock_set_params.call_args
        assert kwargs["use_perch"] == "auto"

    @patch("birdnet_analyzer.analyze.core._set_params")
    @patch("birdnet_analyzer.analyze.utils.analyze_file")
    @patch("birdnet_analyzer.analyze.utils.save_analysis_params")
    def test_analyze_with_perch_gpu(self, mock_save_params, mock_analyze_file, mock_set_params):
        """Test analyze with Perch GPU mode."""
        from birdnet_analyzer.analyze.core import analyze

        mock_set_params.return_value = [("test.wav", {})]
        mock_analyze_file.return_value = "test_results.txt"

        cfg.FILE_LIST = ["test.wav"]
        cfg.LABELS = ["Species1"]
        cfg.SPECIES_LIST = None
        cfg.CPU_THREADS = 1
        cfg.COMBINE_RESULTS = False
        cfg.OUTPUT_PATH = "/tmp"

        analyze("test.wav", "/tmp", use_perch="gpu")

        _, kwargs = mock_set_params.call_args
        assert kwargs["use_perch"] == "gpu"

    @patch("birdnet_analyzer.analyze.core._set_params")
    @patch("birdnet_analyzer.analyze.utils.analyze_file")
    @patch("birdnet_analyzer.analyze.utils.save_analysis_params")
    def test_analyze_with_perch_cpu(self, mock_save_params, mock_analyze_file, mock_set_params):
        """Test analyze with Perch CPU mode."""
        from birdnet_analyzer.analyze.core import analyze

        mock_set_params.return_value = [("test.wav", {})]
        mock_analyze_file.return_value = "test_results.txt"

        cfg.FILE_LIST = ["test.wav"]
        cfg.LABELS = ["Species1"]
        cfg.SPECIES_LIST = None
        cfg.CPU_THREADS = 1
        cfg.COMBINE_RESULTS = False
        cfg.OUTPUT_PATH = "/tmp"

        analyze("test.wav", "/tmp", use_perch="cpu")

        _, kwargs = mock_set_params.call_args
        assert kwargs["use_perch"] == "cpu"

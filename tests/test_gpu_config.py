"""Tests for GPU configuration feature."""

import importlib
import os
import sys

import birdnet_analyzer.config as cfg


def test_gpu_enabled_default():
    """Test that GPU_ENABLED defaults to True."""
    assert hasattr(cfg, "GPU_ENABLED")
    assert cfg.GPU_ENABLED is True


def test_gpu_enabled_can_be_disabled():
    """Test that GPU_ENABLED can be set to False."""
    original_value = cfg.GPU_ENABLED
    try:
        cfg.GPU_ENABLED = False
        assert cfg.GPU_ENABLED is False
    finally:
        cfg.GPU_ENABLED = original_value


def test_gpu_config_affects_cuda_visible_devices():
    """Test that GPU config affects CUDA_VISIBLE_DEVICES environment variable."""
    # Save the original state
    original_cuda_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    original_gpu_enabled = cfg.GPU_ENABLED

    try:
        # Test with GPU disabled
        cfg.GPU_ENABLED = False

        # Remove the module from cache to force reimport
        if "birdnet_analyzer.model" in sys.modules:
            del sys.modules["birdnet_analyzer.model"]

        # Import the model module which should set CUDA_VISIBLE_DEVICES
        import birdnet_analyzer.model

        # When GPU is disabled, CUDA_VISIBLE_DEVICES should be set to ""
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

        # Test with GPU enabled
        cfg.GPU_ENABLED = True

        # Clear the environment variable to simulate fresh state
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # Remove the module from cache to force reimport
        if "birdnet_analyzer.model" in sys.modules:
            del sys.modules["birdnet_analyzer.model"]

        # Import the model module which should NOT set CUDA_VISIBLE_DEVICES
        import birdnet_analyzer.model  # noqa: F401, F811

        # When GPU is enabled, CUDA_VISIBLE_DEVICES should not be set (or remain unset)
        # It might not be in the environment at all, or it might be set to a value
        # The key is that it's NOT set to "" which would disable GPU
        cuda_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert cuda_value != "" or cuda_value is None

    finally:
        # Restore original state
        cfg.GPU_ENABLED = original_gpu_enabled

        if original_cuda_value is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_value
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # Reload the module with original config
        if "birdnet_analyzer.model" in sys.modules:
            del sys.modules["birdnet_analyzer.model"]
        importlib.import_module("birdnet_analyzer.model")

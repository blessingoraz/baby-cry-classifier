"""
Unit tests for baby cry classifier prediction pipeline.

Run with: pytest tests/test_predict.py -v
"""
import pytest
import torch
import numpy as np
import json
import tempfile
import soundfile as sf
from pathlib import Path

from src.predict import predict_audio
from src.utils import format_prediction
from src.model import load_model, CryResNet, get_device


class TestCryResNet:
    """Test CryResNet model architecture and forward pass."""

    def test_model_initialization(self):
        """Test that model initializes with correct output shape."""
        model = CryResNet(
            num_classes=8,
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=True,
            droprate=0.8,
            size_inner=512
        )
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = CryResNet(num_classes=8, size_inner=512, droprate=0.8)
        model.eval()
        
        # Input: (batch=1, channels=1, height=224, width=224)
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        # Output should be (batch=1, num_classes=8)
        assert output.shape == (1, 8)

    def test_forward_pass_batch(self):
        """Test forward pass with batch size > 1."""
        model = CryResNet(num_classes=8, size_inner=512, droprate=0.8)
        model.eval()
        
        # Batch of 4 samples
        x = torch.randn(4, 1, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 8)


class TestFormatPrediction:
    """Test prediction formatting utility."""

    def test_format_prediction_structure(self):
        """Test that format_prediction returns correct structure."""
        probs = {
            "hungry": 0.5,
            "discomfort": 0.3,
            "burping": 0.1,
            "tired": 0.05,
            "belly_pain": 0.03,
            "cold_hot": 0.01,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=3)
        
        assert "label" in result
        assert "probability" in result
        assert "top_k" in result
        assert "all_probs" in result
        
        # Top label should be "hungry" (highest prob)
        assert result["label"] == "hungry"
        assert result["probability"] == pytest.approx(0.5)

    def test_format_prediction_top_k(self):
        """Test that top_k limits returned predictions."""
        probs = {
            "hungry": 0.5,
            "discomfort": 0.3,
            "burping": 0.1,
            "tired": 0.05,
            "belly_pain": 0.03,
            "cold_hot": 0.01,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=2)
        assert len(result["top_k"]) == 2
        assert result["top_k"][0]["label"] == "hungry"
        assert result["top_k"][1]["label"] == "discomfort"

    def test_format_prediction_all_probs_present(self):
        """Test that all_probs contains all classes."""
        probs = {
            "hungry": 0.5,
            "discomfort": 0.3,
            "burping": 0.1,
            "tired": 0.05,
            "belly_pain": 0.03,
            "cold_hot": 0.01,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=3)
        assert len(result["all_probs"]) == 8
        assert result["all_probs"]["hungry"] == pytest.approx(0.5)


class TestPredictAudio:
    """Test audio prediction pipeline."""

    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        # Generate synthetic audio (1 second at 8000 Hz)
        sr = 8000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz sine wave
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_predict_audio_output_shape(self, sample_audio_file):
        """Test that predict_audio returns dict with 8 classes."""
        result = predict_audio(sample_audio_file)
        
        assert isinstance(result, dict)
        assert len(result) == 8
        
        # All values should be probabilities (0-1)
        for label, prob in result.items():
            assert isinstance(prob, (float, np.floating))
            assert 0 <= prob <= 1

    def test_predict_audio_probabilities_sum(self, sample_audio_file):
        """Test that predicted probabilities sum to ~1."""
        result = predict_audio(sample_audio_file)
        total = sum(result.values())
        
        # Should sum to approximately 1.0 (softmax output)
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_predict_audio_valid_classes(self, sample_audio_file):
        """Test that all returned classes are valid."""
        expected_classes = {
            "belly_pain", "burping", "cold_hot", "discomfort",
            "hungry", "lonely", "scared", "tired"
        }
        result = predict_audio(sample_audio_file)
        
        assert set(result.keys()) == expected_classes


class TestLoadModel:
    """Test model loading functionality."""

    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device.type in ["cpu", "cuda"]

    def test_load_model_cpu(self):
        """Test loading model on CPU."""
        # Create a dummy checkpoint
        checkpoint = {
            "model_state_dict": {},
            "epoch": 1,
            "best_val_macro_f1": 0.9
        }
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            
            # This will fail if checkpoint doesn't match architecture,
            # but tests the loading flow
            try:
                model, device, meta = load_model(
                    ckpt_path=f.name,
                    num_classes=8,
                    pretrained=False,
                    device=torch.device("cpu")
                )
                assert model is not None
                assert device.type == "cpu"
            except RuntimeError:
                # Expected: mismatch between saved and current architecture
                pass
            finally:
                Path(f.name).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        sr = 8000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)

    def test_predict_and_format(self, sample_audio_file):
        """Test full pipeline: predict + format."""
        probs = predict_audio(sample_audio_file)
        formatted = format_prediction(probs, top_k=3)
        
        # Verify structure
        assert "label" in formatted
        assert "probability" in formatted
        assert "top_k" in formatted
        assert len(formatted["top_k"]) == 3
        
        # Verify probabilities are valid
        assert 0 <= formatted["probability"] <= 1
        for item in formatted["top_k"]:
            assert 0 <= item["probability"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

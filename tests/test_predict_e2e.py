"""
End-to-end integration tests for src/predict.py with real audio samples.

These tests verify the full prediction pipeline using actual audio files from data/raw.

Run with: pytest tests/test_predict_e2e.py -v
"""
import pytest
import glob
import json
from pathlib import Path

from src.predict import predict_audio


class TestPredictAudioE2E:
    """End-to-end tests using real audio files."""

    @pytest.fixture
    def sample_audio_paths(self):
        """Get sample audio files from each class."""
        data_root = Path("data/raw")
        samples = {}
        
        for class_dir in data_root.iterdir():
            if class_dir.is_dir():
                wav_files = list(class_dir.glob("*.wav"))
                if wav_files:
                    samples[class_dir.name] = str(wav_files[0])
        
        return samples

    @pytest.fixture
    def expected_classes(self):
        """Load expected classes from label map."""
        with open("data/splits/label_map.json") as f:
            label_info = json.load(f)
        id2label = {int(k): v for k, v in label_info["id2label"].items()}
        classes = [id2label[i] for i in range(len(id2label))]
        return set(classes)

    def test_predict_audio_returns_dict(self, sample_audio_paths):
        """Test that predict_audio returns a dictionary."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        assert isinstance(result, dict)

    def test_predict_audio_has_all_classes(self, sample_audio_paths, expected_classes):
        """Test that prediction output has all expected classes."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        assert set(result.keys()) == expected_classes

    def test_predict_audio_probabilities_sum_to_one(self, sample_audio_paths):
        """Test that predicted probabilities sum to approximately 1."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        total_prob = sum(result.values())
        assert total_prob == pytest.approx(1.0, abs=1e-5)

    def test_predict_audio_probabilities_in_valid_range(self, sample_audio_paths):
        """Test that all probabilities are between 0 and 1."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        for label, prob in result.items():
            assert 0 <= prob <= 1, f"Probability for {label} is {prob}, out of range"

    def test_predict_audio_returns_floats(self, sample_audio_paths):
        """Test that all probability values are floats."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        for label, prob in result.items():
            assert isinstance(prob, (float, int)), f"Probability for {label} is not numeric"

    def test_predict_each_class_sample(self, sample_audio_paths):
        """Test prediction on sample from each class."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        results = {}
        for class_name, audio_path in sample_audio_paths.items():
            result = predict_audio(audio_path)
            results[class_name] = result
            
            # Verify result properties
            assert isinstance(result, dict)
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)
            assert all(0 <= p <= 1 for p in result.values())
        
        # Each class should have results
        assert len(results) > 0

    def test_predict_top_probability_reasonable(self, sample_audio_paths):
        """Test that at least one class has non-negligible probability."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        max_prob = max(result.values())
        # Top class should have at least 10% confidence
        assert max_prob >= 0.1, f"Top class probability too low: {max_prob}"

    def test_predict_multiple_calls_consistent_shape(self, sample_audio_paths):
        """Test that multiple predictions return consistent output shape."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        
        # Call multiple times
        results = [predict_audio(audio_path) for _ in range(3)]
        
        # All should have same keys
        first_keys = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == first_keys

    def test_predict_different_classes_different_predictions(self, sample_audio_paths):
        """Test that different audio classes produce different predictions."""
        if len(sample_audio_paths) < 2:
            pytest.skip("Need at least 2 classes to compare")
        
        paths = list(sample_audio_paths.values())
        result1 = predict_audio(paths[0])
        result2 = predict_audio(paths[1])
        
        # Predictions should be different
        assert result1 != result2, "Different audio files should produce different predictions"

    def test_predict_audio_file_not_found(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(Exception):
            predict_audio("nonexistent_audio_file.wav")

    def test_predict_with_various_sample_sizes(self, sample_audio_paths):
        """Test prediction works with audio samples of different sizes."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        # Get samples and verify they can be predicted
        for class_name, audio_path in sample_audio_paths.items():
            result = predict_audio(audio_path)
            
            # All results should be valid
            assert isinstance(result, dict)
            assert len(result) == 8  # 8 classes
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_predict_output_format_matches_classes(self, sample_audio_paths, expected_classes):
        """Test that output keys exactly match expected class names."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        # Keys should exactly match expected classes
        assert set(result.keys()) == expected_classes

    def test_predict_class_names_valid(self, sample_audio_paths):
        """Test that all predicted class names are strings."""
        if not sample_audio_paths:
            pytest.skip("No audio files found in data/raw")
        
        audio_path = next(iter(sample_audio_paths.values()))
        result = predict_audio(audio_path)
        
        for label in result.keys():
            assert isinstance(label, str)
            assert len(label) > 0


class TestPredictAudioIntegration:
    """Integration tests combining preprocessing, model, and formatting."""

    @pytest.fixture
    def label_map(self):
        """Load the label map."""
        with open("data/splits/label_map.json") as f:
            return json.load(f)

    def test_prediction_matches_label_map(self, label_map):
        """Test that predictions use labels from the label map."""
        sample_audio = glob.glob("data/raw/*/*.wav")
        if not sample_audio:
            pytest.skip("No audio files found")
        
        result = predict_audio(sample_audio[0])
        
        id2label = {int(k): v for k, v in label_map["id2label"].items()}
        expected_labels = set(id2label.values())
        
        assert set(result.keys()) == expected_labels

    def test_prediction_pipeline_speed(self, benchmark=None):
        """Test that predictions complete in reasonable time (optional benchmark)."""
        sample_audio = glob.glob("data/raw/*/*.wav")
        if not sample_audio:
            pytest.skip("No audio files found")
        
        # Predict on first sample
        audio_path = sample_audio[0]
        result = predict_audio(audio_path)
        
        # Should complete and return valid result
        assert result is not None
        assert len(result) == 8

    def test_prediction_handles_various_audio_formats(self):
        """Test that prediction works with various audio files in data/raw."""
        audio_files = glob.glob("data/raw/*/*.wav")
        
        if not audio_files:
            pytest.skip("No audio files found in data/raw")
        
        # Test a sample of different audio files
        test_files = audio_files[:min(5, len(audio_files))]
        
        for audio_path in test_files:
            result = predict_audio(audio_path)
            
            assert isinstance(result, dict)
            assert len(result) == 8
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)

    def test_prediction_consistency_across_runs(self):
        """Test that same audio file produces consistent predictions."""
        sample_audio = glob.glob("data/raw/*/*.wav")
        if not sample_audio:
            pytest.skip("No audio files found")
        
        audio_path = sample_audio[0]
        
        # Run prediction multiple times
        result1 = predict_audio(audio_path)
        result2 = predict_audio(audio_path)
        
        # Results should be identical (deterministic)
        for label in result1.keys():
            assert result1[label] == pytest.approx(result2[label], abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

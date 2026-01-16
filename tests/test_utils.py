"""
Unit tests for src/utils.py formatting utilities.

Run with: pytest tests/test_utils.py -v
"""
import pytest
from src.utils import format_prediction


class TestFormatPrediction:
    """Test format_prediction utility function."""

    def test_basic_formatting(self):
        """Test basic formatting with standard probabilities."""
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
        
        # Check structure
        assert isinstance(result, dict)
        assert "label" in result
        assert "probability" in result
        assert "top_k" in result
        assert "all_probs" in result

    def test_top_label_is_highest_prob(self):
        """Test that top label is the one with highest probability."""
        probs = {
            "hungry": 0.6,
            "discomfort": 0.2,
            "burping": 0.1,
            "tired": 0.05,
            "belly_pain": 0.03,
            "cold_hot": 0.01,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs)
        
        assert result["label"] == "hungry"
        assert result["probability"] == pytest.approx(0.6)

    def test_top_k_limits_results(self):
        """Test that top_k parameter limits the number of results."""
        probs = {
            "hungry": 0.4,
            "discomfort": 0.3,
            "burping": 0.15,
            "tired": 0.08,
            "belly_pain": 0.04,
            "cold_hot": 0.02,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        # Test different top_k values
        result_k1 = format_prediction(probs, top_k=1)
        result_k3 = format_prediction(probs, top_k=3)
        result_k5 = format_prediction(probs, top_k=5)
        
        assert len(result_k1["top_k"]) == 1
        assert len(result_k3["top_k"]) == 3
        assert len(result_k5["top_k"]) == 5

    def test_top_k_sorted_descending(self):
        """Test that top_k results are sorted by probability descending."""
        probs = {
            "hungry": 0.4,
            "discomfort": 0.3,
            "burping": 0.15,
            "tired": 0.08,
            "belly_pain": 0.04,
            "cold_hot": 0.02,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=4)
        top_k = result["top_k"]
        
        # Check probabilities are in descending order
        for i in range(len(top_k) - 1):
            assert top_k[i]["probability"] >= top_k[i + 1]["probability"]
        
        # Check correct order
        assert top_k[0]["label"] == "hungry"
        assert top_k[1]["label"] == "discomfort"
        assert top_k[2]["label"] == "burping"
        assert top_k[3]["label"] == "tired"

    def test_all_probs_contains_all_classes(self):
        """Test that all_probs contains all input classes."""
        probs = {
            "hungry": 0.4,
            "discomfort": 0.3,
            "burping": 0.15,
            "tired": 0.08,
            "belly_pain": 0.04,
            "cold_hot": 0.02,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=2)
        all_probs = result["all_probs"]
        
        # Should contain all 8 classes
        assert len(all_probs) == 8
        assert set(all_probs.keys()) == set(probs.keys())
        
        # Probabilities should match input
        for label, prob in probs.items():
            assert all_probs[label] == pytest.approx(prob)

    def test_probability_values_are_floats(self):
        """Test that all probabilities are returned as floats."""
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
        
        result = format_prediction(probs)
        
        # Check top label probability
        assert isinstance(result["probability"], float)
        
        # Check top_k probabilities
        for item in result["top_k"]:
            assert isinstance(item["probability"], float)
        
        # Check all_probs
        for prob in result["all_probs"].values():
            assert isinstance(prob, float)

    def test_edge_case_single_class(self):
        """Test with only one class."""
        probs = {"hungry": 1.0}
        
        result = format_prediction(probs, top_k=1)
        
        assert result["label"] == "hungry"
        assert result["probability"] == pytest.approx(1.0)
        assert len(result["top_k"]) == 1
        assert result["top_k"][0]["label"] == "hungry"

    def test_edge_case_equal_probabilities(self):
        """Test with equal probabilities (order should be consistent)."""
        probs = {
            "hungry": 0.125,
            "discomfort": 0.125,
            "burping": 0.125,
            "tired": 0.125,
            "belly_pain": 0.125,
            "cold_hot": 0.125,
            "lonely": 0.125,
            "scared": 0.125
        }
        
        result = format_prediction(probs, top_k=3)
        
        # All probabilities should be equal
        for item in result["top_k"]:
            assert item["probability"] == pytest.approx(0.125)

    def test_edge_case_very_small_probabilities(self):
        """Test with very small probabilities."""
        probs = {
            "hungry": 1e-10,
            "discomfort": 1e-11,
            "burping": 1e-12,
            "tired": 1e-13,
            "belly_pain": 1e-14,
            "cold_hot": 1e-15,
            "lonely": 1e-16,
            "scared": 1e-17
        }
        
        result = format_prediction(probs, top_k=2)
        
        # Should still work correctly
        assert result["label"] == "hungry"
        assert result["probability"] == pytest.approx(1e-10)
        assert len(result["top_k"]) == 2

    def test_top_k_greater_than_available(self):
        """Test when top_k is greater than number of classes."""
        probs = {
            "hungry": 0.5,
            "discomfort": 0.3,
            "burping": 0.2
        }
        
        result = format_prediction(probs, top_k=10)
        
        # Should return all available classes
        assert len(result["top_k"]) == 3
        assert len(result["all_probs"]) == 3

    def test_default_top_k_is_three(self):
        """Test that default top_k is 3."""
        probs = {
            "hungry": 0.4,
            "discomfort": 0.3,
            "burping": 0.15,
            "tired": 0.08,
            "belly_pain": 0.04,
            "cold_hot": 0.02,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs)
        
        # Default should be 3
        assert len(result["top_k"]) == 3

    def test_top_k_items_have_label_and_probability(self):
        """Test that each top_k item has label and probability keys."""
        probs = {
            "hungry": 0.4,
            "discomfort": 0.3,
            "burping": 0.15,
            "tired": 0.08,
            "belly_pain": 0.04,
            "cold_hot": 0.02,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=3)
        
        for item in result["top_k"]:
            assert "label" in item
            assert "probability" in item
            assert len(item) == 2  # Only these two keys

    def test_return_type_consistency(self):
        """Test that return type is always a dict with expected keys."""
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
        
        result = format_prediction(probs, top_k=5)
        
        required_keys = {"label", "probability", "top_k", "all_probs"}
        assert set(result.keys()) == required_keys

    def test_numpy_float_compatibility(self):
        """Test that function handles numpy float types."""
        import numpy as np
        
        probs = {
            "hungry": np.float32(0.5),
            "discomfort": np.float64(0.3),
            "burping": np.float32(0.1),
            "tired": np.float64(0.05),
            "belly_pain": 0.03,
            "cold_hot": 0.01,
            "lonely": 0.005,
            "scared": 0.005
        }
        
        result = format_prediction(probs, top_k=3)
        
        # Should not raise error and should convert to float
        assert isinstance(result["probability"], float)
        assert result["label"] == "hungry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

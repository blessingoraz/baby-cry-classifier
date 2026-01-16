"""
Unit and integration tests for src/api.py FastAPI endpoints.

Run with: pytest tests/test_api.py -v
"""
import pytest
import tempfile
import soundfile as sf
import numpy as np
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_audio_bytes():
    """Create synthetic audio as bytes for testing."""
    sr = 8000
    duration = 7.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Write to bytes
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def real_audio_file():
    """Get a real audio file from data/raw."""
    import glob
    audio_files = glob.glob("data/raw/*/*.wav")
    if audio_files:
        return audio_files[0]
    return None


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        """Test that /health returns status 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_structure(self, client):
        """Test that /health returns correct JSON structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_returns_200_with_valid_file(self, client, sample_audio_bytes):
        """Test that /predict returns 200 OK with valid audio file."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        assert response.status_code == 200

    def test_predict_returns_json(self, client, sample_audio_bytes):
        """Test that /predict returns JSON response."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert isinstance(data, dict)

    def test_predict_has_required_keys(self, client, sample_audio_bytes):
        """Test that /predict response has required keys."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        
        required_keys = {"label", "probability", "top_k", "all_probs"}
        assert set(data.keys()) == required_keys

    def test_predict_label_is_string(self, client, sample_audio_bytes):
        """Test that predicted label is a string."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert isinstance(data["label"], str)

    def test_predict_probability_is_float(self, client, sample_audio_bytes):
        """Test that top probability is a float."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert isinstance(data["probability"], (int, float))
        assert 0 <= data["probability"] <= 1

    def test_predict_top_k_is_list(self, client, sample_audio_bytes):
        """Test that top_k is a list."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert isinstance(data["top_k"], list)

    def test_predict_top_k_default_length(self, client, sample_audio_bytes):
        """Test that default top_k returns 3 items."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert len(data["top_k"]) == 3

    def test_predict_top_k_custom_value(self, client, sample_audio_bytes):
        """Test that custom top_k parameter works."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": 5}
        )
        data = response.json()
        assert len(data["top_k"]) == 5

    def test_predict_top_k_items_have_label_and_probability(self, client, sample_audio_bytes):
        """Test that each top_k item has label and probability."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        
        for item in data["top_k"]:
            assert "label" in item
            assert "probability" in item
            assert isinstance(item["label"], str)
            assert 0 <= item["probability"] <= 1

    def test_predict_all_probs_has_all_classes(self, client, sample_audio_bytes):
        """Test that all_probs contains all 8 classes."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        assert len(data["all_probs"]) == 8

    def test_predict_all_probs_sum_to_one(self, client, sample_audio_bytes):
        """Test that all probabilities sum to approximately 1."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        total = sum(data["all_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_predict_no_file_uploaded_returns_error(self, client):
        """Test that missing file returns error (422 Unprocessable Entity)."""
        response = client.post("/predict")
        # FastAPI returns 422 for missing required parameters
        assert response.status_code == 422

    def test_predict_empty_file_returns_error(self, client):
        """Test that empty file raises an error."""
        response = client.post(
            "/predict",
            files={"file": ("empty.wav", BytesIO(b""), "audio/wav")}
        )
        # Should return 500 or 400 due to invalid audio
        assert response.status_code in [400, 422, 500]

    def test_predict_with_top_k_1(self, client, sample_audio_bytes):
        """Test prediction with top_k=1."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": 1}
        )
        data = response.json()
        assert len(data["top_k"]) == 1
        assert data["top_k"][0]["label"] == data["label"]

    def test_predict_with_top_k_greater_than_classes(self, client, sample_audio_bytes):
        """Test prediction with top_k > 8 classes."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": 20}
        )
        data = response.json()
        # Should return all 8 classes
        assert len(data["top_k"]) == 8

    def test_predict_response_valid_class_names(self, client, sample_audio_bytes):
        """Test that response contains valid class names."""
        expected_classes = {
            "belly_pain", "burping", "cold_hot", "discomfort",
            "hungry", "lonely", "scared", "tired"
        }
        
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
        )
        data = response.json()
        
        assert set(data["all_probs"].keys()) == expected_classes
        assert data["label"] in expected_classes


class TestPredictEndpointIntegration:
    """Integration tests for /predict endpoint with real audio."""

    def test_predict_with_real_audio_file(self, client, real_audio_file):
        """Test /predict with real audio file from data/raw."""
        if not real_audio_file:
            pytest.skip("No real audio files found in data/raw")
        
        with open(real_audio_file, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": (Path(real_audio_file).name, f, "audio/wav")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert data["probability"] > 0

    def test_predict_multiple_requests(self, client, sample_audio_bytes):
        """Test that multiple predictions work correctly."""
        for i in range(3):
            response = client.post(
                "/predict",
                files={"file": (f"test_{i}.wav", BytesIO(sample_audio_bytes), "audio/wav")}
            )
            assert response.status_code == 200
            data = response.json()
            assert "label" in data

    def test_predict_different_top_k_values(self, client, sample_audio_bytes):
        """Test /predict with various top_k values."""
        for top_k in [1, 2, 3, 5, 8]:
            response = client.post(
                "/predict",
                files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
                params={"top_k": top_k}
            )
            data = response.json()
            expected_length = min(top_k, 8)
            assert len(data["top_k"]) == expected_length

    def test_predict_top_k_sorted_descending(self, client, sample_audio_bytes):
        """Test that top_k results are sorted by probability descending."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": 5}
        )
        data = response.json()
        
        probabilities = [item["probability"] for item in data["top_k"]]
        assert probabilities == sorted(probabilities, reverse=True)


class TestAPIErrorHandling:
    """Test error handling in API."""

    def test_health_endpoint_always_available(self, client):
        """Test that health endpoint is always available."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_predict_invalid_top_k_type(self, client, sample_audio_bytes):
        """Test behavior with invalid top_k parameter type."""
        # FastAPI should handle type coercion or return error
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": "invalid"}
        )
        # Should handle gracefully
        assert response.status_code in [200, 422]

    def test_predict_negative_top_k(self, client, sample_audio_bytes):
        """Test with negative top_k value."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": -1}
        )
        # Should handle gracefully (either error or treat as 0)
        assert response.status_code in [200, 422, 400]

    def test_predict_zero_top_k(self, client, sample_audio_bytes):
        """Test with zero top_k value."""
        response = client.post(
            "/predict",
            files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
            params={"top_k": 0}
        )
        # Should handle gracefully
        assert response.status_code in [200, 422]


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data

    def test_docs_endpoint_available(self, client):
        """Test that API docs endpoint is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client):
        """Test that ReDoc endpoint is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

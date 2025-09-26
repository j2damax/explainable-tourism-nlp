"""
Tests for the FastAPI endpoints in api.py
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after path setup
from backend.api import app, PredictRequest, ExplainRequest

# Create a test client
client = TestClient(app)

# Mock data for tests
SAMPLE_REVIEW = "We loved the eco-friendly resort that used solar power and served organic local food."
MOCK_DIMENSIONS = [
    "Regenerative & Eco-Tourism",
    "Integrated Wellness",
    "Immersive Culinary",
    "Off-the-Beaten-Path Adventure"
]

# Mock return values
MOCK_PREDICTION = [
    {"label": "Regenerative & Eco-Tourism", "score": 0.85},
    {"label": "Immersive Culinary", "score": 0.65},
    {"label": "Integrated Wellness", "score": 0.35},
    {"label": "Off-the-Beaten-Path Adventure", "score": 0.15}
]

MOCK_EXPLAIN = {
    "html": "<div class='shap-html'>Mock SHAP HTML</div>",
    "top_words": {
        "Regenerative & Eco-Tourism": [
            {"word": "eco-friendly", "value": 0.8, "is_positive": True},
            {"word": "solar", "value": 0.7, "is_positive": True},
            {"word": "organic", "value": 0.6, "is_positive": True}
        ]
    }
}


def test_read_root():
    """Test the root endpoint for health check"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "active"


@patch("api.classifier")
@patch("api.model")
@patch("api.tokenizer")
def test_predict_endpoint(mock_tokenizer, mock_model, mock_classifier):
    """Test the predict endpoint with mocked model"""
    # Setup mock
    mock_classifier.return_value = [{"LABEL_0": 0.85, "LABEL_1": 0.35, "LABEL_2": 0.65, "LABEL_3": 0.15}]
    
    # Test the endpoint
    response = client.post(
        "/predict",
        json={"review_text": SAMPLE_REVIEW}
    )
    
    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert len(result) == 4
    
    # Verify the structure of the response
    for item in result:
        assert "label" in item
        assert "score" in item
        assert item["label"] in MOCK_DIMENSIONS
        assert isinstance(item["score"], float)
        assert 0 <= item["score"] <= 1

    # Make sure classifier was called
    mock_classifier.assert_called_once()


@patch("api.shap.plots.force")
@patch("api.shap.Explainer")
@patch("api.model")
@patch("api.tokenizer")
def test_explain_endpoint(mock_tokenizer, mock_model, mock_explainer, mock_force_plot):
    """Test the explain endpoint with mocked SHAP explainer"""
    # Setup mocks
    mock_explainer_instance = MagicMock()
    mock_explainer.return_value = mock_explainer_instance
    
    # Mock the SHAP values
    mock_shap_values = MagicMock()
    mock_explainer_instance.__getitem__.return_value = mock_shap_values
    mock_explainer_instance.return_value = mock_shap_values
    
    # Mock the force plot
    mock_force_plot.return_value = MOCK_EXPLAIN["html"]
    
    # Test the endpoint
    response = client.post(
        "/explain",
        json={"review_text": SAMPLE_REVIEW, "top_n_words": 3}
    )
    
    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert "html" in result
    assert "top_words" in result


def test_predict_with_empty_text():
    """Test the predict endpoint with empty text"""
    response = client.post(
        "/predict",
        json={"review_text": ""}
    )
    # Should still work, even with empty text
    assert response.status_code == 200


def test_explain_with_invalid_request():
    """Test the explain endpoint with invalid request"""
    # Missing required field
    response = client.post(
        "/explain",
        json={}
    )
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main()
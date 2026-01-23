from fastapi.testclient import TestClient
from mlops_exam_project.api import app
from unittest.mock import patch, MagicMock
import torch

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status_code": 200}


def test_predict_dummy():
    # Test the /predict endpoint with dummy features
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response


def test_predict_with_mock_model():
    """Test predict endpoint with mocked model."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    mock_model = MagicMock()
    mock_output = torch.randn(1, 10)
    mock_model.return_value = mock_output

    with patch("mlops_exam_project.api.WineQualityClassifier", mock_model):
        response = client.post("/predict", json=features)
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response
        assert json_response["status_code"] == 200


def test_predict_response_structure():
    """Test that predict response has correct structure."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = client.post("/predict", json=features)
    json_response = response.json()

    assert "message" in json_response
    assert "status_code" in json_response
    assert "prediction" in json_response
    assert json_response["status_code"] == 200


def test_predict_onnx_endpoint_exists():
    """Test that ONNX predict endpoint exists and returns proper structure."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    response = client.post("/predict_onnx", json=features)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "status_code" in json_response


def test_predict_error_handling():
    """Test that predict endpoint handles errors gracefully."""
    # Empty features should still return a response
    features = {}
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response


def test_predict_with_model_error():
    """Test predict when model processing fails."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    mock_model = MagicMock()
    mock_model.side_effect = RuntimeError("Model error")

    with patch("mlops_exam_project.api.WineQualityClassifier", mock_model):
        response = client.post("/predict", json=features)
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["prediction"] == {"quality": 5}


def test_predict_with_tensor_conversion():
    """Test predict endpoint handles tensor conversion."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response


def test_predict_with_multiple_samples():
    """Test predict with different feature sets."""
    feature_sets = [
        {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11,
            "total_sulfur_dioxide": 34,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        },
        {
            "fixed_acidity": 8.5,
            "volatile_acidity": 0.5,
            "citric_acid": 0.1,
            "residual_sugar": 2.0,
            "chlorides": 0.08,
            "free_sulfur_dioxide": 13,
            "total_sulfur_dioxide": 40,
            "density": 0.9980,
            "pH": 3.4,
            "sulphates": 0.6,
            "alcohol": 10.0,
        },
    ]

    for features in feature_sets:
        response = client.post("/predict", json=features)
        assert response.status_code == 200
        json_response = response.json()
        assert "prediction" in json_response


def test_root_response_values():
    """Test that root endpoint returns correct values."""
    response = client.get("/")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "OK"
    assert json_response["status_code"] == 200


def test_predict_returns_json():
    """Test that predict returns valid JSON."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = client.post("/predict", json=features)

    # Verify response is valid JSON
    json_response = response.json()
    assert isinstance(json_response, dict)
    assert isinstance(json_response["message"], str)
    assert isinstance(json_response["status_code"], int)


def test_onnx_endpoint_response_structure():
    """Test ONNX endpoint response structure."""
    features = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = client.post("/predict_onnx", json=features)
    json_response = response.json()

    assert "message" in json_response
    assert "status_code" in json_response
    assert "prediction" in json_response

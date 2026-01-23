from fastapi.testclient import TestClient
from mlops_exam_project.api import app

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

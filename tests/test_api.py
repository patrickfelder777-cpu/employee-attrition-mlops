
import pandas as pd
import pytest

from src.api import app, FEATURES


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict(client):
    df = pd.read_csv("data/raw/employee_attrition.csv")
    employee = df[FEATURES].iloc[0].to_dict()

    response = client.post("/predict", json=employee)
    result = response.get_json()

    assert response.status_code == 200
    assert result["prediction"] in [0, 1]
    assert "probabilities" in result
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_predict_missing_features(client):
    response = client.post("/predict", json={"Age": 35})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing required features"


def test_predict_invalid_json(client):
    response = client.post("/predict", data="invalid", content_type="application/json")

    assert response.status_code == 400
    assert response.get_json()["error"] == "Expected a JSON object"

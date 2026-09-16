from fastapi.testclient import TestClient
import numpy as np

from code.deployment.api import main


class FakeModel:
    def predict_proba(self, _):
        return np.array([[0.25, 0.75]])


def test_prediction_contract():
    main.package.update({
        "model": FakeModel(),
        "metadata": {"threshold": 0.5, "target_definition": "test target"},
    })
    client = TestClient(main.app)
    response = client.post("/predict", json={
        "short_description": "A sufficiently long game description",
        "genre": "Action",
        "tags": "Co-op",
        "categories": "Single-player",
        "price": 10,
        "platforms": ["windows"],
        "languages": ["English"],
        "required_age": 0,
    })
    assert response.status_code == 200
    assert response.json()["success_probability"] == 0.75
    assert response.json()["prediction"] == 1

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_path" in data


def test_detect_image_invalid():
    response = client.post("/detect_image", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
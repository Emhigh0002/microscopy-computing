import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import SessionLocal, engine, Base
from app.models import User, Model, DetectionSession, SessionFrame
from app.core.security import get_password_hash

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    # Make sure tables are created
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Create a test user
        test_user = db.query(User).filter(User.email == "test_tech@microscopy.org").first()
        if not test_user:
            test_user = User(
                email="test_tech@microscopy.org",
                hashed_password=get_password_hash("password123"),
                full_name="Test Technician",
                role="Researcher"
            )
            db.add(test_user)
            db.commit()

        # Create a test model
        test_model = db.query(Model).filter(Model.name == "YOLOv8-Test").first()
        if not test_model:
            test_model = Model(
                name="YOLOv8-Test",
                version="1.0.0",
                type="detection",
                status="active"
            )
            db.add(test_model)
            db.commit()
            
        yield db
    finally:
        db.close()

def get_auth_headers(email="test_tech@microscopy.org", password="password123"):
    res = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_trigger_retraining_epochs_validation(test_db):
    headers = get_auth_headers()
    # Trigger with invalid epochs (<= 0)
    res = client.post(
        "/api/v1/training/retrain",
        headers=headers,
        json={"model_id": "any", "epochs": 0, "learning_rate": 0.001}
    )
    assert res.status_code == 400
    assert "Epochs must be greater than 0" in res.json()["detail"]

def test_session_creation_and_lifecycle(test_db):
    headers = get_auth_headers()
    
    # 1. Start live session
    res = client.post(
        "/api/v1/video/sessions/start",
        headers=headers,
        json={"camera_source": "0", "model_id": None}
    )
    assert res.status_code == 201
    data = res.json()
    session_id = data["id"]
    assert data["status"] == "active"
    assert data["camera_source"] == "0"
    
    # 2. Verify it is listed in sessions
    list_res = client.get("/api/v1/video/sessions", headers=headers)
    assert list_res.status_code == 200
    session_ids = [s["id"] for s in list_res.json()]
    assert session_id in session_ids
    
    # 3. Stop session
    stop_res = client.post(f"/api/v1/video/sessions/{session_id}/stop", headers=headers)
    assert stop_res.status_code == 200
    stop_data = stop_res.json()
    assert stop_data["status"] == "completed"
    assert stop_data["ended_at"] is not None

def test_concurrent_sessions_limit(test_db):
    headers = get_auth_headers()
    
    # Clean up existing active sessions first
    db = test_db
    db.query(DetectionSession).filter(DetectionSession.status == "active").update({"status": "completed"})
    db.commit()
    
    # Start Session 1
    res1 = client.post("/api/v1/video/sessions/start", headers=headers, json={"camera_source": "0"})
    assert res1.status_code == 201
    s1_id = res1.json()["id"]
    
    # Start Session 2
    res2 = client.post("/api/v1/video/sessions/start", headers=headers, json={"camera_source": "1"})
    assert res2.status_code == 201
    s2_id = res2.json()["id"]
    
    # Start Session 3 (should fail due to cap of 2 concurrent sessions)
    res3 = client.post("/api/v1/video/sessions/start", headers=headers, json={"camera_source": "2"})
    assert res3.status_code == 400
    assert "concurrent live sessions" in res3.json()["detail"]
    
    # Clean up
    client.post(f"/api/v1/video/sessions/{s1_id}/stop", headers=headers)
    client.post(f"/api/v1/video/sessions/{s2_id}/stop", headers=headers)

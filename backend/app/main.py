import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.database import engine, Base, SessionLocal
from app.api import auth, images, annotations, predictions, reports, assistant, training, camera
from app.models import User, Model
from app.core.security import get_password_hash

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Middleware (crucial for React/Next.js frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins in dev, customize in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(images.router, prefix=f"{settings.API_V1_STR}/images", tags=["Microscopy Images"])
app.include_router(camera.router, prefix=f"{settings.API_V1_STR}/camera", tags=["Digital Microscope Cameras"])
app.include_router(annotations.router, prefix=f"{settings.API_V1_STR}/annotations", tags=["Corrections & Annotations"])
app.include_router(predictions.router, prefix=f"{settings.API_V1_STR}/predictions", tags=["AI Predictions"])
app.include_router(reports.router, prefix=f"{settings.API_V1_STR}/reports", tags=["Export Reporting"])
app.include_router(assistant.router, prefix=f"{settings.API_V1_STR}/assistant", tags=["AI Clinical Assistant"])
app.include_router(training.router, prefix=f"{settings.API_V1_STR}/training", tags=["Model Retraining"])

@app.on_event("startup")
def startup_populate_data():
    """
    Pre-populates default database models and a default user on startup for testing.
    """
    db = SessionLocal()
    try:
        # 1. Create Default Researcher
        admin = db.query(User).filter(User.email == "researcher@laboratory.org").first()
        if not admin:
            default_user = User(
                email="researcher@laboratory.org",
                hashed_password=get_password_hash("password123"),
                full_name="Dr. Sarah Jenkins",
                role="Researcher"
            )
            db.add(default_user)
            db.commit()

        # 2. Create Base Active Model
        model = db.query(Model).filter(Model.name == "YOLOv8-Microbiology").first()
        if not model:
            default_model = Model(
                name="YOLOv8-Microbiology",
                version="1.0.0",
                type="detection",
                metrics={
                    "accuracy": 0.88,
                    "precision": 0.87,
                    "recall": 0.85,
                    "f1": 0.86,
                    "mAP": 0.84,
                    "iou": 0.76
                },
                status="active"
            )
            db.add(default_model)
            db.commit()
            
    except Exception as e:
        print(f"Error seeding DB: {e}")
    finally:
        db.close()

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

import time
import uuid
import threading
from typing import List
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Model, Annotation, User, AuditLog
from app.schemas import RetrainRequest, RetrainResponse, ModelResponse
from app.api import deps

router = APIRouter()

# Keep track of active jobs in-memory for status checks
training_jobs = {}

def simulate_training_job(job_id: str, model_id: str, epochs: int, db_session_maker):
    """
    Simulates training cycles. Updates model metrics in DB and records progress.
    """
    training_jobs[job_id] = {"status": "Running", "progress": 0, "metrics": {}}
    
    # 1. Fetch training datasets (corrected annotations)
    db = db_session_maker()
    try:
        annotations_count = db.query(Annotation).count()
        
        # Simulated epoch training
        for epoch in range(1, epochs + 1):
            time.sleep(1.5) # simulate GPU cycles per epoch
            
            # Progress calculation
            progress = int((epoch / epochs) * 100)
            
            # Simulated improving metrics
            base_acc = 0.72 + (epoch / epochs) * 0.18
            base_map = 0.68 + (epoch / epochs) * 0.22
            
            training_jobs[job_id]["progress"] = progress
            training_jobs[job_id]["metrics"] = {
                "epoch": epoch,
                "loss": round(1.2 / epoch, 4),
                "accuracy": round(min(0.98, base_acc), 4),
                "precision": round(min(0.97, base_acc - 0.02), 4),
                "recall": round(min(0.96, base_acc - 0.03), 4),
                "mAP": round(min(0.98, base_map), 4)
            }
            
        # 2. Complete training, save new model variant
        training_jobs[job_id]["status"] = "Completed"
        
        # Retrieve base model to clone version
        base_model = db.query(Model).filter(Model.id == model_id).first()
        version_num = "1.0.0"
        name = "YOLOv8-Microbiology"
        if base_model:
            name = base_model.name
            try:
                major, minor, patch = map(int, base_model.version.split('.'))
                version_num = f"{major}.{minor + 1}.0"
            except:
                version_num = f"{base_model.version}-fine-tuned"
        
        # Write new model state
        new_model = Model(
            name=name,
            version=version_num,
            type="detection",
            status="active",
            metrics=training_jobs[job_id]["metrics"]
        )
        db.add(new_model)
        
        # Archive older active model version
        if base_model:
            base_model.status = "archived"
            
        db.commit()
        
    except Exception as e:
        training_jobs[job_id]["status"] = "Failed"
        training_jobs[job_id]["message"] = str(e)
    finally:
        db.close()

@router.post("/retrain", response_model=RetrainResponse)
def trigger_retraining(
    payload: RetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    # Verify model exists
    model = db.query(Model).filter(Model.id == payload.model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Base model not found"
        )
        
    # Check if there are annotations to train on
    annot_count = db.query(Annotation).count()
    if annot_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot train model: No corrected annotations found in the database. Annotate some images first."
        )
        
    job_id = f"job_{uuid.uuid4()}"
    
    # Spawn background training threads
    from app.database import SessionLocal
    background_tasks.add_task(
        simulate_training_job,
        job_id,
        payload.model_id,
        payload.epochs,
        SessionLocal
    )
    
    # Audit log
    log = AuditLog(
        user_id=current_user.id,
        action="trigger_retraining",
        details={"model_id": payload.model_id, "job_id": job_id, "epochs": payload.epochs}
    )
    db.add(log)
    db.commit()
    
    return {
        "job_id": job_id,
        "status": "Submitted",
        "message": f"Retraining job started on {annot_count} corrected microorganism annotations."
    }

@router.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    current_user: User = Depends(deps.get_current_active_user)
):
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    return training_jobs[job_id]

@router.get("/active-models", response_model=List[ModelResponse])
def get_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    return db.query(Model).order_by(Model.created_at.desc()).all()

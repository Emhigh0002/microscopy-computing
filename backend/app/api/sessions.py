from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import DetectionSession, SessionFrame, User
from app.schemas import DetectionSessionCreate, DetectionSessionResponse, SessionFrameCreate, SessionFrameResponse
from app.api import deps

router = APIRouter()

@router.post("/start", response_model=DetectionSessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(
    payload: DetectionSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Starts a new real-time camera microscope detection session.
    """
    session = DetectionSession(
        user_id=current_user.id,
        model_id=payload.model_id,
        camera_source=payload.camera_source or "0",
        status="active",
        started_at=datetime.utcnow(),
        class_counts={},
        avg_confidence=0.0,
        total_frames_processed=0,
        total_frames_dropped=0,
        total_detections=0,
        achieved_inference_fps=0.0
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.post("/{session_id}/frame", response_model=SessionFrameResponse)
def log_session_frame(
    session_id: str,
    payload: SessionFrameCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Logs a processed frame with its AI detections and overlays.
    Automatically increments frame count and calculates rolling session metrics.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Detection session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Cannot log frames to an inactive session")

    frame = SessionFrame(
        session_id=session_id,
        timestamp=datetime.utcnow(),
        frame_path=payload.frame_path,
        detections=payload.detections
    )
    db.add(frame)
    
    # Increment frame count
    session.total_frames_processed += 1
    
    # Update detections count
    detections_count = len(payload.detections)
    session.total_detections += detections_count
    
    # Calculate rolling metrics
    if detections_count > 0:
        total_conf = sum(float(d.get("confidence", 0.0)) for d in payload.detections)
        # Update rolling average confidence
        prev_count = session.total_detections - detections_count
        if session.total_detections > 0:
            current_sum = session.avg_confidence * prev_count
            session.avg_confidence = (current_sum + total_conf) / session.total_detections
        
        # Merge rolling class counts
        counts = dict(session.class_counts or {})
        for d in payload.detections:
            cls = d.get("class", "unknown")
            counts[cls] = counts.get(cls, 0) + 1
        session.class_counts = counts
        
    db.commit()
    db.refresh(frame)
    return frame

@router.post("/{session_id}/end", response_model=DetectionSessionResponse)
def end_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Ends the real-time detection session and computes final metrics (e.g. inference FPS).
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Detection session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is already ended")
        
    session.status = "completed"
    session.ended_at = datetime.utcnow()
    
    # Compute achieved FPS
    duration = (session.ended_at - session.started_at).total_seconds()
    if duration > 0 and session.total_frames_processed > 0:
        session.achieved_inference_fps = round(session.total_frames_processed / duration, 2)
        
    db.commit()
    db.refresh(session)
    return session

@router.get("/", response_model=List[DetectionSessionResponse])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Lists all past real-time digital microscope detection sessions.
    """
    return db.query(DetectionSession).order_by(DetectionSession.started_at.desc()).all()

@router.get("/{session_id}", response_model=DetectionSessionResponse)
def get_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Retrieves full details of a specific microscope detection session.
    """
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Detection session not found")
    return session

@router.get("/{session_id}/frames", response_model=List[SessionFrameResponse])
def get_session_frames(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Retrieves all logged frames and detections captured during a session.
    """
    return db.query(SessionFrame).filter(SessionFrame.session_id == session_id).order_by(SessionFrame.timestamp.asc()).all()

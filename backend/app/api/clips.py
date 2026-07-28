import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import ClipExport, User, AuditLog
from app.schemas import ClipExportRequest, ClipExportResponse
from app.api import deps
from app.services.video import process_background_clip_export

router = APIRouter()

@router.post("/export", response_model=ClipExportResponse, status_code=status.HTTP_202_ACCEPTED)
def request_clip_export(
    payload: ClipExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Schedules an asynchronous background task for rendering and exporting video clips.
    Returns HTTP 202 Accepted immediately without blocking the request/response cycle.
    """
    db_clip = ClipExport(
        user_id=current_user.id,
        image_id=payload.image_id,
        file_name=f"Microscopy_Clip_{payload.image_id[:8] if payload.image_id else 'stream'}.mp4",
        format=payload.format or "mp4",
        duration_seconds=payload.duration_seconds or 5.0,
        draw_overlays=payload.draw_overlays if payload.draw_overlays is not None else True,
        status="queued",
        progress=0.0
    )
    db.add(db_clip)
    db.commit()
    db.refresh(db_clip)
    
    # Offload clip rendering to background task
    background_tasks.add_task(process_background_clip_export, db_clip.id)
    
    # Audit log
    db.add(AuditLog(
        user_id=current_user.id,
        action="request_clip_export",
        details={"clip_id": db_clip.id, "duration": db_clip.duration_seconds}
    ))
    db.commit()
    
    return db_clip

@router.get("/status/{clip_id}", response_model=ClipExportResponse)
def get_clip_export_status(
    clip_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Returns current rendering progress and status for an exported clip task.
    """
    clip = db.query(ClipExport).filter(ClipExport.id == clip_id).first()
    if not clip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip export task not found"
        )
    return clip

@router.get("/{clip_id}/download")
def download_clip_export(
    clip_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Downloads a completed video clip export.
    """
    clip = db.query(ClipExport).filter(ClipExport.id == clip_id).first()
    if not clip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clip export not found"
        )
    if clip.status != "completed" or not clip.file_path or not os.path.exists(clip.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Clip export processing is not yet completed"
        )
    return FileResponse(clip.file_path, media_type="video/mp4", filename=clip.file_name)

@router.get("/", response_model=List[ClipExportResponse])
def list_clip_exports(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Lists all clip export tasks for current user.
    """
    return db.query(ClipExport).order_by(ClipExport.created_at.desc()).all()

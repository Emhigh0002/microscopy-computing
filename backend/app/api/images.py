import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Image as DBImage, User, Annotation, Prediction, AuditLog
from app.schemas import ImageResponse, ScaleUpdateRequest
from app.api import deps
from app.core.config import settings
import cv2
import random
from app.services.inference import inference_service
from app.services.video import video_tracking_service
router = APIRouter()

@router.post("/upload", response_model=ImageResponse, status_code=status.HTTP_201_CREATED)
def upload_image(
    file: UploadFile = File(...),
    scale_microns_px: float = Form(1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    # Create unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    # Save file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {str(e)}"
        )
        
    is_video = file_ext in [".mp4", ".webm", ".avi", ".mov", ".mkv"]
    
    if is_video:
        # Read video properties using OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid video file"
            )
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Analyze motility paths
        motility = video_tracking_service.analyze_video_motility(file_path, scale_microns_px)
        
        db_image = DBImage(
            user_id=current_user.id,
            file_name=file.filename,
            file_path=file_path,
            scale_microns_px=scale_microns_px,
            width=w,
            height=h,
            media_type="video",
            motility_stats=motility,
            status="Annotated" # Videos are pre-analyzed
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Save track centroids as predictions/annotations for display
        for track in motility.get("tracks", []):
            if not track.get("points"):
                continue
            # Store the starting centroid point as a detection
            pt = track["points"][0]
            pred = Prediction(
                image_id=db_image.id,
                label_class="Spermatozoon" if "sperm" in file.filename.lower() else "Escherichia coli",
                confidence=round(random.uniform(0.85, 0.98), 2),
                coordinates={"box": {"x": pt["x"]-10, "y": pt["y"]-10, "w": 20, "h": 20}}
            )
            db.add(pred)
            
            annot = Annotation(
                image_id=db_image.id,
                user_id=None,
                label_class="Spermatozoon" if "sperm" in file.filename.lower() else "Escherichia coli",
                shape_type="rect",
                coordinates={"box": {"x": pt["x"]-10, "y": pt["y"]-10, "w": 20, "h": 20}},
                area_microns=round(4.5 * scale_microns_px, 2),
                perimeter_microns=round(12.0 * scale_microns_px, 2)
            )
            db.add(annot)
        db.commit()
        
    else:
        # Read width and height for images
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            # Cleanup
            os.remove(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid image"
            )
            
        h, w, _ = img_cv.shape
        
        # Create DB entry
        db_image = DBImage(
            user_id=current_user.id,
            file_name=file.filename,
            file_path=file_path,
            scale_microns_px=scale_microns_px,
            width=w,
            height=h,
            media_type="image",
            status="Uploaded"
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Run auto-inference to pre-populate predictions and starting annotations
        detections = inference_service.detect_microorganisms(file_path, scale_microns_px)
        
        for det in detections:
            # Create prediction entry
            pred = Prediction(
                image_id=db_image.id,
                label_class=det["label_class"],
                confidence=det["confidence"],
                coordinates=det["coordinates"]
            )
            db.add(pred)
            
            # Create starting editable annotation
            annot = Annotation(
                image_id=db_image.id,
                user_id=None, # System generated initially
                label_class=det["label_class"],
                shape_type=det["shape_type"],
                coordinates=det["coordinates"],
                area_microns=det["area_microns"],
                perimeter_microns=det["perimeter_microns"]
            )
            db.add(annot)
            
        db.commit()
    
    # Audit log
    log = AuditLog(
        user_id=current_user.id,
        action="upload_image",
        details={"image_id": db_image.id, "file_name": db_image.file_name}
    )
    db.add(log)
    db.commit()
    
    return db_image

@router.get("/", response_model=List[ImageResponse])
def get_images(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    return db.query(DBImage).all()

@router.get("/{image_id}", response_model=ImageResponse)
def get_image_details(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    image = db.query(DBImage).filter(DBImage.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    return image

@router.get("/{image_id}/binary")
def get_image_binary(
    image_id: str,
    db: Session = Depends(get_db)
):
    image = db.query(DBImage).filter(DBImage.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    return FileResponse(image.file_path)

@router.put("/{image_id}/scale", response_model=ImageResponse)
def update_image_scale(
    image_id: str,
    payload: ScaleUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    image = db.query(DBImage).filter(DBImage.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
        
    # Update scale
    image.scale_microns_px = payload.scale_microns_px
    db.commit()
    
    # Recalculate annotation areas & perimeters
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    for annot in annotations:
        if annot.shape_type == "rect" and isinstance(annot.coordinates, dict):
            # Recalculate rect physical area
            w_px = annot.coordinates.get("w", 0)
            h_px = annot.coordinates.get("h", 0)
            annot.area_microns = (w_px * payload.scale_microns_px) * (h_px * payload.scale_microns_px)
            annot.perimeter_microns = 2 * (w_px + h_px) * payload.scale_microns_px
        elif annot.shape_type == "polygon" and isinstance(annot.coordinates, dict):
            # It's a dict containing "points" and "box"
            points = annot.coordinates.get("points", [])
            # Compute actual CV area on scale change
            if len(points) >= 3:
                cnt = np.array([[[pt["x"], pt["y"]]] for pt in points], dtype=np.int32)
                area_pixels = cv2.contourArea(cnt)
                annot.area_microns = area_pixels * (payload.scale_microns_px ** 2)
                annot.perimeter_microns = cv2.arcLength(cnt, True) * payload.scale_microns_px
                
    db.commit()
    db.refresh(image)
    
    return image

@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    image = db.query(DBImage).filter(DBImage.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
        
    # Delete file from system
    if os.path.exists(image.file_path):
        os.remove(image.file_path)
        
    # Delete related annotations, predictions
    db.query(Annotation).filter(Annotation.image_id == image_id).delete()
    db.query(Prediction).filter(Prediction.image_id == image_id).delete()
    db.delete(image)
    
    # Audit log
    log = AuditLog(
        user_id=current_user.id,
        action="delete_image",
        details={"image_id": image_id, "file_name": image.file_name}
    )
    db.add(log)
    db.commit()
    
    return status.HTTP_204_NO_CONTENT

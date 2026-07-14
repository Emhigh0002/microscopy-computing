from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Annotation, Image, User, AuditLog
from app.schemas import AnnotationCreate, AnnotationResponse
from app.api import deps
import numpy as np
import cv2

router = APIRouter()

@router.get("/image/{image_id}", response_model=List[AnnotationResponse])
def get_annotations_for_image(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    # Verify image exists
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
    return db.query(Annotation).filter(Annotation.image_id == image_id).all()

@router.post("/image/{image_id}", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
def add_annotation(
    image_id: str,
    annotation_in: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
        
    # Calculate area and perimeter from coordinates if not provided
    area = annotation_in.area_microns
    perimeter = annotation_in.perimeter_microns
    
    if (area == 0.0 or perimeter == 0.0):
        scale = image.scale_microns_px
        if annotation_in.shape_type == "rect" and isinstance(annotation_in.coordinates, dict):
            w = annotation_in.coordinates.get("w", 0)
            h = annotation_in.coordinates.get("h", 0)
            area = (w * scale) * (h * scale)
            perimeter = 2 * (w + h) * scale
        elif annotation_in.shape_type == "polygon" and isinstance(annotation_in.coordinates, dict):
            points = annotation_in.coordinates.get("points", [])
            if len(points) >= 3:
                cnt = np.array([[[pt["x"], pt["y"]]] for pt in points], dtype=np.int32)
                area_pixels = cv2.contourArea(cnt)
                area = area_pixels * (scale ** 2)
                perimeter = cv2.arcLength(cnt, True) * scale

    db_annotation = Annotation(
        image_id=image_id,
        user_id=current_user.id,
        label_class=annotation_in.label_class,
        shape_type=annotation_in.shape_type,
        coordinates=annotation_in.coordinates,
        area_microns=area,
        perimeter_microns=perimeter
    )
    
    db.add(db_annotation)
    db.commit()
    db.refresh(db_annotation)
    
    # Mark image status as annotated
    image.status = "Annotated"
    db.commit()
    
    return db_annotation

@router.put("/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(
    annotation_id: str,
    annotation_in: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    annot = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found"
        )
        
    image = db.query(Image).filter(Image.id == annot.image_id).first()
    scale = image.scale_microns_px if image else 1.0
    
    # Recalculate physical dimensions
    area = annotation_in.area_microns
    perimeter = annotation_in.perimeter_microns
    if (area == 0.0 or perimeter == 0.0):
        if annotation_in.shape_type == "rect" and isinstance(annotation_in.coordinates, dict):
            w = annotation_in.coordinates.get("w", 0)
            h = annotation_in.coordinates.get("h", 0)
            area = (w * scale) * (h * scale)
            perimeter = 2 * (w + h) * scale
        elif annotation_in.shape_type == "polygon" and isinstance(annotation_in.coordinates, dict):
            points = annotation_in.coordinates.get("points", [])
            if len(points) >= 3:
                cnt = np.array([[[pt["x"], pt["y"]]] for pt in points], dtype=np.int32)
                area_pixels = cv2.contourArea(cnt)
                area = area_pixels * (scale ** 2)
                perimeter = cv2.arcLength(cnt, True) * scale

    annot.label_class = annotation_in.label_class
    annot.shape_type = annotation_in.shape_type
    annot.coordinates = annotation_in.coordinates
    annot.area_microns = area
    annot.perimeter_microns = perimeter
    annot.user_id = current_user.id
    
    db.commit()
    db.refresh(annot)
    return annot

@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    annot = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotation not found"
        )
        
    db.delete(annot)
    db.commit()
    return status.HTTP_204_NO_CONTENT

@router.post("/image/{image_id}/sync", response_model=List[AnnotationResponse])
def sync_annotations(
    image_id: str,
    annotations_in: List[AnnotationCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Overwrites all annotations for a given image. 
    Useful when saving bulk corrections from the annotation tool editor interface.
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
        
    # Delete existing annotations
    db.query(Annotation).filter(Annotation.image_id == image_id).delete()
    
    # Insert new ones
    db_annotations = []
    scale = image.scale_microns_px
    
    for item in annotations_in:
        area = item.area_microns
        perimeter = item.perimeter_microns
        
        if (area == 0.0 or perimeter == 0.0):
            if item.shape_type == "rect" and isinstance(item.coordinates, dict):
                w = item.coordinates.get("w", 0)
                h = item.coordinates.get("h", 0)
                area = (w * scale) * (h * scale)
                perimeter = 2 * (w + h) * scale
            elif item.shape_type == "polygon" and isinstance(item.coordinates, dict):
                points = item.coordinates.get("points", [])
                if len(points) >= 3:
                    cnt = np.array([[[pt["x"], pt["y"]]] for pt in points], dtype=np.int32)
                    area_pixels = cv2.contourArea(cnt)
                    area = area_pixels * (scale ** 2)
                    perimeter = cv2.arcLength(cnt, True) * scale
        
        db_annot = Annotation(
            image_id=image_id,
            user_id=current_user.id,
            label_class=item.label_class,
            shape_type=item.shape_type,
            coordinates=item.coordinates,
            area_microns=area,
            perimeter_microns=perimeter
        )
        db.add(db_annot)
        db_annotations.append(db_annot)
        
    db.commit()
    
    # Mark image status as annotated
    image.status = "Annotated"
    db.commit()
    
    # Audit log
    log = AuditLog(
        user_id=current_user.id,
        action="sync_annotations",
        details={"image_id": image_id, "count": len(db_annotations)}
    )
    db.add(log)
    db.commit()
    
    # Retrieve newly inserted items
    return db.query(Annotation).filter(Annotation.image_id == image_id).all()

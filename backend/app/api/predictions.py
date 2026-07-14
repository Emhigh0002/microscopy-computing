from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Prediction, Image, User
from app.schemas import PredictionResponse
from app.api import deps
from app.services.xai import xai_service

router = APIRouter()

@router.get("/image/{image_id}", response_model=List[PredictionResponse])
def get_predictions_for_image(
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
    return db.query(Prediction).filter(Prediction.image_id == image_id).all()

@router.get("/image/{image_id}/explain")
def get_explainable_heatmap(
    image_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    """
    Returns a simulated Grad-CAM / SHAP activation heatmap overlaid on the original image
    to explain the features the AI model focused on during microorganisms detection.
    Returns JPEG base64 data url.
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found"
        )
        
    predictions = db.query(Prediction).filter(Prediction.image_id == image_id).all()
    
    # Format predictions for service
    formatted_preds = []
    for pred in predictions:
        formatted_preds.append({
            "label_class": pred.label_class,
            "confidence": pred.confidence,
            "coordinates": pred.coordinates
        })
        
    # Generate overlay base64 string
    base64_heatmap = xai_service.generate_heatmap_overlay(image.file_path, formatted_preds)
    
    if not base64_heatmap:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate explainability overlay"
        )
        
    return {"heatmap_image_url": base64_heatmap}

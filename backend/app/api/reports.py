import os
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Report, Image, Annotation, User, AuditLog
from app.schemas import ReportResponse, ReportCreateRequest
from app.api import deps
from app.core.config import settings
from app.services.reports import report_service

router = APIRouter()

@router.post("/generate", response_model=ReportResponse)
def generate_report(
    payload: ReportCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    if len(payload.image_ids) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify at least one image ID to include in the report"
        )
        
    # Retrieve images and annotations
    images_data = []
    organism_stats = {}
    
    for img_id in payload.image_ids:
        image = db.query(Image).filter(Image.id == img_id).first()
        if not image:
            continue
            
        annotations = db.query(Annotation).filter(Annotation.image_id == img_id).all()
        
        detections = []
        for idx, annot in enumerate(annotations):
            detections.append({
                "label_class": annot.label_class,
                "confidence": 1.0,  # Human verified count or default 1.0
                "area_microns": annot.area_microns,
                "perimeter_microns": annot.perimeter_microns,
                "shape_type": annot.shape_type
            })
            
            # Aggregate stats
            lbl = annot.label_class
            if lbl not in organism_stats:
                organism_stats[lbl] = {"count": 0, "areas": [], "confidences": []}
            organism_stats[lbl]["count"] += 1
            organism_stats[lbl]["areas"].append(annot.area_microns)
            organism_stats[lbl]["confidences"].append(1.0)
            
        images_data.append({
            "name": image.file_name,
            "width": image.width,
            "height": image.height,
            "scale": image.scale_microns_px,
            "detections": detections
        })
        
    # Compile summary statistics
    summary = {}
    for org, stats in organism_stats.items():
        summary[org] = {
            "count": stats["count"],
            "mean_area": sum(stats["areas"]) / len(stats["areas"]) if stats["areas"] else 0.0,
            "mean_confidence": sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 1.0
        }
        
    report_data = {
        "operator_name": current_user.full_name or current_user.email,
        "images": images_data,
        "organism_summary": summary
    }
    
    # Save file
    file_ext = payload.format.lower()
    if file_ext == "xlsx":
        file_ext = "xlsx"
    unique_filename = f"report_{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(settings.REPORTS_DIR, unique_filename)
    
    if payload.format.upper() == "PDF":
        report_service.generate_pdf_report(file_path, report_data)
    elif payload.format.upper() in ["XLSX", "EXCEL"]:
        report_service.generate_excel_report(file_path, report_data)
    elif payload.format.upper() == "CSV":
        report_service.generate_csv_report(file_path, report_data)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report format: {payload.format}"
        )
        
    # Create DB entry
    db_report = Report(
        user_id=current_user.id,
        file_name=f"Microscopy_Report_{datetime_now_str()}.{file_ext}",
        file_path=file_path,
        format=payload.format.upper()
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Audit log
    log = AuditLog(
        user_id=current_user.id,
        action="generate_report",
        details={"report_id": db_report.id, "format": db_report.format}
    )
    db.add(log)
    db.commit()
    
    return db_report

@router.get("/", response_model=List[ReportResponse])
def get_reports(
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    return db.query(Report).all()

@router.get("/{report_id}/download")
def download_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(deps.get_current_active_user)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    return FileResponse(report.file_path, filename=report.file_name)

def datetime_now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

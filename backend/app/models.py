import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default="Researcher")  # Admin, Researcher, Pathologist
    created_at = Column(DateTime, default=datetime.utcnow)

class Image(Base):
    __tablename__ = "images"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    scale_microns_px = Column(Float, default=1.0) # microns per pixel
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    status = Column(String, default="Uploaded")  # Uploaded, Annotated, Verified, Retraining
    media_type = Column(String, default="image")  # image or video
    motility_stats = Column(JSON, nullable=True)  # For videos: progressive, non_progressive, immotile, tracks, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

class Annotation(Base):
    __tablename__ = "annotations"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("images.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    label_class = Column(String, nullable=False)
    shape_type = Column(String, nullable=False)  # rect, polygon
    coordinates = Column(JSON, nullable=False)  # For rect: {x, y, w, h}, For polygon: [{x, y}, ...]
    area_microns = Column(Float, default=0.0)
    perimeter_microns = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    image_id = Column(String, ForeignKey("images.id"), nullable=False)
    model_id = Column(String, ForeignKey("models.id"), nullable=True)
    label_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    coordinates = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    type = Column(String, nullable=False)  # detection, classification, segmentation
    file_path = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)  # {accuracy, precision, recall, f1, map, iou}
    status = Column(String, default="active")  # active, archived
    created_at = Column(DateTime, default=datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    format = Column(String, nullable=False)  # PDF, Excel, CSV
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)  # login, upload, annotate, delete, train, export
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

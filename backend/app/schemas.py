from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# --- AUTH SCHEMAS ---
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    user_id: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: Optional[str] = "Researcher"

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- IMAGE SCHEMAS ---
class ImageResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    file_name: str
    scale_microns_px: float
    width: int
    height: int
    status: str
    media_type: str
    motility_stats: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ScaleUpdateRequest(BaseModel):
    scale_microns_px: float

# --- ANNOTATION SCHEMAS ---
class AnnotationCreate(BaseModel):
    label_class: str
    shape_type: str
    coordinates: Union[Dict[str, float], List[Dict[str, float]]] # Dict for rect {x,y,w,h}, List for polygon
    area_microns: Optional[float] = 0.0
    perimeter_microns: Optional[float] = 0.0

class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    user_id: Optional[str] = None
    label_class: str
    shape_type: str
    coordinates: Any
    area_microns: float
    perimeter_microns: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- PREDICTION SCHEMAS ---
class PredictionResponse(BaseModel):
    id: str
    image_id: str
    model_id: Optional[str] = None
    label_class: str
    confidence: float
    coordinates: Any
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- MODEL SCHEMAS ---
class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    type: str
    metrics: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- REPORT SCHEMAS ---
class ReportResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    file_name: str
    format: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ReportCreateRequest(BaseModel):
    image_ids: List[str]
    format: str # PDF, XLSX, CSV

# --- CHAT ASSISTANT SCHEMAS ---
class AssistantRequest(BaseModel):
    query: str
    image_id: Optional[str] = None

class AssistantResponse(BaseModel):
    response: str
    citations: List[str]

# --- RETRAINING SCHEMAS ---
class RetrainRequest(BaseModel):
    model_id: str
    epochs: Optional[int] = 10
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 16

class RetrainResponse(BaseModel):
    job_id: str
    status: str
    message: str

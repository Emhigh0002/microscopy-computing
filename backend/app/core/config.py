import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "Microscopy Detection Platform"
    API_V1_STR: str = "/api/v1"
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "antigravity-microscopy-super-secret-key-change-in-prod")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./microscopy.db")
    
    # File Storage Settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "data/reports")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "data/models")
    
    # Create directories if they do not exist
    model_config = ConfigDict(case_sensitive=True)

settings = Settings()

# Ensure local storage folders exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)

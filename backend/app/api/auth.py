from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, AuditLog
from app.schemas import UserCreate, UserResponse, Token, UserLogin
from app.core import security
from app.api import deps
import json

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists"
        )
        
    hashed_password = security.get_password_hash(user_in.password)
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        role=user_in.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Audit log
    log = AuditLog(
        user_id=db_user.id,
        action="register",
        details={"email": db_user.email, "role": db_user.role}
    )
    db.add(log)
    db.commit()
    
    return db_user

@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Retrieve user
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
        
    access_token = security.create_access_token(subject=user.id)
    
    # Audit log
    log = AuditLog(
        user_id=user.id,
        action="login",
        details={"email": user.email}
    )
    db.add(log)
    db.commit()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "user_id": user.id
    }

@router.post("/login-json", response_model=Token)
def login_json(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not security.verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
        
    access_token = security.create_access_token(subject=user.id)
    
    log = AuditLog(
        user_id=user.id,
        action="login",
        details={"email": user.email}
    )
    db.add(log)
    db.commit()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "user_id": user.id
    }

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(deps.get_current_active_user)):
    return current_user

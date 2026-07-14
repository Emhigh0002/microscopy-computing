import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import jwt
from app.core.config import settings

def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        # Support format: salt_hex:hash_hex
        if ":" not in hashed_password:
            # Fallback for plain text or old formats during transition
            return plain_password == hashed_password
            
        salt_hex, hash_hex = hashed_password.split(":")
        salt = bytes.fromhex(salt_hex)
        # Compute PBKDF2 hash
        new_hash = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt, 100000)
        return new_hash.hex() == hash_hex
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = os.urandom(16)
    db_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return f"{salt.hex()}:{db_hash.hex()}"

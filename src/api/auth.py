"""
API Authentication Module
JWT-based authentication for API endpoints.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..utils.logger import get_logger
from ..utils.config import get_config


# Security schemes
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

logger = get_logger(__name__)
config = get_config()


def get_secret_key() -> str:
    """Get secret key from environment or config."""
    secret_key = os.getenv("API_SECRET_KEY")
    if not secret_key:
        secret_key = config.get("api.auth.secret_key", "dev-secret-key-change-in-production")
        if secret_key.startswith("${"):
            # Fallback if env var not set
            secret_key = "dev-secret-key-change-in-production"
            logger.warning("Using default secret key - NOT SECURE FOR PRODUCTION!")
    return secret_key


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire_minutes = config.get("api.auth.access_token_expire_minutes", 60)
        expire = datetime.utcnow() + timedelta(minutes=expire_minutes)

    to_encode.update({"exp": expire})

    algorithm = config.get("api.auth.algorithm", "HS256")
    secret_key = get_secret_key()

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)

    logger.info(f"Created access token, expires at {expire}")

    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        algorithm = config.get("api.auth.algorithm", "HS256")
        secret_key = get_secret_key()

        payload = jwt.decode(token, secret_key, algorithms=[algorithm])

        if payload.get("sub") is None:
            raise credentials_exception

        logger.info(f"Token verified for user: {payload.get('sub')}")

        return payload

    except JWTError as e:
        logger.error(f"JWT verification failed: {str(e)}")
        raise credentials_exception


def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


# Optional dependency for endpoints that support optional auth
async def optional_verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """
    Optionally verify token if provided.

    Args:
        credentials: Optional HTTP authorization credentials

    Returns:
        Decoded token payload if provided, None otherwise
    """
    if credentials is None:
        return None

    return verify_token(credentials)

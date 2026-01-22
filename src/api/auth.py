"""
API Authentication Module
JWT-based authentication for API endpoints.
"""

import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ..utils.logger import get_logger
from ..utils.config import get_config


# Security schemes
security = HTTPBearer()

logger = get_logger(__name__)
config = get_config()


def get_secret_key() -> str:
    """Get secret key from environment or config.

    In production mode, a proper secret key must be set via API_SECRET_KEY environment variable.
    """
    secret_key = os.getenv("API_SECRET_KEY")
    environment = os.getenv("API_ENVIRONMENT", "development").lower()

    if not secret_key:
        secret_key = config.get("api.auth.secret_key", "")

        # Check if secret key is still an unresolved env var placeholder
        if not secret_key or secret_key.startswith("${"):
            if environment == "production":
                error_msg = (
                    "SECURITY ERROR: API_SECRET_KEY environment variable must be set in production. "
                    "Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
                logger.critical(error_msg)
                raise ValueError(error_msg)
            else:
                # Only allow default key in development/testing
                secret_key = "dev-secret-key-change-in-production"
                logger.warning(
                    "Using default secret key for development. "
                    "Set API_SECRET_KEY environment variable before deploying to production!"
                )
    else:
        # Validate the secret key is sufficiently long
        if len(secret_key) < 32:
            logger.warning(
                f"API_SECRET_KEY is only {len(secret_key)} characters. "
                "Recommended minimum is 32 characters for security."
            )

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
    # Encode password to bytes and hash with bcrypt
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


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

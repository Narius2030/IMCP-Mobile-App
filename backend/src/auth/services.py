import sys
sys.path.append('./')

from fastapi.exceptions import HTTPException
from core.security import verify_password, create_access_token, get_token_payload, create_refresh_token
from core.config import get_settings
from src.auth.models import TokenResponse
from datetime import timedelta
import pymongo


settings = get_settings()

async def get_token(data):
    with pymongo.MongoClient(settings.DATABASE_URL) as client:
        db = client['imcp']
        user = db['user_account'].find_one({"username": data.username})
        # verify existing user
        if not user:
            raise HTTPException(
                status_code=400,
                detail="Username is not registered with us.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # verify valid password if user is existed
        if not verify_password(data.password, user['password']):
            raise HTTPException(
                status_code=400,
                detail="Invalid Login Credentials.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return await _get_user_token(user=user)


async def get_refresh_token(token:str):   
    payload = get_token_payload(token=token)
    
    username = payload.get('username', None)
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    with pymongo.MongoClient(settings.DATABASE_URL) as client:
        db = client['imcp']
        user = db['user_account'].find_one({"username": username})
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
    return await _get_user_token(user=user, refresh_token=token)


async def _get_user_token(user: dict, refresh_token=None):
    payload = {
        "username": user['username'], 
        "password": user['password']
    }
    
    access_token_expiry = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = await create_access_token(payload, access_token_expiry)
    if not refresh_token:
        refresh_token = await create_refresh_token(payload)
        
    return TokenResponse(
        access_token=str(access_token),
        refresh_token=str(refresh_token),
        expires_in=access_token_expiry.seconds  # in seconds
    )
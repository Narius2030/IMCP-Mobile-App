import sys
sys.path.append('./')

from fastapi import APIRouter, status, Depends, Header
from fastapi.security import OAuth2PasswordRequestForm
from src.auth.services import get_token, get_refresh_token
import logging


# router
auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
    responses={404: {"description": "Not found"}},
)


########## Endpoints ##########
logger = logging.getLogger("uvicorn")


@auth_router.post("/token", status_code=status.HTTP_201_CREATED)
async def authenticate_user(data: OAuth2PasswordRequestForm = Depends()):
    data = await get_token(data)
    logger.info(f"Authenticated data: {data}")
    return data


@auth_router.post("/refresh", status_code=status.HTTP_201_CREATED)
async def refresh_access_token(refresh_token: str = Header()):
    return await get_refresh_token(refresh_token)
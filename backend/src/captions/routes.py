import sys
sys.path.append('./')

from core.database import Database
from fastapi import status, APIRouter, Depends
from fastapi.responses import JSONResponse
from src.captions.services import getCaptionTokens, getOnlyTokens, getOnlyTexts
from core.config import get_settings
from motor import motor_asyncio
from core.security import oauth2_scheme
import logging


settings = get_settings()

# router
caption_router = APIRouter(
    prefix="/api/v1/captions",
    tags=["Captions"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme)]
)


########## Endpoints ##########
logger = logging.getLogger("uvicorn")


# Get text + token for caption and short_caption
@caption_router.get("/text-tokens/{num_rows}", status_code=status.HTTP_200_OK)
async def get_caption_tokens(num_rows):
    data = await getCaptionTokens(num_rows)
    payload = {
        "data": data
    }
    logger.info(f'========== Fetched successfully texts-tokens: {len(data)} ==========')
    return JSONResponse(content=payload)


# Get only token for caption and short_caption
@caption_router.get("/tokens/{num_rows}", status_code=status.HTTP_200_OK)
async def get_tokens(num_rows):
    data = await getOnlyTokens(num_rows)
    payload = {
        "data": data
    }
    logger.info(f'========== Fetched successfully tokens: {len(data)} ==========')
    return JSONResponse(content=payload)


# Get only text for caption and short_caption
@caption_router.get("/texts/{num_rows}", status_code=status.HTTP_200_OK)
async def get_texts(num_rows):
    data = await getOnlyTexts(num_rows)
    payload = {
        "data": data
    }
    logger.info(f'========== Fetched successfully caption texts: {len(data)} ==========')
    return JSONResponse(content=payload)
import sys
sys.path.append('./')

from fastapi import status, APIRouter, Depends
from services import getCaptionTokens, getOnlyTokens, getOnlyTexts
from core.security import oauth2_scheme
import logging


# router
metadata_router = APIRouter(
    prefix="/api/v1/metadata",
    tags=["Metadata"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(oauth2_scheme)]
)


########## Endpoints ##########
logger = logging.getLogger("uvicorn")


# Get text + token for caption and short_caption
@metadata_router.get("/full", status_code=status.HTTP_200_OK)
async def get_caption_tokens(page:int, per_page:int):
    data = await getCaptionTokens(page, per_page)
    logger.info(f'========== Fetched successfully texts-tokens: {len(data)} ==========')
    return data


# Get only token for caption and short_caption
@metadata_router.get("/onlytok", status_code=status.HTTP_200_OK)
async def get_tokens(page:int, per_page:int):
    data = await getOnlyTokens(page, per_page)
    logger.info(f'========== Fetched successfully tokens: {len(data)} ==========')
    return data


# Get only text for caption and short_caption
@metadata_router.get("/onlycap", status_code=status.HTTP_200_OK)
async def get_texts(page:int, per_page:int):
    data = await getOnlyTexts(page, per_page)
    logger.info(f'========== Fetched successfully caption texts: {len(data)} ==========')
    return data


# @metadata_router.get("/mobile/text-tokens", status_code=status.HTTP_200_OK)
# async def get_caption_tokens_mobile(page:int, per_page:int):
#     data = await getLatestCaption(page, per_page)
#     logger.info(f'========== Fetched successfully texts-tokens: {len(data)} ==========')
#     return data
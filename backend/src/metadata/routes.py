import sys
sys.path.append('./')

from fastapi import status, APIRouter, Body
from services import getMetadata, getEncodedFiles, getLatestEncodedFiles
from typing import Optional
import logging


# router
caption_router = APIRouter(
    prefix="/api/v1/metadata",
    tags=["Metadata"],
    responses={404: {"description": "Not found"}}
)


########## Endpoints ##########
logger = logging.getLogger("uvicorn")


# Get text + token for caption and short_caption
@caption_router.post("/metadata", status_code=status.HTTP_200_OK)
async def get_caption_tokens(
    latest_time:Optional[str] = Body(default="1970-01-01T00:00:00.000+00:00", 
                                     examples=["1970-01-01T00:00:00.000+00:00"])
):
    data = await getMetadata(latest_time)
    logger.info(f'========== Fetched successfully texts-tokens: {len(data)} ==========')
    return data

@caption_router.get("/encoded-data", status_code=status.HTTP_200_OK)
async def get_encoded_files():
    data = await getLatestEncodedFiles()
    logger.info(f'========== Fetched successfully: {data["object_keys"]} ==========')
    return data

@caption_router.post("/encoded-data", status_code=status.HTTP_200_OK)
async def get_encoded_files(partitions:list[str]=Body(...)):
    data = await getEncodedFiles(partitions)
    logger.info(f'========== Fetched successfully: {data["object_keys"]} ==========')
    return data
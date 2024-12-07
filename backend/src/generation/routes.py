import sys
sys.path.append('./')

from fastapi import status, APIRouter
from fastapi.responses import JSONResponse
from src.generation.models import Images, InsertUserData
from src.generation.services import imcpVGG16LM, imcpDarkNetLM, imcpDarkNetVG2, ingestUserData


generation_router = APIRouter(
    prefix='/api/v1/generation',
    tags=['Generator'],
    responses={404: {"description":"Not Found"}}
)

@generation_router.post('/vgg16lm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16lm(img: Images):
    data = await imcpVGG16LM(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/darknetlm', status_code=status.HTTP_201_CREATED)
async def imcp_darknetlm(img: Images):
    data = await imcpDarkNetLM(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/darknetvg2', status_code=status.HTTP_201_CREATED)
async def imcp_darknetvg2(img: Images):
    data = await imcpDarkNetVG2(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/ingest-user-data', status_code=status.HTTP_201_CREATED)
async def ingest_user_data(user_data: InsertUserData):
    data = await ingestUserData(user_data)
    payload = {
        "message": "SUCCESS",
        "user_data": data
    }
    return JSONResponse(content=payload)
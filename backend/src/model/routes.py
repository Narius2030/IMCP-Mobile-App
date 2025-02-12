import sys
sys.path.append('./')

from fastapi import status, APIRouter
from fastapi.responses import JSONResponse
from models import Images, InsertUserData
from services import imcpVGG16LM, imcpDarkNetLM, imcpDarkNetVG2, ingestUserData


model_router = APIRouter(
    prefix='/api/v1/model',
    tags=['Model'],
    responses={404: {"description":"Not Found"}}
)

@model_router.post('/vgg16lm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16lm(img: Images):
    data = await imcpVGG16LM(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@model_router.post('/darknetlm', status_code=status.HTTP_201_CREATED)
async def imcp_darknetlm(img: Images):
    data = await imcpDarkNetLM(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@model_router.post('/darknetvg2', status_code=status.HTTP_201_CREATED)
async def imcp_darknetvg2(img: Images):
    data = await imcpDarkNetVG2(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@model_router.post('/ingest-user-data', status_code=status.HTTP_201_CREATED)
async def ingest_user_data(user_data: InsertUserData):
    data = await ingestUserData(user_data)
    payload = {
        "message": "SUCCESS",
        "user_data": data
    }
    return JSONResponse(content=payload)
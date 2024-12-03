import sys
sys.path.append('./')

from fastapi import status, APIRouter
from fastapi.responses import JSONResponse
from src.generation.models import Images, InsertUserData
from src.generation.services import imcpVGG16, imcpYoLoBert, imcpYoLoGPT, ingestUserData


generation_router = APIRouter(
    prefix='/api/v1/generation',
    tags=['Generator'],
    responses={404: {"description":"Not Found"}}
)

@generation_router.get('/vgg16-lstm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16(img: Images):
    data = await imcpVGG16(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.get('/yolo8-bert-lstm', status_code=status.HTTP_201_CREATED)
async def imcp_yolo_bert(img: Images):
    data = await imcpYoLoBert(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.get('/yolo8-gpt', status_code=status.HTTP_201_CREATED)
async def imcp_yolo_gpt(img: Images):
    data = await imcpYoLoGPT(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/ingest-user-data', status_code=status.HTTP_201_CREATED)
async def imcp_yolo_gpt(user_data: InsertUserData):
    data = await ingestUserData(user_data)
    payload = {
        "message": "SUCCESS",
        "user_data": data
    }
    return JSONResponse(content=payload)
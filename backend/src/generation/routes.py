import sys
sys.path.append('./')

from fastapi import status, APIRouter, Depends
from fastapi.responses import JSONResponse
from src.generation.models import Image
from src.generation.services import imcpVGG16, imcpYoLoBert, imcpYoLo


generation_router = APIRouter(
    prefix='/api/v1/generation',
    tags=['Generator'],
    responses={404: {"description":"Not Found"}}
)

@generation_router.post('/vgg16-lstm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16(img: Image):
    data = await imcpVGG16(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/yolo8-bert-lstm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16(img: Image):
    data = await imcpYoLoBert(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@generation_router.post('/yolo8-nobert-lstm', status_code=status.HTTP_201_CREATED)
async def imcp_vgg16(img: Image):
    data = await imcpYoLo(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)
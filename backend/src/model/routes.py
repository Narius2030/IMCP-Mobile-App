import sys
sys.path.append('./')

from fastapi import status, APIRouter
from fastapi.responses import JSONResponse
from src.model.models import Images
from src.model.services import callModel, ingestDataToKafka


model_router = APIRouter(
    prefix='/api/v1/model',
    tags=['Model'],
    responses={404: {"description":"Not Found"}}
)

@model_router.post('/predict-caption', status_code=status.HTTP_201_CREATED)
async def call_model(img: Images):
    data = await callModel(img)
    payload = {
        "message": "SUCCESS",
        "predicted_caption": data
    }
    return JSONResponse(content=payload)

@model_router.post('/ingest-user-data', status_code=status.HTTP_201_CREATED)
async def ingest_user_data(img: Images):
    data = await ingestDataToKafka(img)
    payload = {
        "status": "SUCCESS",
        "message": data
    }
    return JSONResponse(content=payload)
import sys
sys.path.append('./')

import pymongo
import pickle
import torch
from datetime import timezone
from fastapi.exceptions import HTTPException
from core.config import get_settings
from utils.operators.trinodb import SQLOperators
from utils.operators.storage import MinioStorageOperator

## Global config variabels
settings = get_settings()
sql_opt = SQLOperators("imcp", settings)
minio_opt = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', access_key=settings.MINIO_USER, secret_key=settings.MINIO_PASSWD)


###
### TODO: Get "text + token" for caption and short_caption
###
async def getMetadata(latest_time:str):
    tokens = []
    response = {}
    try:
        columns = ['original_url', 's3_url', 'short_caption', 'tokenized_caption']
        for batch in sql_opt.data_generator('refined', columns, latest_time):
            tokens += batch
        response = {
            "status": "success",
            "total": len(tokens),
            "data": tokens
        }
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return response


###
### TODO: Get encoded data of image and caption
###
async def getLatestEncodedFiles():
    files = []
    try:
        latest_time = sql_opt.get_latest_fetching_time('gold', 'encoded_data')
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)
        # get list of object
        objects = minio_opt.get_list_objects("lakehouse")
        for obj in objects:
            if obj.last_modified > latest_time:
                files.append(obj.object_name)
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return {
        "bucket": "lakehouse",
        "total": len(files),
        "object_keys": files
    }
    
    
###
### TODO: Get encoded data of image and caption
###
async def getEncodedFiles(partitions:list[str]):
    files = []
    try:
        for partition in partitions:
            objects = minio_opt.get_list_objects("lakehouse", partition)
            for obj in objects:
                files.append(obj.object_name)
    except HTTPException:
        raise HTTPException(status_code=505, detail="Fetching is failed - watch in internal")
    return {
        "bucket": "lakehouse",
        "total": len(files),
        "object_keys": files
    }
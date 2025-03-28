import sys
sys.path.append('./')

import numpy as np
import asyncio
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi.exceptions import HTTPException
from core.config import get_settings
from utils.kafka_client import Prod
from src.model.models import Images
from utils.operators.image import ImageOperator
from utils.operators.storage import MinioStorageOperator
from utils.operators.modelflow import MlflowOperator


settings = get_settings()
mlflow_operator = MlflowOperator(endpoint="http://160.191.244.13:7893")
model_path = mlflow_operator.get_model_path("Image_Captioning")
img_opt = ImageOperator('mlflow', model_path)
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', 
                                      access_key=settings.MINIO_USER, 
                                      secret_key=settings.MINIO_PASSWD)


async def ingestDataToKafka(image: Images):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        value = {
            'image_name': f"image_{timestamp}.jpg",
            'image_base64': str(image.image_pixels),
            'image_size': f"{image.shape[0]}x{image.shape[1]}"
        }
        message = {"key": timestamp, "value": value}
        producer = Prod(settings.KAFKA_ADDRESS, 'mobile-images', message)
        producer.run()
        return "Ingest data to kafka successfully"
    
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in ingesting user data to storage --- \n {str(ex)}")
    

async def callModel(image: Images):
    # Create an image from the NumPy array
    image_bytes = base64.b64decode(image.image_pixels)
    image_pil = Image.open(BytesIO(image_bytes)).convert('RGB')
    image_rgb = np.array(image_pil)
    
    # check image array
    if image_rgb is None:
        raise HTTPException(status_code=404, detail="Image array is null!")
    
    try:
        caption = img_opt.predict_caption(image_bytes)
        # for word in img_opt.predict_caption(image_bytes):
        #     yield word + " "
        #     await asyncio.sleep(0.5)
        return caption
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in process image to create caption!\n {str(ex)}")
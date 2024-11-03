import sys
sys.path.append('./')

import numpy as np
import base64
import cv2
import io
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi.exceptions import HTTPException
from core.config import get_settings
from src.generation.models import Image
from PIL import Image
from utils.generator import VGG16Generator, YOLOGenerator
from utils.storage import MinioStorageOperator
from utils.database import MongoDBOperator



## Global config variabels
settings = get_settings()
vgg_operator = None #VGG16Generator('full', 'mlflow','/models/vgg16_lstm/img_caption_model.h5')
yolo_bert_opt = YOLOGenerator('mlflow','/models/yolo_lstm/yolo_bert_lstm_8ep_cp-0001.weights.h5', 'bert')
yolo_nobert_opt = YOLOGenerator('mlflow','/models/yolo_lstm/yolo_lstm_30ep_cp-0005.weights.h5')
minio_operator = MinioStorageOperator(endpoint='116.118.50.253:9000', access_key='minio', secret_key='minio123')
mongo_operator = MongoDBOperator('imcp', settings.DATABASE_URL)



async def upload_image(image_bytes):
    # Create image name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_name = f"image_{timestamp}.jpg"
    # Ensure color for image
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image_rgb = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Upload to Object Store
    _, encoded_image = cv2.imencode('.jpg', image_rgb)
    minio_image_bytes = io.BytesIO(encoded_image)
    minio_operator.upload_object_bytes(minio_image_bytes, 'mlflow', f'/user_images/{image_name}', "image/jpeg")
    return image_name
    
    
async def insert_user_data(image_rgb, image_name, predicted_caption):
    # Insert to MongoDB
    data = {}
    datasets = []
    data['url'] = f"""{settings.MINIO_URL}/mlflow/user_images/{image_name}"""
    data['image_shape'] = image_rgb.shape
    data['predicted_caption'] = str(predicted_caption)
    data['manualcaption'] = ""
    data['created_time'] = datetime.now()
    datasets.append(data)
    mongo_operator.insert_batches('user_data', datasets)
    


async def imcpVGG16(image: Image):
    # Create an image from the NumPy array
    image_bytes = base64.b64decode(image.image_pixels)
    image_pil = Image.open(BytesIO(image_bytes)).resize((224,224))
    image_rgb = np.array(image_pil)
    
    # check image array
    if image_rgb is None:
        raise HTTPException(status_code=404, detail="Image array is null!")
    
    try:
        caption = vgg_operator.generate_caption(image_rgb)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in process image to create caption!\n {str(ex)}")
    
    return caption


async def imcpYoLoBert(image: Image):
    # Create an image from the NumPy array
    image_bytes = base64.b64decode(image.image_pixels)
    image_pil = Image.open(BytesIO(image_bytes))
    image_rgb = np.array(image_pil)
    
    # check image array
    if image_rgb is None:
        raise HTTPException(status_code=404, detail="Image array is null!")
    
    try:
        image_name = await upload_image(image_bytes)
        caption = yolo_bert_opt.generate_caption(image_rgb)
        await insert_user_data(image_rgb, image_name, caption)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in process image to create caption!\n {str(ex)}")
    
    return caption


async def imcpYoLo(image: Image):
    # Create an image from the NumPy array
    image_bytes = base64.b64decode(image.image_pixels)
    image_pil = Image.open(BytesIO(image_bytes))
    image_rgb = np.array(image_pil)
    
    # check image array
    if image_rgb is None:
        raise HTTPException(status_code=404, detail="Image array is null!")
    
    try:
        image_name = await upload_image(image_bytes)
        caption = yolo_nobert_opt.generate_caption(image_rgb)
        await insert_user_data(image_rgb, image_name, caption)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in process image to create caption!\n {str(ex)}")
    
    return caption
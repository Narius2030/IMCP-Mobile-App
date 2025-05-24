import sys
sys.path.append('./')

import numpy as np
import re
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi.exceptions import HTTPException
from core.config import get_settings
from utils.kafka_client import Prod
from src.model.models import Images
from utils.operators.image import ImageOperator
from utils.operators.modelflow import MlflowOperator


settings = get_settings()
mlflow_operator = MlflowOperator(endpoint="http://36.50.135.226:7893")
model_path = mlflow_operator.get_latest_model_path("BartPho_ViT_GPT2_LoRA_ICG", "Production")
img_opt = ImageOperator('mlflow', model_path)


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
        
        normalized_caption = re.split(r'[ _]', caption)
        # concatenate '.' with forward word
        final_caption_words = list()
        for idx, word in enumerate(normalized_caption):
            if word == '.' and final_caption_words:
                final_caption_words[-1] = normalized_caption[idx-1] + word
            else:
                final_caption_words.append(word)
        # unify to a string
        final_caption = ' '.join(final_caption_words)
        print(f"CAPTION: {type(final_caption)} - {final_caption}")
        return final_caption
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error in process image to create caption!\n {str(ex)}")
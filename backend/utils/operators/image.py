import cv2
import torch
import requests
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer
from utils.operators.load_models import ModelLoaders


class ImageOperator:
    def __init__(self, bucket_name, model_path) -> None:
        try:
            self.bucket_name = bucket_name
            self.model_path = model_path
            self.model = ModelLoaders.load_gptmodel_from_configs(self.bucket_name, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
            self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            raise ImportError(f"Load model failed ---> {e}")
    
    
    def load_image(self, image_bytes):
        try:
            # Chuyển nội dung về mảng numpy
            image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            # Đọc ảnh bằng OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            # Kiểm tra nếu ảnh không hợp lệ
            if image is None:
                print("Error: OpenCV could not decode the image.")
                return None
            # Chuyển đổi từ BGR (OpenCV) sang RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("Image shape: ", image_rgb.shape)
            return image_rgb

        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None


    def predict_caption(self, image_bytes):
        # Load tokenizer của BartPho
        pixel_values =  self.feature_extractor(self.load_image(image_bytes), return_tensors="pt")["pixel_values"]
        print("Featured values shape: ", pixel_values.shape)
        output_ids = self.model.generate(
                    pixel_values
                    ,max_length=250
                    ,min_length=100
                    ,num_beams = 5
                    ,early_stopping=True
                    ,pad_token_id=self.tokenizer.eos_token_id
                )
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        yield caption
    
    # def predict_caption(self, image_bytes):
    #     # ... existing code ...
    #     pixel_values = self.feature_extractor(self.load_image(image_bytes), return_tensors="pt")["pixel_values"]
    #     print("Featured values shape: ", pixel_values.shape)
        
    #     # Initialize with just the beginning of sequence
    #     generated_ids = [self.tokenizer.bos_token_id]
        
    #     # Generate one token at a time
    #     for _ in range(150):  # max_length as before
    #         outputs = self.model.generate(
    #             pixel_values,
    #             max_length=len(generated_ids) + 2,  # +2 to account for BOS and new token
    #             min_length=1,
    #             num_beams=3,  # Use greedy decoding for streaming
    #             early_stopping=True,
    #             pad_token_id=self.tokenizer.eos_token_id,
    #             decoder_input_ids=torch.tensor([generated_ids]) if generated_ids else None,
    #             use_cache=True
    #         )
            
    #         next_token = outputs[0][-1]
    #         generated_ids.append(next_token.item())
            
    #         # Decode the current token
    #         current_word = self.tokenizer.decode([next_token], skip_special_tokens=True)
    #         if current_word.strip():  # Only yield non-empty strings
    #             yield current_word
            
    #         # Check if we've hit the end token
    #         if next_token == self.tokenizer.eos_token_id:
    #             break

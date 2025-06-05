import cv2
import requests
import torch
import numpy as np
from transformers import ViTImageProcessor, AutoTokenizer, CLIPModel, CLIPProcessor
from utils.operators.load_models import ModelLoaders


class ImageOperator:
    """
    A class for handling image operations including loading and caption generation.
    
    This class provides functionality to load images from bytes and generate captions
    using a pre-trained model. It uses ViT for image feature extraction and BART for
    caption generation.
    """
    
    def __init__(self, bucket_name, model_path) -> None:
        """
        Initialize the ImageOperator with model configurations.
        Args:
            bucket_name (str): Name of the bucket containing model configurations
            model_path (str): Path to the model within the bucket
        Raises:
            ImportError: If model loading fails
        """
        try:
            self.bucket_name = bucket_name
            self.model_path = model_path
            self.model = ModelLoaders.load_gptmodel_from_configs(self.bucket_name, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
            self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            raise ImportError(f"Load model failed ---> {e}")
    
    
    def load_image(self, image_bytes):
        """
        Load and preprocess an image from bytes.
        Args:
            image_bytes (bytes): Raw image data in bytes format
        Returns:
            numpy.ndarray: Preprocessed RGB image array, or None if loading fails
        Note:
            The image is converted from BGR (OpenCV default) to RGB format
        """
        try:
            image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
            # read image by opencv
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                print("Error: OpenCV could not decode the image.")
                return None
            # convert from bgr (opencv) sang rgb
            # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("Image shape: ", image.shape)
            return image

        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {e}")
            return None

    def classify_traffic_image(self, labels, image) -> dict[str, any]:
        try:
            inputs = self.clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.clip(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            prediction = labels[probs.argmax()]
            confidence = probs.max().item()
            return {
                "prediction": prediction,
                "confidence": confidence,
                "traffic_prob": probs[0][0].item(),
                "non_traffic_prob": probs[0][1].item(),
                "error": None
            }
        except Exception as e:
            return {
                "prediction": None,
                "confidence": None,
                "traffic_prob": None,
                "non_traffic_prob": None,
                "error": str(e)
            }
    
    def predict_caption(self, image_bytes, labels) -> str:
        image = self.load_image(image_bytes)
        image_type = self.classify_traffic_image(labels, image)
        print(image_type)
        try:
            if image_type['prediction'] != 'non-traffic image':
                pixel_values =  self.feature_extractor(image, return_tensors="pt")["pixel_values"]
                print("Featured values shape: ", pixel_values.shape)
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=150,  # +2 to account for bos and new token
                    min_length=50,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.8,  # cool down
                    top_k=20,
                    top_p=0.9,  # extend choices
                    no_repeat_ngram_size=3,  # prevent 5-ngram repetition
                    repetition_penalty=2.0,  # repetition fine
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    decoder_start_token_id=self.tokenizer.bos_token_id
                )
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return caption
            else: 
                return "Xin lỗi, tôi chỉ có khả năng mô tả các hình ảnh liên quan đến giao thông. Ảnh này nằm ngoài phạm vi hiểu biết của tôi về giao thông, nên không thể sinh chú thích."
        except Exception as ex:
            raise Exception(f'====> ERROR: {str(ex)}')

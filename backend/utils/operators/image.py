import cv2
import requests
import torch
from torch.nn import functional as F
import numpy as np
from transformers import ViTImageProcessor, AutoTokenizer
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

    def compute_caption_entropy(self, token_logits):
        probs = F.softmax(token_logits, dim=-1)  # Shape: (T, V)
        log_probs = F.log_softmax(token_logits, dim=-1)  # Shape: (T, V)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (T,)
        avg_entropy = entropy.mean().item()
        return avg_entropy

    def predict_caption(self, image_bytes):
        pixel_values = self.feature_extractor(self.load_image(image_bytes), return_tensors="pt")["pixel_values"]
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        generated_ids = [self.model.config.decoder_start_token_id]
        max_length = 150
        probs = []
        logits = []
        for _ in range(max_length):
            input_ids = torch.tensor([generated_ids])
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, decoder_input_ids=input_ids)

            # Lấy xác suất tại bước hiện tại
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)  # [1, vocab_size]
            probs.append(next_token_probs.squeeze().cpu().numpy())  # Lưu xác suất
            logits.append(next_token_logits)  # Lưu logits

            # Sinh token tiếp theo (greedy)
            next_token = torch.argmax(next_token_probs, dim=-1).item()
            generated_ids.append(next_token)

            # Nếu gặp token kết thúc, dừng lại
            if next_token == self.tokenizer.eos_token_id:
                break
        
        all_logits = torch.cat(logits, dim=0)
        caption_entropy = self.compute_caption_entropy(all_logits)
        print(f"Caption entropy: {caption_entropy}")
        
        if caption_entropy > 0.5:
            caption = "Xin lỗi, tôi chỉ có khả năng mô tả các hình ảnh liên quan đến giao thông. Ảnh này nằm ngoài phạm vi hiểu biết của tôi về giao thông, nên không thể sinh chú thích."
            return caption
        
        caption = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return caption

    # def predict_caption(self, image_bytes):
    #     """
    #     Generate a caption for the input image.
        
    #     Args:
    #         image_bytes (bytes): Raw image data in bytes format
            
    #     Returns:
    #         str: Generated caption for the image
            
    #     Note:
    #         The caption generation uses beam search with specific parameters for
    #         controlling length, diversity, and repetition.
    #     """
    #     # Load tokenizer của BartPho
    #     pixel_values =  self.feature_extractor(self.load_image(image_bytes), return_tensors="pt")["pixel_values"]
    #     print("Featured values shape: ", pixel_values.shape)
    #     output_ids = self.model.generate(
    #                 pixel_values,
    #                 max_length=150,  # +2 to account for BOS and new token
    #                 min_length=50,
    #                 num_beams=4,
    #                 do_sample=True,
    #                 temperature=0.8,  # Giảm nhiệt độ
    #                 top_k=20,
    #                 top_p=0.9,  # Mở rộng lựa chọn từ
    #                 no_repeat_ngram_size=3,  # Ngăn lặp cụm 5 từ
    #                 repetition_penalty=2.0,  # Phạt lặp mạnh hơn
    #                 early_stopping=True,
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #                 eos_token_id=self.tokenizer.eos_token_id,
    #                 decoder_start_token_id=self.tokenizer.bos_token_id
    #             )
    #     caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #     normalized_caption = re.split(r'[ _]', caption)
    #     # concatenate '.' with forward word
    #     final_caption_words = list()
    #     for idx, word in enumerate(normalized_caption):
    #         if word == '.' and final_caption_words:
    #             final_caption_words[-1] = normalized_caption[idx-1] + word
    #         else:
    #             final_caption_words.append(word)
    #     # unify to a string
    #     final_caption = ' '.join(final_caption_words)
    #     logging.info(f"CAPTION: {type(final_caption)} - {final_caption}")
        
    #     return final_caption

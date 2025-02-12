import sys
sys.path.append('./')

import pickle
import numpy as np
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing
sys.modules['keras.src.preprocessing'] = preprocessing
from utils.operators.storage import MinioStorageOperator
from utils.generator.load_models import ModelLoaders
from utils.extractor import VGG16FeatureExtractorModel
from core.config import get_settings

settings = get_settings()
loader = ModelLoaders()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', access_key=settings.MINIO_USER, secret_key=settings.MINIO_PASSWD)


class VGG16Generator(VGG16FeatureExtractorModel):
    def __init__(self, bucket_name, file_path) -> None:
        super().__init__()
        try:
            self.vgg_model = loader.load_h5model_from_h5(bucket_name, file_path)
        except Exception as ex:
            raise ImportError(f'load model failed - {str(ex)}')
        
        with open(settings.LSTM_TOKENIZERS, 'rb') as f:
            self.tokenizer = pickle.load(f)
    
    def idx_to_word(self, integer, tokenizer):
        """
        Converts a numerical token ID back to its corresponding word using a tokenizer.

        Args:
            integer: The integer ID representing the word.
            tokenizer: The tokenizer object that was used to tokenize the text.

        Returns:
            The word corresponding to the integer ID, or None if the ID is not found.
        """
        # Iterate through the tokenizer's vocabulary
        for word, index in tokenizer.word_index.items():
            # If the integer ID matches the index of a word, return the word
            if index == integer:
                return word
        # If no matching word is found, return None
        return None
    
    def predict_caption(self, image_feature, max_length):
        """
        Generates a caption for an image using a trained image captioning model.

        Args:
            model: The trained image captioning model.
            image: The image to generate a caption for.
            tokenizer: The tokenizer used to convert text to numerical sequences.
            max_length: The maximum length of the generated caption.

        Returns:
            The generated caption as a string.
        """
        in_text = 'startseq'
    
        for _ in range(max_length):
            # Tokenize
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
            
            # Predict
            yhat = self.vgg_model.predict([image_feature, sequence], verbose=0)
            # Get word with highest probability, but add randomness
            yhat = yhat[0]
            # Lấy top 3 words có xác suất cao nhất
            top_indices = np.argsort(yhat)[-3:]
            # Chọn ngẫu nhiên 1 trong 3 words
            yhat = np.random.choice(top_indices, p=yhat[top_indices]/np.sum(yhat[top_indices]))
            # Convert prediction to word
            word = self.idx_to_word(yhat, self.tokenizer)
            
            if word is None:
                break
                
            in_text += ' ' + word
            
            if word == 'endseq':
                break

        caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        return caption
    
    
    def generate_caption(self, image_rgb):
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError("Input image must be a NumPy array.")

        try:
            # Extract features from the image using VGG16
            feature_matrix = self.forward(image_rgb)
            caption = self.predict_caption(feature_matrix, 35)
        except Exception as ex:
            caption = None
            raise ValueError(str(ex))
        
        return caption
import pickle
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16, preprocess_input
from utils.storage import MinioStorageOperator
from core.config import get_settings

settings = get_settings()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', access_key=settings.MINIO_USER, secret_key=settings.MINIO_PASSWD)


class VGG16Generator():
    def __init__(self, mode:str, bucket:str, file_path:str) -> None:
        self.mode = mode
        try:
            self.vgg_extractor = VGG16()
            self.vgg_extractor = keras.models.Model(inputs=self.vgg_extractor.inputs, outputs=self.vgg_extractor.layers[-2].output)
            if mode == 'lite':
                self.interpreter = tf.lite.Interpreter(model_content=minio_operator.load_object_bytes('mlflow','/models/vgg16_lstm/img_caption_model.tflite'))
                self.interpreter.allocate_tensors()
            elif mode == 'full':
                self.vgg_model = minio_operator.load_model_from_minio('model', bucket, file_path)
            else:
                raise NameError("Model mode is not valid. There are two mode which is 'lite' and 'full'")
        except Exception as ex:
            raise ImportError(f'load model failed - {str(ex)}')
        
        with open('./utils/vgg16-lstm/tokenizer.pkl', 'rb') as f:
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
    
    def predict_caption(self, image_feature, tokenizer, max_length):
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
        # Add start tag for generation process
        in_text = 'startseq'
        
        # Iterate over the max length of sequence
        for _ in range(max_length):
            # Tokenize the current caption into a sequence of integers
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # Pad the sequence
            sequence = pad_sequences([sequence], maxlen=max_length)
            # Predict next word
            yhat = self.vgg_model.predict([image_feature, sequence], verbose=0)
            # Get the index of the word with the highest probability
            yhat = np.argmax(yhat)
            # Convert index to word
            word = self.idx_to_word(yhat, tokenizer)
            # Stop if word not found
            if word is None:
                break
            # Append word as input for generating next word
            in_text += " " + word
            # Stop if we reach the end tag
            if word == 'endseq':
                break
        return in_text
        
    def predict_caption_tflite(self, image_feature, tokenizer, max_length):
        """
        Generates a caption for an image using a TFLite model.

        Args:
            image_feature: The feature vector of the image.
            tokenizer: The tokenizer used to convert text to numerical sequences.
            max_length: The maximum length of the generated caption.

        Returns:
            The generated caption as a string.
        """
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Bắt đầu với chuỗi khởi đầu
        in_text = 'startseq'
        # Duyệt tối đa max_length từ
        for _ in range(max_length):
            # Chuyển đổi chuỗi hiện tại thành dãy số
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # Điền padding nếu cần
            sequence = pad_sequences([sequence], maxlen=max_length).astype(np.float32)

            # Đặt tensor đầu vào cho interpreter
            self.interpreter.set_tensor(input_details[0]['index'], sequence)
            self.interpreter.set_tensor(input_details[1]['index'], image_feature.astype(np.float32))

            # Chạy mô hình để dự đoán
            self.interpreter.invoke()

            # Lấy kết quả dự đoán
            yhat = self.interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Lấy từ có xác suất cao nhất
            predicted_id = np.argmax(yhat)
            word = self.idx_to_word(predicted_id, tokenizer)

            # Dừng nếu không tìm thấy từ hoặc gặp 'endseq'
            if word is None or word == 'endseq':
                break

            # Thêm từ vào chuỗi hiện tại
            in_text += ' ' + word

        # Loại bỏ các tag không cần thiết
        final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        return final_caption
    
    def generate_caption(self, image_array):
        if not isinstance(image_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array.")
        
        # Đảm bảo shape của ảnh là (1, height, width, channels)
        if image_array.ndim == 3: 
            image = image_array #np.resize(image_array, (224,224,3))
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            print(image.shape)

        try:
            # Extract features from the image using VGG16
            feature = self.vgg_extractor.predict(image, verbose=0)
            if self.mode == 'lite':
                caption = self.predict_caption_tflite(feature, self.tokenizer, 35)
            elif self.mode == 'full':
                caption = self.predict_caption(feature, self.tokenizer, 35)
        except Exception as ex:
            caption = None
            raise ValueError(str(ex))
        
        return caption
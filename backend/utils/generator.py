import pickle
import numpy as np
import keras
import tensorflow as tf
import torch
import io
from ultralytics import YOLO
from torchvision import transforms
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Flatten
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from utils.storage import MinioStorageOperator
from core.config import get_settings



settings = get_settings()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}',
                                      access_key=settings.MINIO_USER,
                                      secret_key=settings.MINIO_PASSWD)

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
    

class YOLOGenerator():
    def __init__(self, bucket_name, file_path, model_name=None) -> None:
        self.model_name = model_name
        try:
            if self.model_name == 'bert':
                model = self.create_model(max_length=315, vocab_size=30522)
                self.tokenizer = BertTokenizer.from_pretrained('./utils/yolo8-bert-lstm/bert-tokenizer-files')
            else:
                model = self.create_model(max_length=315, vocab_size=12963)
                with open('./utils/yolo8-bert-lstm/yolo_lstm_tokenizer.pkl', 'rb') as f:
                    self.tokenizer = pickle.load(f)
            self.yolo8_model = minio_operator.load_model_from_minio('weight', bucket_name, file_path, base_model=model)
        except Exception as ex:
            raise ImportError(f"load model failed!\n {str(ex)}")

    def get_backbone(self, model_name:str='yolov8n.pt'):
        model = YOLO(model_name)
        # Access the backbone layers
        backbone = model.model.model[:10]  # Layers 0 to 9 form the backbone
        # Create a new Sequential model with just the backbone layers
        backbone_model = torch.nn.Sequential(*backbone)
        return backbone_model

    def extract_features(self, image):
        backbone_model = self.get_backbone(model_name='./utils/pre-trained/yolov8n.pt')
        with torch.no_grad():
            features = backbone_model(image)
        return features

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),  # Chuyển từ numpy sang PIL
            transforms.Resize((640, 640)),  # Thay đổi kích thước ảnh về 640x640
            transforms.ToTensor(),  # Chuyển thành tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo chuẩn của ImageNet
        ])
        tensor_image = preprocess(image).unsqueeze(0)  # Thêm batch dimension
        return tensor_image
    
    def create_model(self, max_length, vocab_size):
        # image feature layers
        inputs1 = Input(shape=(256, 20, 20))  # Đầu vào có kích thước (256, 20, 20)
        fe1 = Flatten()(inputs1)  # Làm phẳng kích thước thành (None, 102400)
        fe1 = Dropout(0.4)(fe1)
        fe2 = Dense(256, activation='relu')(fe1)

        # sequence feature layers
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se2)

        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    def idx_to_word_nobert(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def idx_to_word_bert(self, tensor_ids, tokenizer):
        # Kiểm tra xem tensor_ids có phải là một tensor hay không
        if isinstance(tensor_ids, torch.Tensor):
            # Chuyển tensor thành list để xử lý từng phần tử
            tensor_ids = tensor_ids.tolist()

        # Kiểm tra nếu tensor_ids chỉ là một giá trị (như numpy.int64) và không thể lặp
        if isinstance(tensor_ids, (int, np.integer)):
            tensor_ids = [tensor_ids]  # Chuyển giá trị đơn thành danh sách

        words = []
        for integer in tensor_ids:
            if integer == 0:  # Bỏ qua nếu giá trị bằng 0 (giả sử 0 là token padding)
                continue
            # Decode từng token ID thành từ
            words.append(tokenizer.decode([integer], skip_special_tokens=True))
        
        return " ".join(words)
    
    
    # Hàm kiểm tra lặp chuỗi con
    def has_repeated_substring(self, seq, min_length=3):
        words = seq.split()
        for i in range(len(words) - min_length + 1):
            substring = ' '.join(words[i:i + min_length])
            remaining_seq = ' '.join(words[i + min_length:])
            if substring in remaining_seq:
                return True
        return False


    def predict_caption_beam_search(self, image_feature, max_length=315, beam_width=3):
        # Khởi tạo danh sách các câu, mỗi câu là tuple (câu, điểm số)
        sequences = [('', 0.0)]  # Chuỗi ban đầu trống với điểm số bằng 0
        # Từ điển đếm số lần xuất hiện của từ
        word_count = {}
        # Vòng lặp sinh từ mới trong tối đa max_length bước
        for _ in range(max_length):
            all_candidates = []  # Danh sách các câu khả dĩ mới tại bước này
            # Duyệt qua từng câu trong danh sách beam width hiện tại
            for seq, score in sequences:
                # Kiểm tra nếu câu đã đạt đến giới hạn số từ hoặc chứa token kết thúc, thêm vào danh sách và bỏ qua việc mở rộng
                if len(seq.split()) >= max_length or '[EOS]' in seq:
                    all_candidates.append((seq, score))
                    continue
                # Encode câu hiện tại bằng BERT tokenizer
                encoding = self.tokenizer.encode(seq, add_special_tokens=False)
                # Pad encoding
                encoding = pad_sequences([encoding], maxlen=max_length, padding='post')
                # Convert encoding to int32
                encoding = tf.cast(encoding, dtype=tf.int32)
                # Dự đoán xác suất của từ tiếp theo
                yhat = self.yolo8_model.predict([image_feature, encoding], verbose=0)
                # Lấy top từ có xác suất cao nhất (beam_width từ)
                top_indices = np.argsort(yhat[0])[-beam_width:]
                # Loại bỏ các từ đã xuất hiện nhiều hơn 2 lần trong câu
                top_indices = [idx for idx in top_indices if word_count.get(self.idx_to_word_bert(idx, self.tokenizer), 0) < 5]

                # Tạo các câu mới và tính toán điểm số cho từng từ được thêm
                for idx in top_indices:
                    word = self.idx_to_word_bert(idx, self.tokenizer)
                    if word is None:
                        continue
                    if word.startswith('##'):  # Xử lý subword
                        word = word[2:]
                        new_seq = seq.rstrip() + word  # Nối vào từ trước
                    else:
                        new_seq = seq + ' ' + word
                    # Tăng số lần xuất hiện của từ trong từ điển đếm
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
                    # # Kiểm tra nếu chuỗi có chuỗi con lặp lại
                    # if self.has_repeated_substring(new_seq):
                    #     continue  # Bỏ qua chuỗi này nếu có lặp chuỗi con
                    
                    # Penalization: Giảm điểm nếu từ đã xuất hiện nhiều lần
                    repetition_penalty = 1.0 / (word_count[word] ** 0.15)
                    # Normalization by length: chia log xác suất cho chiều dài câu
                    new_score = score + (np.log(yhat[0][idx]) * repetition_penalty) / (len(new_seq.split()) + 1)
                    all_candidates.append((new_seq, new_score))

            # Nếu không có ứng cử viên nào được tạo ra, dừng vòng lặp
            if not all_candidates:
                break
            # Sắp xếp tất cả các câu khả dĩ theo điểm số và chỉ giữ lại beam_width câu tốt nhất
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]

        # Kiểm tra nếu sequences trống
        if not sequences:
            return ""  # Trả về chuỗi rỗng nếu không có câu nào được tạo ra
        # Trả về câu có điểm số cao nhất trong danh sách beam width
        best_seq = sequences[0][0]
        return best_seq.strip()  # Loại bỏ khoảng trắng thừa
    
    
    def predict_caption(self, image_feature, max_length):
        """
        Generates a caption for an image using a trained image captioning model.

        Args:
            model: The trained image captioning model.
            image_feature: The image to generate a caption for.
            tokenizer: The tokenizer used to convert text to numerical sequences.
            max_length: The maximum length of the generated caption.

        Returns:
            The generated caption as a string.
        """
        in_text = 'startseq'  # Bắt đầu bằng chuỗi trống, không cần [CLS]
        for _ in range(max_length):
            # encode the text using tokenizer
            if self.model_name == 'bert':
                encoding = self.tokenizer.encode(in_text, add_special_tokens=False)  # Không thêm special tokens
            else:
                encoding = self.tokenizer.texts_to_sequences([in_text])[0]
            # pad encoding
            encoding = pad_sequences([encoding], maxlen=max_length)
            # Convert encoding to int32
            encoding = tf.cast(encoding, dtype=tf.int32)
            
            # predict next word
            yhat = self.yolo8_model.predict([image_feature, encoding], verbose=0)
            # get index with highest probability
            yhat = np.argmax(yhat)
            # map integer to word
            if self.model_name == 'bert':
                word = self.idx_to_word_bert(yhat, self.tokenizer)
            else:
                word = self.idx_to_word_nobert(yhat, self.tokenizer)

            if word is None:
                break  # Dừng lại nếu không tìm thấy từ hợp lệ
            # Avoid adding special tokens like [SEP]
            in_text += ' ' + word

            if word == 'ends' or word == 'endseq':  # Dừng lại khi gặp token kết thúc [SEP]
                break

        return in_text.strip()


    def generate_caption(self, image_rgb):
        if not isinstance(image_rgb, np.ndarray):
            raise ValueError("Input image must be a NumPy array.")
        
        # Đảm bảo shape của ảnh là (1, height, width, channels)
        if image_rgb.ndim == 3:
            transformed_image = self.preprocess_image(image_rgb)
            feature_matrix = self.extract_features(transformed_image)
            print('HERE', feature_matrix.shape)

        try:
            caption = self.predict_caption(feature_matrix, 315)
        except Exception as ex:
            caption = None
            raise ValueError(str(ex))
        return caption
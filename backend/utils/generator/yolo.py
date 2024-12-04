import numpy as np
import tensorflow as tf
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Flatten #type: ignore
from tensorflow.keras.models import Model #type: ignore
from utils.storage import MinioStorageOperator
from utils.extractor import YOLOFeatureExtractorModel
from utils.generator.load_models import ModelLoaders
from core.config import get_settings

settings = get_settings()
loader = ModelLoaders()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', access_key=settings.MINIO_USER, secret_key=settings.MINIO_PASSWD)


class YOLOGptGenerator(YOLOFeatureExtractorModel):
    def __init__(self, bucket_name, file_path, backbone_path) -> None:
        super().__init__(backbone_path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.model = loader.load_gptmodel_from_configs(bucket_name, file_path)
        except Exception as ex:
            raise ImportError(f"load model failed!\n {str(ex)}")
    
    def feature_extractor(self, image_rgb):
        features = self.forward(image_rgb)
        # Kiểm tra và điều chỉnh kích thước đầu ra của features nếu cần
        if features.shape[1] != 3 or features.shape[2:] != (224, 224):
            features = torch.nn.functional.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
            features = features[:, :3, :, :]  # Chỉ giữ 3 kênh đầu
        return features
    
    def predict_internet_caption(self, image_rgb):
        # Sử dụng FeatureExtractorModel để trích xuất đặc trưng
        pixel_values = self.feature_extractor(image_rgb)
        output_ids = self.model.generate(
                    pixel_values
                    ,max_length=150 
                    ,min_length=10
                    ,temperature=0.8
                    ,repetition_penalty=1.2 
                    ,early_stopping=True
                )
        
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption



class YOLOBertGenerator(YOLOFeatureExtractorModel):
    def __init__(self, bucket_name, file_path, model_path) -> None:
        super().__init__(model_path)
        try:
            model = self.create_model(max_length=315, vocab_size=30522)
            self.tokenizer = BertTokenizer.from_pretrained(settings.BERT_TOKENIZERS)
            self.yolo8_model = loader.load_h5model_from_weights(bucket_name, file_path, base_model=model)
        except Exception as ex:
            raise ImportError(f"load model failed!\n {str(ex)}")
    
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
            encoding = self.tokenizer.encode(in_text, add_special_tokens=False)  # Không thêm special tokens
            # pad encoding
            encoding = pad_sequences([encoding], maxlen=max_length)
            # Convert encoding to int32
            encoding = tf.cast(encoding, dtype=tf.int32)
            
            # predict next word
            yhat = self.yolo8_model.predict([image_feature, encoding], verbose=0)
            # get index with highest probability
            yhat = np.argmax(yhat)
            # map integer to word
            word = self.idx_to_word_bert(yhat, self.tokenizer)

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
            feature_matrix = self.forward(image_rgb)
            print('HERE', feature_matrix.shape)

        try:
            caption = self.predict_caption(feature_matrix, 315)
        except Exception as ex:
            caption = None
            raise ValueError(str(ex))
        return caption
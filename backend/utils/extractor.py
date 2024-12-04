import torch
from torchvision import transforms
# from ultralytics import YOLO
import torch
import torch.nn as nn
import torch
import tensorflow as tf
from ultralytics import YOLO
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model


class YOLOFeatureExtractorModel(nn.Module):
    def __init__(self, model_path:str='yolov8n.pt'):
        super(YOLOFeatureExtractorModel, self).__init__()
        self.model_path = model_path
        self.preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_backbone(self):
        model = YOLO(self.model_path)
        # Access the backbone layers
        print('HERE')
        backbone = model.model.model[:10]  # Layers 0 to 9 form the backbone
        print('END HERE')
        # Create a new Sequential model with just the backbone layers
        backbone_model = torch.nn.Sequential(*backbone)
        return backbone_model

    def preprocess_image(self, image):
        tensor_image = self.preprocessor(image).unsqueeze(0)  # Thêm batch dimension
        return tensor_image

    def forward(self, image):
        # Tiền xử lý ảnh
        image_tensor = self.preprocess_image(image)
        backbone_model = self.get_backbone()
        # Trích xuất đặc trưng với backbone model
        with torch.no_grad():
            features = backbone_model(image_tensor)
        return features
    

# Định nghĩa lớp LSTM tùy chỉnh để xử lý tham số time_major
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Loại bỏ tham số time_major nếu có
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)


class VGG16FeatureExtractorModel():
    def __init__(self) -> None:
        vgg16 = VGG16()
        self.vgg_extractor = Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)
        
    def preprocess_image(self, image):
        if image.ndim == 3:
            # image = np.resize(image, (224, 224))
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image_tensor = preprocess_input(image)
            return image_tensor
        else:
            raise Exception('image dimension is not valid')
            
    def forward(self, image):
        # Tiền xử lý ảnh
        image_tensor = self.preprocess_image(image)
        features = self.vgg_extractor.predict(image_tensor, verbose=0)
        return features
    
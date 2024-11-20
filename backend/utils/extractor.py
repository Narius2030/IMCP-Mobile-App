import torch
from torchvision import transforms
# from ultralytics import YOLO
import torch
import torch.nn as nn
import torch
from ultralytics import YOLO


class FeatureExtractorModel(nn.Module):
    def __init__(self, model_path:str='yolov8n.pt'):
        super(FeatureExtractorModel, self).__init__()
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
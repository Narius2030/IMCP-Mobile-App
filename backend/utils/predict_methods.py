from transformers import VisionEncoderDecoderModel, AutoTokenizer
import torch
from PIL import Image
import cv2
from torchvision import transforms
from tqdm.notebook import tqdm
# from ultralytics import YOLO
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.nn as nn
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model_yolo = YOLO('yolov8n.pt')

# Access the backbone layers
backbone = model_yolo.model.model[:10]  # Layers 0 to 9 form the backbone

# Create a new Sequential model with just the backbone layers
backbone_model = torch.nn.Sequential(*backbone)

class FeatureExtractorModel(nn.Module):
    def __init__(self, backbone_model):
        super(FeatureExtractorModel, self).__init__()
        self.backbone_model = backbone_model
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        tensor_image = self.preprocess(image).unsqueeze(0)  # Thêm batch dimension
        return tensor_image

    def forward(self, image):
        # Tiền xử lý ảnh
        image_tensor = self.preprocess_image(image)

        # Trích xuất đặc trưng với backbone model
        with torch.no_grad():
            features = self.backbone_model(image_tensor)
        
        # Kiểm tra và điều chỉnh kích thước đầu ra của features nếu cần
        if features.shape[1] != 3 or features.shape[2:] != (224, 224):
            features = torch.nn.functional.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
            features = features[:, :3, :, :]  # Chỉ giữ 3 kênh đầu

        # Loại bỏ chiều batch
        features = features#.squeeze(0)
        return features

# Sử dụng FeatureExtractorModel để trích xuất đặc trưng từ ảnh
# Giả sử bạn đã có `backbone_model` (ví dụ, YOLOv8 model)
feature_extractor = FeatureExtractorModel(backbone_model)

# feature_extractor = torch.load('D:/test/yolo_feature_extractor/feature_extractor_model.pth')
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained('D:/test/yolo_gpt2_v6ep')

# Predict local image
def predict_caption(image_path):
    # Mở ảnh và chuyển đổi thành RGB
    image = Image.open(image_path).convert("RGB")
    
    # Chuyển đổi ảnh thành tensor trước khi truyền vào feature_extractor
    transform_to_tensor = transforms.ToTensor()

    image_tensor = transform_to_tensor(image)  # Thêm batch dimension
    print(image_tensor.shape)

    # Trích xuất đặc trưng từ ảnh
    pixel_values = feature_extractor(image_tensor)
    print(pixel_values.shape)
    
    # Sinh chú thích cho ảnh
    output_ids = model.generate(
                pixel_values
                ,max_length=150 
                ,min_length=10
                ,temperature=0.8
                ,repetition_penalty=1.2 
                ,early_stopping=True
            )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Hiển thị ảnh và chú thích
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return caption

# Thử tạo caption cho một ảnh mới
image_path = "D:/test/img2.jpg"
print(predict_caption(image_path))

    
# Predict internet image
def predict_internet_caption(image_path):
    response = requests.get(image_path)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    # Đọc ảnh bằng OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Chuyển đổi từ BGR (OpenCV) sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Sử dụng FeatureExtractorModel để trích xuất đặc trưng
    pixel_values = feature_extractor(image_rgb)
    
    output_ids = model.generate(
                pixel_values
                ,max_length=150 
                ,min_length=10
                ,temperature=0.8
                ,repetition_penalty=1.2 
                ,early_stopping=True
            )
    
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    plt.imshow(image_rgb)
    return caption

# Test image from app
# image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241106132957.jpg"
# image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241106133350.jpg"
# image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241110014605.jpg"
# image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241110100224.jpg"
# image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241110100425.jpg"
image_path = "http://116.118.50.253:9000/mlflow/user_images/image_20241110100519.jpg"

print(predict_internet_caption(image_path))
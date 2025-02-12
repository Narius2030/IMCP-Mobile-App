import requests
# import base64
# import numpy as np
# import cv2
# from PIL import Image
# from core.config import get_settings

# Bên Phú
def send_request():
    payload = {
        "page": 1,
        "per_page": 1
    }
    
    header = {
        'Authorization': 'Bearer <YOUR-TOKEN>'
    }
    
    resp = requests.request(method='GET', url='http://116.118.50.253:80/api/v1/metadata/full', params=payload, headers=header)
    print(resp.json()[0])
    # resp = requests.request(method='POST', url='http://116.118.50.253:8000/api/v1/generation/yolo8-gpt', json=payload)
    # print(resp.content)
    # resp = requests.request(method='POST', url='http://localhost:8000/api/v1/generation/ingest-user-data', json=payload)
    # print(resp.content)
    # resp = requests.request(method='GET', url='http://localhost:8000/api/v1/generation/vgg16-lstm', json=payload)
    # print(resp.content)



if __name__ == '__main__':
    send_request()
    

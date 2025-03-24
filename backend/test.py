import requests
import base64
import time
from PIL import Image


def image_generator():
    with open('./data/workspace.jpg', 'rb') as file:
        image_bytes = file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    with Image.open('./data/workspace.jpg') as img:
        width, height = img.size
    return image_base64, width, height


def send_request(image_base64, width, height):
    metadata_payload = {
        "page": 1,
        "per_page": 1
    }
    
    image_payload = {
        "image_pixels": image_base64,
        'shape': tuple([width, height])
    }
    
    
    # resp = requests.request(method='POST', url='http://36.50.135.226:80/api/v1/metadata/metadata', json="1970-01-01T00:00:00.000+00:00")
    # print(len(resp.json()))
    
    resp = requests.request(method='POST', url='http://36.50.135.226:80/api/v1/model/predict-caption', json=image_payload)
    print(resp.content.decode('utf-8'))

    # resp = requests.request(method='POST', url='http://36.50.135.226:80/api/v1/model/ingest-user-data', json=image_payload)
    # print(resp.content.decode('utf-8'))


if __name__ == '__main__':
    image_base64, width, height = image_generator()
    send_request(image_base64, width, height)
    

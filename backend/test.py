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
    # print(resp.json())
    
    # resp = requests.request(method='POST', url='http://36.50.135.226:80/api/v1/model/predict-caption', json=image_payload)
    # print(resp.content.decode('utf-8'))
    with requests.post('http://localhost:8000/api/v1/model/predict-caption', json=image_payload, stream=True) as resp:
        for chunk in resp.iter_lines(decode_unicode=True):
            if chunk:
                print(chunk)  # Print ngay khi nhận data

    # resp = requests.request(method='POST', url='http://36.50.135.226:80/api/v1/model/ingest-user-data', json=image_payload)
    # print(resp.content.decode('utf-8'))


if __name__ == '__main__':
    # image_base64, width, height = image_generator()
    # send_request(image_base64, width, height)
    import re
    
    caption = "bãi đỗ xe có nhiều xe_máy . biển số 08b2 ở bên phải . xe_máy cùng chiều với tôi . vị_trí bạn đứng là giữa bãi đỗ xe . vỉa_hè không có ở đây . di_chuyển an_toàn bằng cách chú_ý các xe xung_quanh ."
    normalized_caption = re.split(r'[ _]', caption)
    # concatenate '.' with forward word
    final_caption_words = list()
    for idx, word in enumerate(normalized_caption):
        if word == '.' and final_caption_words:
            final_caption_words[-1] = normalized_caption[idx-1] + word
        else:
            final_caption_words.append(word)
    # unify to a string
    final_caption = ' '.join(final_caption_words)
    print(final_caption)
    

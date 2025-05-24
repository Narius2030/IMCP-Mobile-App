import torch
import torch.nn.functional as F
import requests
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from io import BytesIO

# Load Extract Feature
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load tokenizer của BartPho
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")

# Đường dẫn thư mục chứa các tệp mô hình
model_path = './tmp/bartpho_vit_gpt2'

# Tải mô hình VisionEncoderDecoder
model = VisionEncoderDecoderModel.from_pretrained(model_path, ignore_mismatched_sizes=True)

# Load ảnh từ URL
# url = "http://160.191.244.13:9000/lakehouse/imcp/augmented-data/images/85b5875685050868f8309974c79ae6ed.jpg"
url = "https://media.istockphoto.com/id/172304454/vi/anh/vintage-ng%C6%B0%E1%BB%9Di-%C4%91%C3%A0n-%C3%B4ng-%C4%91%E1%BA%A7u-ti%C3%AAn-tr%C3%AAn-tem-m%E1%BA%B7t-tr%C4%83ng-m%C6%B0%E1%BB%9Di-xu.jpg?s=612x612&w=0&k=20&c=5vNeO4nK3PsqSP6Ns0zPyt2JK9OK8nph-Nh505g13eM="
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# Tiền xử lý ảnh
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.bos_token_id
generated_ids = [model.config.decoder_start_token_id]
max_length = 150
probs = []
logits = []

def compute_caption_entropy(token_logits):
    probs = F.softmax(token_logits, dim=-1)  # Shape: (T, V)
    log_probs = F.log_softmax(token_logits, dim=-1)  # Shape: (T, V)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (T,)
    avg_entropy = entropy.mean().item()
    return avg_entropy

# Sinh từng token và lưu lại xác suất tương ứng
for _ in range(max_length):
    input_ids = torch.tensor([generated_ids])
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, decoder_input_ids=input_ids)

    # Lấy xác suất tại bước hiện tại
    next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)  # [1, vocab_size]
    probs.append(next_token_probs.squeeze().cpu().numpy())  # Lưu xác suất
    logits.append(next_token_logits)  # Lưu logits

    # Sinh token tiếp theo (greedy)
    next_token = torch.argmax(next_token_probs, dim=-1).item()
    generated_ids.append(next_token)

    # Nếu gặp token kết thúc, dừng lại
    if next_token == tokenizer.eos_token_id:
        break

all_logits = torch.cat(logits, dim=0)
caption_entropy = compute_caption_entropy(all_logits)
print(f"Caption entropy: {caption_entropy:.4f}")

# Hiển thị caption và xác suất
caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Generated caption:", caption)

# # probs: danh sách các vector xác suất ứng với từng token trong caption
# caption_probs = []
# print("\nProbabilities per token:")
# for i, prob in enumerate(probs):
#     word_token = prob.argsort()[-1]
#     token = tokenizer.decode([word_token])
#     print(f"Word {i+1}:  {token:>10} ---> {prob[word_token]:.4f}")
#     caption_probs.append(prob[word_token])
#     print(caption_probs)

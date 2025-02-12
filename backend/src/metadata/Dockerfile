# Sử dụng Python 3.11 làm image cơ sở
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY . .

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Khởi chạy FastAPI khi container được khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
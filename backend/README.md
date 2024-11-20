# Swagger UI

![image](https://github.com/user-attachments/assets/8827a046-8f53-43e5-8c14-607304f54c73)



# Containerizing project

```Dockerfile
# Sử dụng Python 3.11 làm image cơ sở
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY . .

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install the cv2 dependencies that might be missing in  Docker container 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Mở port 8000
EXPOSE 8000

# Khởi chạy FastAPI khi container được khởi động
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

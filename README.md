![image](https://github.com/user-attachments/assets/f63f8add-2e4d-45b9-ac22-807c986b4ff0)


For containerizing the backend to deploy on server, we should run this Dockerfile. Furthermore, Our backend is erect on `FastAPI` which is one of Python's frameworks
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

Demo of mobile application

![image](https://github.com/user-attachments/assets/9d0ef6cc-d816-49b5-af7b-c3fef1dac79c)

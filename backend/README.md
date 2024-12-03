# Swagger UI

![image](https://github.com/user-attachments/assets/00c4c543-73f4-4740-bcb0-92848f4dca0c)


# Setup Storage System
We are using MinIO and MongoDB for storage system:
* MongoDB: we utilize Atlas which is a cloud platform
* MinIO: we run on Ubuntu server throughout Docker Compose
 ```docker-compose
  version: "3.9"

services:
  minio:
    hostname: minio
    image: "minio/minio"
    container_name: minio
    ports:
      - "9001:9001"
      - "9000:9000"
    command: [ "server", "/data", "--console-address", ":9001" ]
    volumes:
      - ./data/miniodata:/data
    env_file:
      - .env
    networks:
      - data_network

networks:
  data_network:
    driver: bridge
    name: data_network

volumes:
  postgres_data_h: {}
  miniodata: {}
```


# Containerizing project

We use Docker to deploy API services. These code below will containerize your backend application
```Dockerfile
# Use Python 3.11 as base image
FROM python:3.11-slim

# Set up working directory
WORKDIR /app

# Copy the necessary files into the container
COPY . .

# Install necessary libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install the cv2 dependencies that might be missing in  Docker container 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Open port 8000
EXPOSE 8000

# Command line to start the service on container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

# Application Architecture

![image](https://github.com/user-attachments/assets/2f49ba20-5926-4e5a-b921-12c1389a27fa)

# API Routing

In file `./gateway/nginx.conf`, you have to identify endpoints in Nginx that point to appropriate services for each request

```config
server {
        listen 80;

        location /docs {
            proxy_pass http://metadata-service:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/v1/metadata {
            proxy_pass http://metadata-service:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
}
```

- Configure Model Service Container

```Dockerfile
# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files into the container
COPY . .

# Install the required libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install the cv2 dependencies that might be missing in the Docker container 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Start FastAPI when the container is launched
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- Configure Metadata Service Container

```Dockerfile
# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files into the container
COPY . .

# Install the required libraries
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Start FastAPI when the container is launched
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


# Containerizing project

Run below command to create and start containers of micro-service

```cmd
docker-compose up --build
```

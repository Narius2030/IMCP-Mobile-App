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
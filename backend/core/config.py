import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv('MONGO_ATLAS_PYTHON')
    
    # Kafka
    KAFKA_ADDRESS:str = os.getenv('KAFKA_ADDRESS')
    KAFKA_PORT:str = os.getenv('KAFKA_PORT')
    
    # JWT 
    JWT_SECRET: str = os.getenv('JWT_SECRET')
    JWT_ALGORITHM: str = os.getenv('JWT_ALGORITHM')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv('JWT_TOKEN_EXPIRE_MINUTES')
    
    # MinIO
    MINIO_HOST:str = os.getenv('MINIO_HOST')
    MINIO_PORT:str = os.getenv('MINIO_PORT')
    MINIO_USER:str = os.getenv('MINIO_USER')
    MINIO_PASSWD:str = os.getenv('MINIO_PASSWD')
    MINIO_URL:str = os.getenv('MINIO_URL')
    
    MINIO_HOST_VPS02:str = os.getenv('MINIO_HOST_VPS02')
    MINIO_PORT_VPS02:str = os.getenv('MINIO_PORT_VPS02')
    MINIO_USER_VPS02:str = os.getenv('MINIO_USER_VPS02')
    MINIO_PASSWD_VPS02:str = os.getenv('MINIO_PASSWD_VPS02')
    MINIO_URL_VPS02:str = os.getenv('MINIO_URL_VPS02')
    
    # Trino
    TRINO_USER:str = os.getenv('TRINO_USER')
    TRINO_HOST:str = os.getenv('TRINO_HOST')
    TRINO_PORT:str = os.getenv('TRINO_PORT')
    TRINO_CATALOG:str = os.getenv('TRINO_CATALOG')
    
    # Image Labels
    LABELS: list[str] = [
        "traffic image",
        "non-traffic image",
        "sidewalk",
        "pedestrian bridge",
        "underpass",
        "traffic light",
        "crosswalk",
        "road sign",
        "highway",
        "intersection",
        "roundabout",
        "guardrail",
        "bus stop",
        "parking lot",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle"
    ]
    
def get_settings() -> Settings:
    return Settings()
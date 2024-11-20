import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv('MONGO_ATLAS_PYTHON')
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
    # Tokenizer
    BERT_TOKENIZERS:str = os.getenv('BERT_TOKENIZERS')
    
def get_settings() -> Settings:
    return Settings()
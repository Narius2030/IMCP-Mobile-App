import sys
sys.path.append('./')

import os
from core.config import get_settings
from transformers import VisionEncoderDecoderModel
from utils.operators.storage import MinioStorageOperator

settings = get_settings()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST_VPS02}:{settings.MINIO_PORT_VPS02}', 
                                      access_key=settings.MINIO_USER_VPS02, 
                                      secret_key=settings.MINIO_PASSWD_VPS02)


class ModelLoaders():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load_gpt_configs(bucket_name:str, object_name:str, temp_dir:str='./tmp/bartpho_vit_gpt2', version_id=None):
        """Get `GPT configs` data in stream bytes and save in temporary file

        Args:
            bucket_name (str): Name of bucket containing that config object MinIO
            object_name (str): the path to config object in MinIO
            temp_dir (str, optional): the path of directory to save temporaily. Defaults to '/app/tmp'.
            version_id (str, optional): The version of weight object. Defaults to None - latest.

        Returns:
            str: the temporary path of `GPT configs` file
        """
        try:
            config_url = minio_operator.get_object_bytes(bucket_name, f"{object_name}/config.json", version_id=version_id)
            generation_config_url = minio_operator.get_object_bytes(bucket_name, f"{object_name}/generation_config.json", version_id=version_id)
            model_url = minio_operator.get_object_bytes(bucket_name, f"{object_name}/model.safetensors", version_id=version_id)
            # config_url = None
            # generation_config_url = None
            # model_url = None
            datasets = {
                'config.json': config_url, 
                'generation_config.json': generation_config_url,
                'model.safetensors': model_url
            }
            # save model's configs in temporary file
            file_names = []
            for key, data in datasets.items():
                temp_file_path = os.path.join(temp_dir, key)
                if os.path.exists(temp_file_path):
                    print(f"File {temp_file_path} already exists")
                else:
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(data)
                    print(f"Object loaded successfully from MinIO. {key}: {len(data)}")
                file_names.append(temp_file_path)
            return temp_dir, file_names
        
        except Exception as e:
            print(f"Error loading configs from MinIO: {str(e)}")
            return None
    
    @staticmethod
    def load_gptmodel_from_configs(bucket_name:str, object_name:str) -> VisionEncoderDecoderModel:
        """Load GPT model from config data in stream bytes and save in temporary file

        Args:
            bucket_name (str): Name of bucket containing that config object MinIO
            object_name (str): the path to weight config in MinIO

        Returns:
            VisionEncoderDecoderModel: the GPT model
        """
        model_path, _ = ModelLoaders.load_gpt_configs(bucket_name, object_name)
        try:
            print(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
            print("Model loaded successfully from temporary file.")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        finally:
            for file in file_names:
                os.remove(file)
import tempfile
import os
from core.config import get_settings
from transformers import VisionEncoderDecoderModel
from utils.storage import MinioStorageOperator

settings = get_settings()
minio_operator = MinioStorageOperator(endpoint=f'{settings.MINIO_HOST}:{settings.MINIO_PORT}', access_key=settings.MINIO_USER, secret_key=settings.MINIO_PASSWD)


class ModelLoaders():
    def __init__(self) -> None:
        pass
    
    def load_weight_bytes(self, bucket_name:str, object_name:str, temp_dir:str='D:/tmp', version_id=None) -> str:
        """Get `h5 weights` data in stream bytes and save in temporary file

        Args:
            bucket_name (str): Name of bucket containing that weight object MinIO
            object_name (str): the path to weight object in MinIO
            temp_dir (str, optional): the path of directory to save temporaily. Defaults to '/app/tmp'.
            version_id (str, optional): The version of weight object. Defaults to None - latest.

        Returns:
            str: the temporary path of `h5 weight` file
        """        
        try:
            # Lấy đối tượng từ MinIO dưới dạng byte stream
            data = minio_operator.load_object_bytes(bucket_name, object_name, version_id=version_id)
            # Tạo một file tạm để lưu dữ liệu
            with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".weights.h5") as temp_file:
                temp_file.write(data)
            print(f"Object loaded successfully from MinIO. Object size: {len(data)} bytes")
            return temp_file.name
        except Exception as e:
            print(f"Error loading weights from MinIO: {str(e)}")
            return None
    
    def load_gpt_configs(self, bucket_name:str, object_name:str, temp_dir:str='D:/tmp/yolo_gpt2_v6ep', version_id=None):
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
            # Lấy đối tượng từ MinIO dưới dạng byte stream
            config_url = minio_operator.load_object_bytes(bucket_name, f"{object_name}/config.json", version_id=version_id)
            generation_config_url = minio_operator.load_object_bytes(bucket_name, f"{object_name}/generation_config.json", version_id=version_id)
            model_url = minio_operator.load_object_bytes(bucket_name, f"{object_name}/model.safetensors", version_id=version_id)
            datasets = {
                'config.json': config_url, 
                'generation_config.json': generation_config_url, 
                'model.safetensors': model_url
            }
            # Tạo một file tạm để lưu dữ liệu
            file_names = []
            for key, data in datasets.items():
                temp_file_path = os.path.join(temp_dir, key)
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(data)
                file_names.append(temp_file_path)
                print(f"Object loaded successfully from MinIO. {key}: {len(data)}")
            return temp_dir, file_names
        except Exception as e:
            print(f"Error loading configs from MinIO: {str(e)}")
            return None
    
    def load_gptmodel_from_minio(self, bucket_name:str, object_name:str) -> VisionEncoderDecoderModel:
        """Load GPT model from config data in stream bytes and save in temporary file

        Args:
            bucket_name (str): Name of bucket containing that config object MinIO
            object_name (str): the path to weight config in MinIO

        Returns:
            VisionEncoderDecoderModel: the GPT model
        """
        model_path, file_names = self.load_gpt_configs(bucket_name, object_name)
        try:
            print(model_path)
            # Load VisionEncoderDecoderModel
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
            print("Model loaded successfully from temporary file.")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        finally:
            for file in file_names:
                os.remove(file)
            
    def load_h5model_from_minio(self, bucket_name:str, object_name:str, version_id=None, base_model=None):
        """Load GPT model from config data in stream bytes and save in temporary file

        Args:
            bucket_name (str): Name of bucket containing that weight object MinIO
            object_name (str): the path to weight config in MinIO

        Returns:
            any: the Keras model
        """
        temp_weight_path = self.load_weight_bytes(bucket_name, object_name, version_id)
        print(temp_weight_path)
        try:
            base_model.load_weights(temp_weight_path)
            print("Model loaded successfully from temporary file.")
            return base_model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        finally:
            os.remove(temp_weight_path)
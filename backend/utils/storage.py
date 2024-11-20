from minio import Minio
from minio.error import S3Error
import tempfile
import os


class MinioStorageOperator:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        """
        Khởi tạo kết nối với MinIO.
        
        :param endpoint: Địa chỉ máy chủ MinIO (host:port).
        :param access_key: Khóa truy cập MinIO.
        :param secret_key: Khóa bí mật MinIO.
        :param secure: Sử dụng HTTPS (mặc định là True).
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure  # Đặt thành True nếu dùng HTTPS
        )

    def upload_file(self, bucket_name, object_name, file_path):
        """
        Upload tệp lên MinIO.

        :param bucket_name: Tên bucket trong MinIO.
        :param file_path: Đường dẫn đến tệp cần upload.
        :param object_name: Tên đối tượng sẽ lưu trên MinIO.
        """
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"Upload thành công: {object_name}")
        except S3Error as e:
            print(f"Lỗi khi upload tệp: {str(e)}")

    def download_file(self, bucket_name, object_name, download_path, version_id=None):
        """
        Download tệp từ MinIO về máy.

        :param bucket_name: Tên bucket trong MinIO.
        :param object_name: Tên đối tượng trên MinIO cần tải về.
        :param download_path: Đường dẫn lưu tệp tải về.
        """
        try:
            self.client.fget_object(bucket_name, object_name, download_path, version_id=version_id)
            print(f"Download thành công: {object_name}")
        except S3Error as e:
            print(f"Lỗi khi download tệp: {str(e)}")

    def create_bucket(self, bucket_name):
        """
        Tạo bucket trong MinIO nếu chưa tồn tại.

        :param bucket_name: Tên bucket cần tạo.
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print(f"Đã tạo bucket: {bucket_name}")
            else:
                print(f"Bucket {bucket_name} đã tồn tại.")
        except S3Error as e:
            print(f"Lỗi khi tạo bucket: {str(e)}")
    
    def create_presigned_url(self, bucket_name, object_name) -> str:
        """
        Create new *Presigned URL* in MinIO.

        :param bucket_name: Name of bucket containing that object.
        """
        return self.client.get_presigned_url(
            method='GET',
            bucket_name=bucket_name,
            object_name=object_name
        )
        
    def load_object_bytes(self, bucket_name, object_name, version_id=None):
        """
        Get object in stream bytes from MinIO.

        :param bucket_name: Name of bucket containing that object.
        """
        try:
            # Lấy đối tượng từ MinIO dưới dạng byte stream
            response  = self.client.get_object(bucket_name, object_name, version_id=version_id)
            data = response.read()
            return data
        except S3Error as err:
            print(f"Error loading file: {err}")
    
    def upload_object_bytes(self, objec_data, bucket_name:str, object_name:str, content_type:str):
        """
        Upload đối tượng dưới dạng bytes từ một đường dẫn URL

        :param url: đường dẫn gốc của đối tượng trên internet
        :param bucket_name: tên bucket
        :param object_name: đường dẫn tới tên của đối tượng trên MinIO
        """
        try:
            self.client.put_object(
                bucket_name = bucket_name,
                object_name = object_name,
                data = objec_data,
                length = objec_data.getbuffer().nbytes,
                content_type=content_type
            )
            print(f"Successfully uploaded {object_name} to {bucket_name}!")
        except S3Error as err:
            print(f"Error uploading file: {err}")
    
    def write_object_to_local(self, stream, path):
        if stream:
            with open(path, "wb") as f:
                f.write(stream.getvalue())
            print("File downloaded and saved locally.")
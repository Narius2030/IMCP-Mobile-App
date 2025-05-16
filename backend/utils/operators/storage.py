from minio import Minio
from minio.error import S3Error
import tempfile
import os


class MinioStorageOperator:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        """
        Initialize a connection to MinIO storage service.

        Args:
            endpoint (str): MinIO server endpoint in format 'host:port'.
            access_key (str): MinIO access key for authentication.
            secret_key (str): MinIO secret key for authentication.
            secure (bool, optional): Whether to use HTTPS for the connection. Defaults to False.

        Note:
            The Minio client is initialized with the provided credentials and stored in the instance.
            Set secure=True if you need to use HTTPS for the connection.
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
    
    def get_list_objects(self, bucket_name:str, partition:str=""):
        """
        Retrieve a list of objects from a specified MinIO bucket and partition.

        Args:
            bucket_name (str): Name of the MinIO bucket to list objects from.
            partition (str, optional): Partition path within the bucket. Defaults to empty string.
                Objects will be listed under the path 'imcp/encoded-data/{partition}'.

        Returns:
            list: A list of MinIO objects found in the specified bucket and partition.

        Note:
            The function recursively lists all objects under the specified partition path.
            If an error occurs during listing, it will print the error message.
        """
        try:
            files = []
            objects = self.client.list_objects(bucket_name, prefix=f"imcp/encoded-data/{partition}", recursive=True)
            for obj in objects:
                files.append(obj)
            return files
        except S3Error as err:
            print(f"Error listing objects: {err}")
    
    def upload_file(self, bucket_name, object_name, file_path):
        """
        Upload a file to MinIO storage.

        Args:
            bucket_name (str): Name of the MinIO bucket to upload to.
            object_name (str): Name of the object to be stored in MinIO.
            file_path (str): Path to the local file to be uploaded.

        Note:
            Prints success message if upload is successful, or error message if upload fails.
        """
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            print(f"Upload thành công: {object_name}")
        except S3Error as e:
            print(f"Lỗi khi upload tệp: {str(e)}")

    def download_file(self, bucket_name, object_name, download_path, version_id=None):
        """
        Download a file from MinIO storage to the local machine.

        Args:
            bucket_name (str): Name of the MinIO bucket containing the file.
            object_name (str): Name/path of the object to download from MinIO.
            download_path (str): Local file path where the downloaded file will be saved.
            version_id (str, optional): Specific version ID of the object to download. 
                                      If None, downloads the latest version.

        Note:
            Prints a success message if the download is successful, or an error message if the download fails.
            Uses MinIO's fget_object method to perform the download operation.
        """
        try:
            self.client.fget_object(bucket_name, object_name, download_path, version_id=version_id)
            print(f"Download thành công: {object_name}")
        except S3Error as e:
            print(f"Lỗi khi download tệp: {str(e)}")

    def create_bucket(self, bucket_name):
        """
        Create a new bucket in MinIO storage if it doesn't already exist.

        Args:
            bucket_name (str): Name of the bucket to create.

        Note:
            Prints a success message if the bucket is created, or a message if it already exists.
            Prints an error message if bucket creation fails.
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
        Generate a presigned URL for accessing an object in MinIO storage.

        Args:
            bucket_name (str): Name of the bucket containing the object.
            object_name (str): Name/path of the object to generate URL for.

        Returns:
            str: A presigned URL that can be used to access the object.
        """
        return self.client.get_presigned_url(
            method='GET',
            bucket_name=bucket_name,
            object_name=object_name
        )
        
    def get_object_bytes(self, bucket_name, object_name, version_id=None):
        """
        Retrieve an object from MinIO storage as a byte stream.

        Args:
            bucket_name (str): Name of the bucket containing the object.
            object_name (str): Name/path of the object to retrieve.
            version_id (str, optional): Specific version ID of the object to retrieve.
                                      If None, retrieves the latest version.

        Returns:
            bytes: The object data as a byte stream.

        Note:
            Prints an error message if the retrieval fails.
        """
        try:
            response = self.client.get_object(bucket_name, object_name, version_id=version_id)
            data = response.read()
            response.close()
            return data
        except S3Error as err:
            print(f"Error loading file: {err}")
    
    def upload_object_bytes(self, objec_data, bucket_name:str, object_name:str, content_type:str):
        """
        Upload an object to MinIO storage from byte data.

        Args:
            objec_data (bytes): The object data to upload.
            bucket_name (str): Name of the bucket to upload to.
            object_name (str): Name/path to store the object under in MinIO.
            content_type (str): MIME type of the object being uploaded.

        Note:
            Prints a success message if the upload is successful, or an error message if it fails.
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
        """
        Write a stream object to a local file.

        Args:
            stream (io.BytesIO): The stream object containing the data to write.
            path (str): Local file path where the data will be written.

        Note:
            Prints a success message if the write operation is successful.
            Only attempts to write if the stream is not None.
        """
        if stream:
            with open(path, "wb") as f:
                f.write(stream.getvalue())
            print("File downloaded and saved locally.")
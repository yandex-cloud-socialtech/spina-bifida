import boto3
import os
import uuid
import logging
from datetime import datetime
from botocore.exceptions import NoCredentialsError

# Настройка логирования
logger = logging.getLogger(__name__)

class S3Storage:
    def __init__(self, bucket_name):
        """
        Инициализация клиента S3
        
        Args:
            bucket_name (str): Имя бакета в Object Storage
        """
        self.bucket_name = bucket_name
        self.endpoint_url = 'https://storage.yandexcloud.net'
        self.access_key = os.getenv('ACCESS_KEY')
        self.secret_key = os.getenv('SECRET_KEY')
        self.s3_client = self._init_s3_client()
        
    def _init_s3_client(self):
        """
        Инициализация клиента S3
        
        Returns:
            boto3.client: Клиент S3 или None в случае ошибки
        """
        try:
            session = boto3.session.Session()
            s3_client = session.client(
                service_name='s3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key
            )
            return s3_client
        except Exception as e:
            logger.error(f"Ошибка инициализации S3 клиента: {e}")
            return None
    
    def upload_file(self, file_path, object_name=None):
        """
        Загрузка файла в S3
        
        Args:
            file_path (str): Путь к локальному файлу
            object_name (str, optional): Имя объекта в S3. Если не указано, используется имя файла
            
        Returns:
            tuple: (bool, str) - результат загрузки и имя объекта в S3
        """
        try:
            if not self.s3_client:
                logger.error("S3 клиент не инициализирован")
                return False, None
            
            if file_path is None:
                logger.error("Путь к файлу не может быть None")
                return False, None
            
            # Если имя объекта не указано, используем имя файла
            if object_name is None:
                object_name = os.path.basename(file_path)
            
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            logger.info(f"Файл {file_path} успешно загружен в S3 как {object_name}")
            
            # Удаление локального файла после успешной загрузки
            try:
                os.remove(file_path)
                logger.info(f"Локальный файл {file_path} удален")
            except Exception as e:
                logger.warning(f"Не удалось удалить локальный файл {file_path}: {e}")
            
            return True, object_name
        except FileNotFoundError:
            logger.error(f"Файл {file_path} не найден")
            return False, None
        except NoCredentialsError:
            logger.error("Ошибка с учетными данными S3")
            return False, None
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path} в S3: {e}")
            return False, None
    
    def upload_bytes(self, file_bytes, object_name):
        """
        Загрузка байтов в S3
        
        Args:
            file_bytes (bytes): Байты файла
            object_name (str): Имя объекта в S3
            
        Returns:
            tuple: (bool, str) - результат загрузки и имя объекта в S3
        """
        try:
            if not self.s3_client:
                logger.error("S3 клиент не инициализирован")
                return False, None
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=file_bytes
            )
            logger.info(f"Байты успешно загружены в S3 как {object_name}")
            return True, object_name
        except NoCredentialsError:
            logger.error("Ошибка с учетными данными S3")
            return False, None
        except Exception as e:
            logger.error(f"Ошибка загрузки байтов в S3: {e}")
            return False, None
    
    def upload_json(self, json_data, object_name):
        """
        Загрузка JSON данных в S3
        
        Args:
            json_data (str): JSON данные в виде строки
            object_name (str): Имя объекта в S3
            
        Returns:
            tuple: (bool, str) - результат загрузки и имя объекта в S3
        """
        try:
            if not self.s3_client:
                logger.error("S3 клиент не инициализирован")
                return False, None
            
            if json_data is None:
                logger.error("JSON данные не могут быть None")
                return False, None
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"JSON данные успешно загружены в S3 как {object_name}")
            return True, object_name
        except NoCredentialsError:
            logger.error("Ошибка с учетными данными S3")
            return False, None
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON данных в S3: {e}")
            return False, None

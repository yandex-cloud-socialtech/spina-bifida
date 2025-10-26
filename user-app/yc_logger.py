import json
import logging
import uuid
import os
import time
from datetime import datetime, timezone

import yandexcloud
from yandex.cloud.logging.v1.log_ingestion_service import WriteRequest, LogIngestionServiceStub
from yandex.cloud.logging.v1.log_resource import LogEntryResource
from yandex.cloud.logging.v1.log_entry import LogEntry, LogEntryLevel
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

# Настройка логирования
logger = logging.getLogger(__name__)

class YCLogger:
    def __init__(self, folder_id=None, log_group_id=None, service_account_key_file=None):
        """
        Инициализация логгера Yandex Cloud
        
        Args:
            folder_id (str, optional): ID каталога в Yandex Cloud
            log_group_id (str, optional): ID группы логов
            service_account_key_file (str, optional): Путь к файлу с ключом сервисного аккаунта
        """
        self.folder_id = folder_id or os.getenv('YC_FOLDER_ID')
        self.log_group_id = log_group_id or os.getenv('YC_LOG_GROUP_ID')
        self.service_account_key_file = service_account_key_file or os.getenv('YC_SERVICE_ACCOUNT_KEY_FILE')
        self.sdk = None
        self.stub = None
        
        # Проверка наличия необходимых параметров
        if not self.folder_id or not self.log_group_id or not self.service_account_key_file:
            logger.warning("Не указаны folder_id, log_group_id или service_account_key_file для Yandex Cloud Logging. Логирование будет работать только локально.")
            self.enabled = False
        else:
            try:
                # Проверка существования файла с ключом сервисного аккаунта
                if not os.path.exists(self.service_account_key_file):
                    logger.error(f"Файл с ключом сервисного аккаунта не найден: {self.service_account_key_file}")
                    self.enabled = False
                else:
                    self.enabled = True
                    # Инициализация SDK
                    self._init_sdk()
                    logger.info(f"Yandex Cloud Logging успешно инициализирован: folder_id={self.folder_id}, log_group_id={self.log_group_id}")
            except Exception as e:
                logger.error(f"Ошибка инициализации Yandex Cloud Logging: {e}")
                self.enabled = False
    
    def _init_sdk(self):
        """
        Инициализация Yandex Cloud SDK
        
        Returns:
            bool: True, если SDK успешно инициализирован, иначе False
        """
        try:
            # Инициализация SDK с использованием ключа сервисного аккаунта
            self.sdk = yandexcloud.SDK(service_account_key=self.service_account_key_file)
            
            # Создание клиента для LogIngestionService
            self.stub = self.sdk.client(LogIngestionServiceStub)
            
            logger.info("Yandex Cloud SDK успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации Yandex Cloud SDK: {e}")
            self.enabled = False
            return False
    
    def _prepare_log_entry(self, user_id, user_email, user_organization, action_type, action_details=None):
        """
        Подготовка записи лога
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            action_type (str): Тип действия
            action_details (dict, optional): Дополнительные детали действия
            
        Returns:
            dict: Запись лога
        """
        # Формирование лога
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "user_email": user_email,
            "user_organization": user_organization,
            "action_type": action_type,
            "action_details": action_details or {}
        }
        
        return log_entry
    
    def _send_to_yc_logging(self, log_entry, action_type):
        """
        Отправка лога в Yandex Cloud Logging через SDK
        
        Args:
            log_entry (dict): Запись лога
            action_type (str): Тип действия
            
        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.enabled:
            return False
        
        try:
            # Проверка инициализации SDK
            if not self.stub:
                if not self._init_sdk():
                    logger.error("Не удалось инициализировать Yandex Cloud SDK для отправки лога")
                    return False
            
            # Создание timestamp для лога
            timestamp = Timestamp()
            timestamp.GetCurrentTime()
            
            # Создание JSON payload
            json_payload = Struct()
            json_payload.update(log_entry)
            
            # Создание записи лога
            log_entry_pb = LogEntry(
                timestamp=timestamp,
                level=LogEntryLevel.INFO,
                message=f"User action: {action_type} by {log_entry['user_email']}",
                json_payload=json_payload
            )
            
            # Создание ресурса
            resource = LogEntryResource(
                type="app",
                id=f"spina-bifida-{action_type}"
            )
            
            # Создание запроса
            request = WriteRequest(
                destination=WriteRequest.Destination(log_group_id=self.log_group_id),
                resource=resource,
                entries=[log_entry_pb]
            )
            
            # Отправка запроса
            response = self.stub.Write(request)
            
            logger.info(f"Лог действия пользователя {log_entry['user_email']} успешно отправлен в Yandex Cloud Logging: {action_type}")
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки лога в Yandex Cloud Logging: {e}")
            return False
    
    def log_action(self, user_id, user_email, user_organization, action_type, action_details=None):
        """
        Логирование действия пользователя
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            action_type (str): Тип действия
            action_details (dict, optional): Дополнительные детали действия
            
        Returns:
            bool: True, если логирование успешно, иначе False
        """
        try:
            # Подготовка записи лога
            log_entry = self._prepare_log_entry(
                user_id=user_id,
                user_email=user_email,
                user_organization=user_organization,
                action_type=action_type,
                action_details=action_details
            )
            
            # Логирование локально
            logger.info(f"Лог действия пользователя {user_email}: {action_type}")
            logger.debug(f"Детали лога: {json.dumps(log_entry, ensure_ascii=False)}")
            
            # Отправка в Yandex Cloud Logging
            if self.enabled:
                return self._send_to_yc_logging(log_entry, action_type)
            
            return True
        except Exception as e:
            logger.error(f"Ошибка логирования действия пользователя {user_email}: {e}")
            return False
    
    def log_auth_action(self, user_id, user_email, user_organization, action_type, action_details=None):
        """
        Логирование действий авторизации (login/logout)
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            action_type (str): Тип действия (login/logout)
            action_details (dict, optional): Дополнительные детали действия
            
        Returns:
            bool: True, если логирование успешно, иначе False
        """
        return self.log_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type=action_type,
            action_details=action_details
        )
    
    def log_image_upload(self, user_id, user_email, user_organization, original_image_name, image_data, image_url=None):
        """
        Логирование загрузки изображения
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            original_image_name (str): Оригинальное имя изображения
            image_data: Данные изображения (PIL.Image)
            image_url (str, optional): URL изображения в Object Storage
            
        Returns:
            tuple: (bool, str) - результат логирования и URL изображения
        """
        # Генерация уникального имени файла
        unique_id = str(uuid.uuid4())
        new_image_name = f"img_{unique_id}_{original_image_name}"
        
        # Если URL не указан, используем сгенерированное имя
        if not image_url:
            image_url = new_image_name
        
        # Логирование действия
        success = self.log_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type="image_upload",
            action_details={
                "original_image_name": original_image_name,
                "image_url": image_url,
                "upload_time": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return success, image_url
    
    def log_model_response(self, user_id, user_email, user_organization, image_name, s3_object_name, model_responses):
        """
        Логирование ответов моделей по изображению
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            image_name (str): Оригинальное имя изображения
            s3_object_name (str): Имя объекта в S3
            model_responses (dict): Ответы моделей
            
        Returns:
            bool: True, если логирование успешно, иначе False
        """
        return self.log_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type="model_response",
            action_details={
                "original_image_name": image_name,
                "s3_object_name": s3_object_name,
                "model_responses": model_responses,
                "analysis_time": datetime.now(timezone.utc).isoformat()
            }
        )

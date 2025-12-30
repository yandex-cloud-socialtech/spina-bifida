import json
import os
import uuid
import logging
import shutil
from datetime import datetime, timezone
import threading
import time
from s3_storage import S3Storage

# Настройка логирования
logger = logging.getLogger(__name__)

class UserLogger:
    def __init__(self, logs_bucket, images_bucket, local_logs_dir="./logs", max_local_logs_size_mb=500, cleanup_interval_hours=24):
        """
        Инициализация логгера пользовательских действий
        
        Args:
            logs_bucket (str): Имя бакета для логов в Object Storage
            images_bucket (str): Имя бакета для изображений в Object Storage
            local_logs_dir (str): Директория для локального хранения логов
            max_local_logs_size_mb (int): Максимальный размер локальных логов в МБ
            cleanup_interval_hours (int): Интервал очистки локальных логов в часах
        """
        self.logs_bucket = logs_bucket
        self.images_bucket = images_bucket
        self.local_logs_dir = local_logs_dir
        self.max_local_logs_size_mb = max_local_logs_size_mb
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # Создание директории для логов, если она не существует
        os.makedirs(self.local_logs_dir, exist_ok=True)
        
        # Инициализация S3 клиентов
        self.logs_s3 = S3Storage(logs_bucket)
        self.images_s3 = S3Storage(images_bucket)
        
        # Запуск фонового потока для очистки логов
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """
        Запуск фонового потока для периодической очистки локальных логов
        """
        def cleanup_worker():
            while True:
                try:
                    # Очистка логов
                    self._cleanup_local_logs()
                    # Ожидание следующей очистки
                    time.sleep(self.cleanup_interval_hours * 3600)
                except Exception as e:
                    logger.error(f"Ошибка в потоке очистки логов: {e}")
                    # В случае ошибки ждем 1 час перед повторной попыткой
                    time.sleep(3600)
        
        # Создание и запуск потока
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_local_logs(self):
        """
        Очистка локальных логов: сначала пытаемся отправить неотправленные логи, затем удаляем старые
        """
        try:
            # Сначала пытаемся отправить неотправленные логи
            self._retry_failed_uploads()
            
            # Проверка размера директории с логами
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.local_logs_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            # Конвертация в МБ
            total_size_mb = total_size / (1024 * 1024)
            
            # Если размер превышает максимальный, удаляем старые файлы
            if total_size_mb > self.max_local_logs_size_mb:
                logger.info(f"Размер локальных логов ({total_size_mb:.2f} МБ) превышает максимальный ({self.max_local_logs_size_mb} МБ). Начинаем очистку...")
                
                # Получаем список всех файлов с их временем создания
                files_with_time = []
                for dirpath, dirnames, filenames in os.walk(self.local_logs_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        creation_time = os.path.getctime(fp)
                        files_with_time.append((fp, creation_time))
                
                # Сортируем файлы по времени создания (от старых к новым)
                files_with_time.sort(key=lambda x: x[1])
                
                # Удаляем старые файлы, пока размер не станет меньше 80% от максимального
                target_size_mb = self.max_local_logs_size_mb * 0.8
                current_size_mb = total_size_mb
                
                for file_path, _ in files_with_time:
                    if current_size_mb <= target_size_mb:
                        break
                    
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    try:
                        os.remove(file_path)
                        current_size_mb -= file_size_mb
                        logger.info(f"Удален старый лог: {file_path} ({file_size_mb:.2f} МБ)")
                    except Exception as e:
                        logger.error(f"Ошибка при удалении файла {file_path}: {e}")
                
                logger.info(f"Очистка завершена. Новый размер: {current_size_mb:.2f} МБ")
        except Exception as e:
            logger.error(f"Ошибка при очистке локальных логов: {e}")
    
    def _retry_failed_uploads(self):
        """
        Повторная попытка отправки неотправленных логов в S3
        """
        try:
            logger.info("Попытка повторной отправки неотправленных логов...")
            
            # Получаем список всех локальных логов
            for dirpath, dirnames, filenames in os.walk(self.local_logs_dir):
                for f in filenames:
                    file_path = os.path.join(dirpath, f)
                    
                    # Если это JSON файл лога
                    if f.endswith('.json'):
                        try:
                            # Читаем лог
                            with open(file_path, 'rb') as log_file:
                                log_bytes = log_file.read()
                            
                            # Формируем имя объекта в S3 на основе структуры директорий
                            # local_log_path имеет вид: logs/2025-06-04/image_upload/user_id_uuid.json
                            relative_path = os.path.relpath(file_path, self.local_logs_dir)
                            s3_object_name = f"logs/{relative_path}"
                            
                            # Пытаемся отправить в S3
                            success, _ = self.logs_s3.upload_bytes(
                                file_bytes=log_bytes,
                                object_name=s3_object_name
                            )
                            
                            if success:
                                logger.info(f"Лог успешно отправлен в S3: {file_path}")
                                # Удаляем локальный файл после успешной отправки
                                try:
                                    os.remove(file_path)
                                    logger.debug(f"Локальный лог удален: {file_path}")
                                except Exception as e:
                                    logger.warning(f"Не удалось удалить локальный лог {file_path}: {e}")
                            else:
                                logger.debug(f"Не удалось отправить лог в S3: {file_path}")
                        except Exception as e:
                            logger.error(f"Ошибка при повторной отправке лога {file_path}: {e}")
                        
                        # Очищаем пустые папки после обработки каждого файла
                        self._cleanup_empty_directories()
            
            logger.info("Повторная отправка неотправленных логов завершена")
        except Exception as e:
            logger.error(f"Ошибка при повторной отправке неотправленных логов: {e}")
    
    def _cleanup_empty_directories(self):
        """
        Удаление пустых директорий после удаления логов
        """
        try:
            logger.debug("Очистка пустых директорий...")
            
            # Обходим директории снизу вверх (от вложенных к корневым)
            for root, dirs, files in os.walk(self.local_logs_dir, topdown=False):
                # Проверяем, является ли директория пустой
                if not os.listdir(root):
                    try:
                        os.rmdir(root)
                        logger.debug(f"Удалена пустая директория: {root}")
                    except Exception as e:
                        logger.debug(f"Не удалось удалить директорию {root}: {e}")
            
            logger.debug("Очистка пустых директорий завершена")
        except Exception as e:
            logger.error(f"Ошибка при очистке пустых директорий: {e}")
    
    def log_user_action(self, user_id, user_email, user_organization, action_type, action_details=None):
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
            # Формирование лога с использованием UTC для временных меток
            log_entry = {
                "log_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "user_email": user_email,
                "user_organization": user_organization,
                "action_type": action_type,
                "action_details": action_details or {}
            }
            
            # Формирование имени файла лога по лучшим практикам
            log_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_uuid = str(uuid.uuid4())
            log_filename = f"{log_date}/{action_type}/{user_id}_{log_uuid}.json"
            
            # Полный путь к локальному файлу
            local_log_dir = os.path.join(self.local_logs_dir, log_date, action_type)
            os.makedirs(local_log_dir, exist_ok=True)
            local_log_path = os.path.join(local_log_dir, f"{user_id}_{log_uuid}.json")
            
            # Сохранение лога в JSON локально
            with open(local_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
            
            # Проверка существования файла перед загрузкой
            logger.debug(f"Путь к локальному файлу лога: {local_log_path}, существует: {os.path.exists(local_log_path)}")
            
            # Загрузка лога в Object Storage
            try:
                with open(local_log_path, 'rb') as f:
                    log_bytes = f.read()
                success, _ = self.logs_s3.upload_bytes(
                    file_bytes=log_bytes,
                    object_name=f"logs/{log_filename}"
                )
            except Exception as e:
                logger.error(f"Ошибка чтения локального лога для загрузки в S3: {e}")
                success = False
            
            if success:
                logger.info(f"Лог действия пользователя {user_email} успешно сохранен в S3: {action_type}")
                # Удаляем локальный файл после успешной отправки в S3
                try:
                    os.remove(local_log_path)
                    logger.debug(f"Локальный лог удален после успешной отправки: {local_log_path}")
                    
                    # Очищаем пустые папки после удаления лога
                    self._cleanup_empty_directories()
                except Exception as e:
                    logger.warning(f"Не удалось удалить локальный лог {local_log_path}: {e}")
                return True
            else:
                logger.warning(f"Не удалось сохранить лог действия пользователя {user_email} в Object Storage. Лог сохранен локально: {local_log_path}")
                # Лог остается локально для повторной попытки отправки
                return False
        except Exception as e:
            logger.error(f"Ошибка логирования действия пользователя {user_email}: {e}")
            return False
    
    def log_image_upload(self, user_id, user_email, user_organization, original_image_name, image_data):
        """
        Логирование загрузки изображения и сохранение изображения в S3
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            original_image_name (str): Оригинальное имя изображения
            image_data: Данные изображения (PIL.Image)
            
        Returns:
            tuple: (bool, str) - результат логирования и имя объекта в S3
        """
        try:
            # Генерация уникального имени файла с использованием UUID
            unique_id = str(uuid.uuid4())
            new_image_name = f"img_{unique_id}_{original_image_name}"
            
            # Временное сохранение изображения
            image_data.save(new_image_name)
            
            # Загрузка изображения в Object Storage
            success, s3_object_name = self.images_s3.upload_file(
                file_path=new_image_name,
                object_name=new_image_name
            )
            
            if not success:
                logger.error(f"Не удалось сохранить изображение {original_image_name} в Object Storage")
                return False, None
            
            # Логирование действия
            log_success = self.log_user_action(
                user_id=user_id,
                user_email=user_email,
                user_organization=user_organization,
                action_type="image_upload",
                action_details={
                    "original_image_name": original_image_name,
                    "s3_object_name": s3_object_name,
                    "upload_time": datetime.now(timezone.utc).isoformat()
                }
            )
            
            return log_success, s3_object_name
        except Exception as e:
            logger.error(f"Ошибка при логировании загрузки изображения {original_image_name}: {e}")
            return False, None
    
    def log_model_response(self, user_id, user_email, user_organization, image_name, model_responses, processing_time=None):
        """
        Логирование ответов моделей по изображению
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            image_name (str): Оригинальное имя изображения
            model_responses (dict): Ответы моделей
            processing_time (float, optional): Время обработки изображения в секундах
            
        Returns:
            bool: True, если логирование успешно, иначе False
        """
        return self.log_user_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type="model_response",
            action_details={
                "original_image_name": image_name,
                "model_responses": model_responses,
                "analysis_time": processing_time if processing_time is not None else datetime.now(timezone.utc).isoformat()
            }
        )
    
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
        # Используем метод log_user_action, который уже работает корректно
        return self.log_user_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type=action_type,
            action_details=action_details
        )
    
    def log_feedback(self, user_id, user_email, user_organization, feedback_data, image_name, s3_object_name):
        """
        Логирование обратной связи
        
        Args:
            user_id (str): ID пользователя
            user_email (str): Email пользователя
            user_organization (str): Организация пользователя
            feedback_data (dict): Данные обратной связи
            image_name (str): Оригинальное имя изображения
            s3_object_name (str): Имя объекта в S3
            
        Returns:
            bool: True, если логирование успешно, иначе False
        """
        # Генерация уникального имени файла для JSON с обратной связью
        unique_id = str(uuid.uuid4())
        json_file_name = f"form_data_{unique_id}.json"
        
        # Сохранение JSON с обратной связью локально
        with open(json_file_name, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        # Загрузка JSON с обратной связью в Object Storage
        success, json_s3_object_name = self.images_s3.upload_file(
            file_path=json_file_name,
            object_name=json_file_name
        )
        
        if not success:
            logger.error(f"Не удалось сохранить JSON с обратной связью в Object Storage")
            return False
        
        # Логирование действия
        return self.log_user_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type="submit_feedback",
            action_details={
                "original_image_name": image_name,
                "image_s3_object_name": s3_object_name,
                "feedback_s3_object_name": json_s3_object_name,
                "feedback_data": feedback_data,
                "feedback_time": datetime.now(timezone.utc).isoformat()
            }
        )

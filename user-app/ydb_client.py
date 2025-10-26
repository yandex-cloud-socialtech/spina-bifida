import ydb
import ydb.iam
import os
from datetime import datetime
import uuid
import logging
import os
import contextlib
import sys

# Подавление предупреждений gRPC о fork()
os.environ['GRPC_POLL_STRATEGY'] = 'poll'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YDBClient:
    """
    Класс для работы с YDB (Yandex Database)
    """
    def __init__(self, endpoint, database, credentials):
        """
        Инициализация клиента YDB
        
        Args:
            endpoint (str): Endpoint для подключения к YDB
            database (str): Путь к базе данных YDB
            credentials: Учетные данные для подключения к YDB
        """
        self.endpoint = endpoint
        self.database = database
        self.credentials = credentials
        self.driver = None
        self.connect()
        
    def connect(self):
        """
        Подключение к YDB
        
        Returns:
            bool: True, если подключение успешно, иначе False
        """
        try:
            # Подавляем stderr во время инициализации YDB для устранения предупреждений gRPC
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                driver_config = ydb.DriverConfig(
                    endpoint=self.endpoint,
                    database=self.database,
                    credentials=self.credentials
                )
                self.driver = ydb.Driver(driver_config)
                self.driver.wait(timeout=5)
            logger.info("Успешное подключение к YDB")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к YDB: {e}")
            return False
    
    def close(self):
        """
        Закрытие соединения с YDB
        
        Returns:
            bool: True, если соединение закрыто успешно, иначе False
        """
        try:
            if self.driver:
                self.driver.close()
                logger.info("Соединение с YDB успешно закрыто")
                return True
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с YDB: {e}")
            return False
    
    def create_tables(self):
        """
        Создание таблиц в базе данных YDB
        
        Returns:
            bool: True, если таблицы созданы успешно, иначе False
        """
        try:
            session = self.driver.table_client.session().create()
            
            # Проверяем, существует ли таблица users, пытаясь выполнить простой запрос
            try:
                test_query = """
                    PRAGMA TablePathPrefix("{}");
                    
                    SELECT COUNT(*) FROM users LIMIT 1;
                """.format(self.database)
                
                session.transaction().execute(test_query, commit_tx=True)
                logger.info("Таблица users уже существует")
                return True
            except Exception as e:
                # Если произошла ошибка, значит таблица не существует
                logger.info("Таблица users не существует, создаем...")
            
            # YDB использует другой синтаксис для создания таблиц
            query = """
                PRAGMA TablePathPrefix("{}");
                
                CREATE TABLE users (
                    id String,
                    email String,
                    password_hash String,
                    organization String,
                    role String,
                    status String,
                    created_at Timestamp,
                    last_login Timestamp,
                    PRIMARY KEY (id)
                );
            """.format(self.database)
            
            try:
                session.execute_scheme(query)
                logger.info("Таблица users успешно создана")
            except Exception as e:
                # Если произошла ошибка при создании таблицы, проверяем, не существует ли она уже
                if "Table name conflict" in str(e):
                    logger.info("Таблица users уже существует")
                else:
                    raise e
            
            # Создаем индекс по email для быстрого поиска
            try:
                index_query = """
                    PRAGMA TablePathPrefix("{}");
                    
                    ALTER TABLE users ADD INDEX users_email_idx GLOBAL ON (email);
                """.format(self.database)
                
                session.execute_scheme(index_query)
                logger.info("Индекс users_email_idx успешно создан")
            except Exception as e:
                # Если произошла ошибка при создании индекса, проверяем, не существует ли он уже
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in [
                    "index already exists", 
                    "already has an index", 
                    "path exist", 
                    "already exists"
                ]):
                    logger.info("Индекс users_email_idx уже существует")
                else:
                    logger.warning(f"Ошибка при создании индекса: {e}")
            
            logger.info("Таблицы и индексы успешно проверены/созданы")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания таблиц: {e}")
            return False
    
    def add_user(self, email, password_hash, organization, role='user', status='active'):
        """
        Добавление нового пользователя в базу данных
        
        Args:
            email (str): Email пользователя
            password_hash (str): Хеш пароля
            organization (str): Организация пользователя
            role (str, optional): Роль пользователя. По умолчанию 'user'
            status (str, optional): Статус пользователя. По умолчанию 'pending'
            
        Returns:
            str: ID созданного пользователя или None в случае ошибки
        """
        try:
            session = self.driver.table_client.session().create()
            
            # Проверяем, существует ли пользователь с таким email
            query = f"""
                SELECT id FROM users WHERE email = "{email}"
            """
            result_sets = session.transaction().execute(query, commit_tx=True)
            
            if result_sets[0].rows:
                logger.warning(f"Пользователь с email {email} уже существует")
                return None
            
            # Создаем нового пользователя
            user_id = str(uuid.uuid4())
            now = datetime.now()
            
            # Форматируем дату/время в формате, который поддерживает YDB
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Логирование для отладки
            logger.info(f"Добавление пользователя: email={email}, role={role}, status={status}")
            
            query = f"""
                UPSERT INTO users (id, email, password_hash, organization, role, status, created_at, last_login)
                VALUES ("{user_id}", "{email}", "{password_hash}", "{organization}", "{role}", "{status}", Timestamp("{timestamp}"), Timestamp("{timestamp}"))
            """
            session.transaction().execute(query, commit_tx=True)
            
            logger.info(f"Пользователь {email} успешно добавлен")
            return user_id
        except Exception as e:
            logger.error(f"Ошибка добавления пользователя: {e}")
            return None
    
    def get_user_by_email(self, email):
        """
        Получение пользователя по email
        
        Args:
            email (str): Email пользователя
            
        Returns:
            dict: Данные пользователя или None, если пользователь не найден
        """
        try:
            session = self.driver.table_client.session().create()
            
            query = f"""
                SELECT id, email, password_hash, organization, role, status, created_at, last_login
                FROM users WHERE email = "{email}"
            """
            result_sets = session.transaction().execute(query, commit_tx=True)
            
            if not result_sets[0].rows:
                logger.warning(f"Пользователь с email {email} не найден")
                return None
            
            row = result_sets[0].rows[0]
            user = {
                'id': row.id,
                'email': row.email,
                'password_hash': row.password_hash,
                'organization': row.organization,
                'role': row.role,
                'status': row.status,
                'created_at': row.created_at,
                'last_login': row.last_login
            }
            
            return user
        except Exception as e:
            logger.error(f"Ошибка получения пользователя: {e}")
            return None
    
    def update_user_status(self, user_id, status):
        """
        Обновление статуса пользователя
        
        Args:
            user_id (str): ID пользователя
            status (str): Новый статус пользователя
            
        Returns:
            bool: True, если статус обновлен успешно, иначе False
        """
        try:
            session = self.driver.table_client.session().create()
            
            # Логирование для отладки
            logger.info(f"Обновление статуса пользователя: id={user_id}, новый статус={status}")
            
            # Сначала получим текущий статус пользователя для отладки
            debug_query = f"""
                SELECT id, email, status
                FROM users WHERE id = "{user_id}"
            """
            debug_result_sets = session.transaction().execute(debug_query, commit_tx=True)
            
            if debug_result_sets[0].rows:
                row = debug_result_sets[0].rows[0]
                current_status = row.status
                if isinstance(current_status, bytes):
                    current_status = current_status.decode('utf-8')
                logger.info(f"Текущий статус пользователя {user_id}: {current_status}")
            
            # Обновляем статус пользователя
            query = f"""
                UPDATE users SET status = "{status}" WHERE id = "{user_id}"
            """
            session.transaction().execute(query, commit_tx=True)
            
            # Проверяем, что статус обновился
            verify_query = f"""
                SELECT id, email, status
                FROM users WHERE id = "{user_id}"
            """
            verify_result_sets = session.transaction().execute(verify_query, commit_tx=True)
            
            if verify_result_sets[0].rows:
                row = verify_result_sets[0].rows[0]
                new_status = row.status
                if isinstance(new_status, bytes):
                    new_status = new_status.decode('utf-8')
                logger.info(f"Новый статус пользователя {user_id}: {new_status}")
            
            logger.info(f"Статус пользователя {user_id} обновлен на {status}")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления статуса пользователя: {e}")
            return False
    
    def update_user_role(self, user_id, role):
        """
        Обновление роли пользователя
        
        Args:
            user_id (str): ID пользователя
            role (str): Новая роль пользователя
            
        Returns:
            bool: True, если роль обновлена успешно, иначе False
        """
        try:
            session = self.driver.table_client.session().create()
            
            query = f"""
                UPDATE users SET role = "{role}" WHERE id = "{user_id}"
            """
            session.transaction().execute(query, commit_tx=True)
            
            logger.info(f"Роль пользователя {user_id} обновлена на {role}")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления роли пользователя: {e}")
            return False
    
    def get_pending_users(self):
        """
        Получение списка пользователей, ожидающих подтверждения
        
        Returns:
            list: Список пользователей, ожидающих подтверждения
        """
        try:
            session = self.driver.table_client.session().create()
            
            # Добавляем логирование для отладки
            logger.info("Получение списка пользователей, ожидающих подтверждения")
            
            # Сначала получим всех пользователей для отладки
            debug_query = """
                SELECT id, email, organization, status, created_at
                FROM users
            """
            debug_result_sets = session.transaction().execute(debug_query, commit_tx=True)
            
            # Логируем всех пользователей для отладки
            for row in debug_result_sets[0].rows:
                status = row.status
                if isinstance(status, bytes):
                    status = status.decode('utf-8')
                logger.info(f"Пользователь в БД: id={row.id}, email={row.email}, status={status}")
            
            # Теперь выполняем запрос для получения пользователей со статусом pending
            # Используем CAST для преобразования байтовых строк в обычные строки
            query = """
                SELECT id, email, organization, created_at
                FROM users 
                WHERE CAST(status as String) = "pending" 
                   OR CAST(status as String) = 'pending'
                   OR status = "pending" 
                   OR status = 'pending'
            """
            result_sets = session.transaction().execute(query, commit_tx=True)
            
            users = []
            for row in result_sets[0].rows:
                # Преобразование created_at в строку для безопасного отображения
                created_at = row.created_at
                try:
                    if isinstance(created_at, int) or isinstance(created_at, float):
                        # Логирование для отладки
                        logger.info(f"Преобразование timestamp: {created_at}, тип: {type(created_at)}")
                        
                        # Если это timestamp в секундах (обычный формат)
                        if created_at < 10000000000:  # Примерно до 2286 года
                            from datetime import datetime
                            created_at = datetime.fromtimestamp(created_at)
                        # Если это timestamp в миллисекундах
                        elif created_at < 10000000000000:  # Примерно до 2286 года * 1000
                            from datetime import datetime
                            created_at = datetime.fromtimestamp(created_at / 1000)
                        # Если это timestamp в микросекундах
                        elif created_at < 10000000000000000:  # Примерно до 2286 года * 1000000
                            from datetime import datetime
                            created_at = datetime.fromtimestamp(created_at / 1000000)
                        else:
                            # Если это слишком большое число, просто преобразуем в строку
                            created_at = str(created_at)
                    elif isinstance(created_at, datetime):
                        # Если это уже datetime, оставляем как есть
                        pass
                    else:
                        # Для всех остальных типов преобразуем в строку
                        created_at = str(created_at)
                except Exception as e:
                    logger.warning(f"Ошибка преобразования даты: {e}, тип: {type(created_at)}, значение: {created_at}")
                    created_at = str(created_at)
                
                user = {
                    'id': row.id,
                    'email': row.email,
                    'organization': row.organization,
                    'created_at': created_at
                }
                users.append(user)
                
                # Логирование для отладки
                logger.info(f"Получен пользователь, ожидающий подтверждения: id={row.id}, email={row.email}, тип created_at={type(created_at)}")
            
            return users
        except Exception as e:
            logger.error(f"Ошибка получения списка пользователей: {e}")
            return []
    
    def update_last_login(self, user_id):
        """
        Обновление времени последнего входа пользователя
        
        Args:
            user_id (str): ID пользователя
            
        Returns:
            bool: True, если время обновлено успешно, иначе False
        """
        try:
            session = self.driver.table_client.session().create()
            
            now = datetime.now()
            # Форматируем дату/время в формате, который поддерживает YDB
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            query = f"""
                UPDATE users SET last_login = Timestamp("{timestamp}") WHERE id = "{user_id}"
            """
            session.transaction().execute(query, commit_tx=True)
            
            logger.info(f"Время последнего входа пользователя {user_id} обновлено")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления времени последнего входа: {e}")
            return False

# Функция для инициализации клиента YDB
def init_ydb_client():
    """
    Инициализация клиента YDB из переменных окружения
    
    Returns:
        YDBClient: Инициализированный клиент YDB или None в случае ошибки
    """
    try:
        endpoint = os.getenv('YDB_ENDPOINT')
        database = os.getenv('YDB_DATABASE')
        service_account_key_file = os.getenv('YDB_SERVICE_ACCOUNT_KEY_FILE')
        
        if not endpoint or not database:
            logger.error("Не указаны обязательные переменные окружения YDB_ENDPOINT и YDB_DATABASE")
            return None
        
        if service_account_key_file:
            credentials = ydb.iam.ServiceAccountCredentials.from_file(service_account_key_file)
        else:
            credentials = ydb.AnonymousCredentials()
        
        client = YDBClient(endpoint, database, credentials)
        client.create_tables()
        
        return client
    except Exception as e:
        logger.error(f"Ошибка инициализации клиента YDB: {e}")
        return None

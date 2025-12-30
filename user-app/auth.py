import bcrypt
import streamlit as st
import re
import logging
import os
from datetime import datetime, timezone

# Импорт логгера из единого экземпляра
from logger_instance import user_logger

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hash_password(password):
    """
    Хеширование пароля с использованием bcrypt
    
    Args:
        password (str): Пароль в открытом виде
        
    Returns:
        str: Хешированный пароль
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def check_password(password, password_hash):
    """
    Проверка пароля
    
    Args:
        password (str): Пароль в открытом виде
        password_hash (str): Хешированный пароль
        
    Returns:
        bool: True, если пароль верный, иначе False
    """
    try:
        password_bytes = password.encode('utf-8')
        
        # Проверяем, является ли password_hash строкой или байтами
        if isinstance(password_hash, str):
            hash_bytes = password_hash.encode('utf-8')
        else:
            hash_bytes = password_hash
            
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception as e:
        logger.error(f"Ошибка проверки пароля: {e}")
        return False

def validate_email(email):
    """
    Валидация email
    
    Args:
        email (str): Email для проверки
        
    Returns:
        bool: True, если email валидный, иначе False
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """
    Валидация пароля
    
    Args:
        password (str): Пароль для проверки
        
    Returns:
        tuple: (bool, str) - результат проверки и сообщение об ошибке
    """
    if len(password) < 8:
        return False, "Пароль должен содержать не менее 8 символов"
    
    if not re.search(r'[A-Z]', password):
        return False, "Пароль должен содержать хотя бы одну заглавную букву"
    
    if not re.search(r'[a-z]', password):
        return False, "Пароль должен содержать хотя бы одну строчную букву"
    
    if not re.search(r'[0-9]', password):
        return False, "Пароль должен содержать хотя бы одну цифру"
    
    return True, ""

def register_user(ydb_client, email, password, organization):
    """
    Регистрация нового пользователя
    
    Args:
        ydb_client: Клиент YDB
        email (str): Email пользователя
        password (str): Пароль пользователя
        organization (str): Организация пользователя
        
    Returns:
        tuple: (bool, str) - результат регистрации и сообщение
    """
    try:
        # Валидация email
        if not validate_email(email):
            return False, "Некорректный формат email"
        
        # Валидация пароля
        is_valid, message = validate_password(password)
        if not is_valid:
            return False, message
        
        # Проверка, существует ли пользователь с таким email
        existing_user = ydb_client.get_user_by_email(email)
        if existing_user:
            return False, "Пользователь с таким email уже существует"
        
        # Хеширование пароля
        password_hash = hash_password(password)
        
        # Добавление пользователя в базу данных
        user_id = ydb_client.add_user(email, password_hash, organization)
        
        if user_id:
            logger.info(f"Пользователь {email} успешно зарегистрирован")
            
            # Логирование действия регистрации
            user_logger.log_user_action(
                user_id=user_id,
                user_email=email,
                user_organization=organization,
                action_type="register",
                action_details={
                    "registration_time": datetime.now(timezone.utc).isoformat(),
                    "status": "active"
                }
            )
            
            return True, "Регистрация успешна. Вы можете войти в систему."
        else:
            return False, "Ошибка при регистрации пользователя"
    except Exception as e:
        logger.error(f"Ошибка регистрации пользователя: {e}")
        return False, f"Произошла ошибка: {str(e)}"

def login_user(ydb_client, email, password):
    """
    Авторизация пользователя
    
    Args:
        ydb_client: Клиент YDB
        email (str): Email пользователя
        password (str): Пароль пользователя
        
    Returns:
        tuple: (bool, str) - результат авторизации и сообщение
    """
    try:
        # Получение пользователя по email
        user = ydb_client.get_user_by_email(email)
        
        if not user:
            return False, "Пользователь с таким email не найден"
        
        # Логирование информации о пользователе для отладки
        logger.info(f"Данные пользователя {email}: id={user['id']}, role={user['role']}, status={user['status']}, тип status={type(user['status'])}")
        
        # Приведение статуса к строке и удаление лишних пробелов
        status = user['status']
        
        # Если статус является байтовой строкой, декодируем ее
        if isinstance(status, bytes):
            status = status.decode('utf-8')
            logger.info(f"Статус декодирован из байтов: {status}")
        
        # Приведение к строке и удаление лишних пробелов
        status = str(status).strip()
        
        # Проверка статуса пользователя
        if status == 'rejected':
            logger.warning(f"Попытка входа с отклоненной учетной записью: {email}")
            return False, "Ваша учетная запись была отклонена администратором"
        
        # Проверка пароля
        if not check_password(password, user['password_hash']):
            return False, "Неверный пароль"
        
        # Обновление времени последнего входа
        ydb_client.update_last_login(user['id'])
        
        # Подготовка данных пользователя для сохранения в сессии
        user_id = user['id']
        if isinstance(user_id, bytes):
            user_id = user_id.decode('utf-8')
            
        email = user['email']
        if isinstance(email, bytes):
            email = email.decode('utf-8')
            
        organization = user['organization']
        if isinstance(organization, bytes):
            organization = organization.decode('utf-8')
            
        role = user['role']
        if isinstance(role, bytes):
            role = role.decode('utf-8')
        
        # Сохранение информации о пользователе в сессии
        st.session_state['user'] = {
            'id': user_id,
            'email': email,
            'organization': organization,
            'role': role
        }
        
        # Логирование информации о пользователе
        logger.info(f"Пользователь {email} успешно авторизован с ролью: {user['role']}")
        
        # Логирование действия авторизации
        user_logger.log_auth_action(
            user_id=user_id,
            user_email=email,
            user_organization=organization,
            action_type="login",
            action_details={
                "login_time": datetime.now(timezone.utc).isoformat(),
                "role": role
            }
        )
        
        logger.info(f"Пользователь {email} успешно авторизован")
        return True, "Авторизация успешна"
    except Exception as e:
        logger.error(f"Ошибка авторизации пользователя: {e}")
        return False, f"Произошла ошибка: {str(e)}"

def logout_user():
    """
    Выход пользователя из системы
    """
    if 'user' in st.session_state:
        user = st.session_state['user']
        user_id = user['id']
        user_email = user['email']
        user_organization = user['organization']
        
        # Логирование действия выхода из системы
        user_logger.log_auth_action(
            user_id=user_id,
            user_email=user_email,
            user_organization=user_organization,
            action_type="logout",
            action_details={
                "logout_time": datetime.now(timezone.utc).isoformat()
            }
        )
        
        del st.session_state['user']
        logger.info(f"Пользователь {user_email} вышел из системы")

def is_authenticated():
    """
    Проверка, авторизован ли пользователь
    
    Returns:
        bool: True, если пользователь авторизован, иначе False
    """
    return 'user' in st.session_state

def is_admin():
    """
    Проверка, является ли пользователь администратором
    
    Returns:
        bool: True, если пользователь является администратором, иначе False
    """
    if not is_authenticated():
        logger.warning("Попытка проверки роли администратора для неавторизованного пользователя")
        return False
    
    user = st.session_state['user']
    
    # Логирование информации о пользователе для отладки
    logger.info(f"Проверка роли администратора для пользователя {user['email']}: role={user['role']}, тип role={type(user['role'])}")
    
    # Получаем роль пользователя
    role = user['role']
    
    # Если роль является байтовой строкой, декодируем ее
    if isinstance(role, bytes):
        role = role.decode('utf-8')
        logger.info(f"Роль декодирована из байтов: {role}")
    
    # Приведение к строке и удаление лишних пробелов
    role = str(role).strip()
    is_admin_role = role == 'admin'
    
    if is_admin_role:
        logger.info(f"Пользователь {user['email']} имеет роль администратора")
    else:
        logger.warning(f"Пользователь {user['email']} не имеет роли администратора, текущая роль: '{role}'")
    
    return is_admin_role

def approve_user(ydb_client, user_id):
    """
    Подтверждение регистрации пользователя администратором
    
    Args:
        ydb_client: Клиент YDB
        user_id (str): ID пользователя
        
    Returns:
        bool: True, если подтверждение успешно, иначе False
    """
    try:
        # Декодирование ID пользователя, если это байтовая строка
        if isinstance(user_id, bytes):
            user_id = user_id.decode('utf-8')
        
        # Логирование для отладки
        logger.info(f"Подтверждение регистрации пользователя: id={user_id}")
        
        # Получение информации о пользователе
        user_info = None
        for user in ydb_client.get_pending_users():
            if user['id'] == user_id or (isinstance(user['id'], bytes) and user['id'].decode('utf-8') == user_id):
                user_info = user
                break
        
        result = ydb_client.update_user_status(user_id, 'active')
        if result:
            logger.info(f"Регистрация пользователя {user_id} подтверждена")
            
            # Если пользователь авторизован и является администратором
            if is_authenticated() and is_admin():
                admin = st.session_state['user']
                
                # Логирование действия подтверждения регистрации
                user_logger.log_user_action(
                    user_id=admin['id'],
                    user_email=admin['email'],
                    user_organization=admin['organization'],
                    action_type="approve_user",
                    action_details={
                        "approved_user_id": user_id,
                        "approved_user_email": user_info['email'] if user_info else "unknown",
                        "approval_time": datetime.now(timezone.utc).isoformat()
                    }
                )
        else:
            logger.error(f"Не удалось подтвердить регистрацию пользователя {user_id}")
        return result
    except Exception as e:
        logger.error(f"Ошибка подтверждения регистрации: {e}")
        return False

def reject_user(ydb_client, user_id):
    """
    Отклонение регистрации пользователя администратором
    
    Args:
        ydb_client: Клиент YDB
        user_id (str): ID пользователя
        
    Returns:
        bool: True, если отклонение успешно, иначе False
    """
    try:
        # Декодирование ID пользователя, если это байтовая строка
        if isinstance(user_id, bytes):
            user_id = user_id.decode('utf-8')
        
        # Логирование для отладки
        logger.info(f"Отклонение регистрации пользователя: id={user_id}")
        
        # Получение информации о пользователе
        user_info = None
        for user in ydb_client.get_pending_users():
            if user['id'] == user_id or (isinstance(user['id'], bytes) and user['id'].decode('utf-8') == user_id):
                user_info = user
                break
        
        result = ydb_client.update_user_status(user_id, 'rejected')
        if result:
            logger.info(f"Регистрация пользователя {user_id} отклонена")
            
            # Если пользователь авторизован и является администратором
            if is_authenticated() and is_admin():
                admin = st.session_state['user']
                
                # Логирование действия отклонения регистрации
                user_logger.log_user_action(
                    user_id=admin['id'],
                    user_email=admin['email'],
                    user_organization=admin['organization'],
                    action_type="reject_user",
                    action_details={
                        "rejected_user_id": user_id,
                        "rejected_user_email": user_info['email'] if user_info else "unknown",
                        "rejection_time": datetime.now(timezone.utc).isoformat()
                    }
                )
        else:
            logger.error(f"Не удалось отклонить регистрацию пользователя {user_id}")
        return result
    except Exception as e:
        logger.error(f"Ошибка отклонения регистрации: {e}")
        return False

def create_admin_if_not_exists(ydb_client, admin_email, admin_password, admin_organization):
    """
    Создание администратора, если он не существует
    
    Args:
        ydb_client: Клиент YDB
        admin_email (str): Email администратора
        admin_password (str): Пароль администратора
        admin_organization (str): Организация администратора
        
    Returns:
        bool: True, если администратор создан или уже существует, иначе False
    """
    try:
        # Проверка, существует ли администратор
        admin = ydb_client.get_user_by_email(admin_email)
        
        if admin:
            # Получаем роль пользователя
            role = admin['role']
            
            # Если роль является байтовой строкой, декодируем ее
            if isinstance(role, bytes):
                role = role.decode('utf-8')
                logger.info(f"Роль декодирована из байтов: {role}")
            
            # Приведение к строке и удаление лишних пробелов
            role = str(role).strip()
            
            # Получаем статус пользователя
            status = admin['status']
            
            # Если статус является байтовой строкой, декодируем его
            if isinstance(status, bytes):
                status = status.decode('utf-8')
                logger.info(f"Статус декодирован из байтов: {status}")
            
            # Приведение к строке и удаление лишних пробелов
            status = str(status).strip()
            
            # Проверяем, является ли пользователь администратором
            if role == 'admin':
                logger.info(f"Администратор {admin_email} уже существует")
                
                # Проверяем статус администратора
                if status != 'active':
                    # Активируем администратора, если он не активен
                    ydb_client.update_user_status(admin['id'], 'active')
                    logger.info(f"Статус администратора {admin_email} обновлен на 'active'")
                
                return True
            else:
                # Пользователь существует, но не является администратором
                logger.warning(f"Пользователь {admin_email} существует, но не является администратором. Текущая роль: {admin['role']}")
                
                # Обновляем роль пользователя на 'admin'
                if ydb_client.update_user_role(admin['id'], 'admin'):
                    logger.info(f"Роль пользователя {admin_email} обновлена на 'admin'")
                    
                    # Проверяем статус пользователя
                    if admin['status'] != 'active':
                        # Активируем пользователя, если он не активен
                        ydb_client.update_user_status(admin['id'], 'active')
                        logger.info(f"Статус пользователя {admin_email} обновлен на 'active'")
                    
                    return True
                else:
                    logger.error(f"Не удалось обновить роль пользователя {admin_email} на 'admin'")
                    return False
        
        # Хеширование пароля
        password_hash = hash_password(admin_password)
        
        # Добавление администратора в базу данных
        user_id = ydb_client.add_user(admin_email, password_hash, admin_organization, role='admin', status='active')
        
        if user_id:
            logger.info(f"Администратор {admin_email} успешно создан")
            
            # Логирование создания администратора
            user_logger.log_user_action(
                user_id=user_id,
                user_email=admin_email,
                user_organization=admin_organization,
                action_type="create_admin",
                action_details={
                    "creation_time": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    "role": "admin"
                }
            )
            
            return True
        else:
            logger.error(f"Не удалось создать администратора {admin_email}")
            return False
    except Exception as e:
        logger.error(f"Ошибка создания администратора: {e}")
        return False

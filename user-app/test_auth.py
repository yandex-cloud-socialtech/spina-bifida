import os
import logging
from dotenv import load_dotenv
from ydb_client import init_ydb_client
from auth import create_admin_if_not_exists, hash_password, check_password, validate_email, validate_password

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ydb_connection():
    """
    Тестирование подключения к YDB
    """
    logger.info("Тестирование подключения к YDB...")
    
    # Загрузка переменных окружения
    load_dotenv()
    
    # Инициализация YDB клиента
    ydb_client = init_ydb_client()
    
    if ydb_client:
        logger.info("Подключение к YDB успешно установлено")
        return True
    else:
        logger.error("Не удалось подключиться к YDB")
        return False

def test_admin_creation():
    """
    Тестирование создания администратора
    """
    logger.info("Тестирование создания администратора...")
    
    # Загрузка переменных окружения
    load_dotenv()
    
    # Инициализация YDB клиента
    ydb_client = init_ydb_client()
    
    if not ydb_client:
        logger.error("Не удалось подключиться к YDB")
        return False
    
    # Получение данных администратора из переменных окружения
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')
    admin_organization = os.getenv('ADMIN_ORGANIZATION')
    
    if not admin_email or not admin_password or not admin_organization:
        logger.error("Не указаны данные администратора в .env файле")
        return False
    
    # Создание администратора
    result = create_admin_if_not_exists(ydb_client, admin_email, admin_password, admin_organization)
    
    if result:
        logger.info("Администратор успешно создан или уже существует")
        return True
    else:
        logger.error("Не удалось создать администратора")
        return False

def test_password_hashing():
    """
    Тестирование хеширования и проверки пароля
    """
    logger.info("Тестирование хеширования и проверки пароля...")
    
    # Тестовый пароль
    password = "TestPassword123"
    
    # Хеширование пароля
    password_hash = hash_password(password)
    
    # Проверка хеша
    if check_password(password, password_hash):
        logger.info("Хеширование и проверка пароля работают корректно")
        return True
    else:
        logger.error("Ошибка при хешировании или проверке пароля")
        return False

def test_email_validation():
    """
    Тестирование валидации email
    """
    logger.info("Тестирование валидации email...")
    
    # Тестовые email адреса
    valid_emails = [
        "test@example.com",
        "user.name@domain.com",
        "user-name@domain.co.uk",
        "user123@domain.org"
    ]
    
    invalid_emails = [
        "test",
        "test@",
        "@domain.com",
        "test@domain",
        "test@domain."
    ]
    
    # Проверка валидных email
    for email in valid_emails:
        if not validate_email(email):
            logger.error(f"Валидный email {email} не прошел валидацию")
            return False
    
    # Проверка невалидных email
    for email in invalid_emails:
        if validate_email(email):
            logger.error(f"Невалидный email {email} прошел валидацию")
            return False
    
    logger.info("Валидация email работает корректно")
    return True

def test_password_validation():
    """
    Тестирование валидации пароля
    """
    logger.info("Тестирование валидации пароля...")
    
    # Тестовые пароли
    valid_passwords = [
        "Password123",
        "StrongP@ssw0rd",
        "Abcdef123456"
    ]
    
    invalid_passwords = [
        "password",  # Нет заглавной буквы
        "PASSWORD",  # Нет строчной буквы
        "Password",  # Нет цифры
        "Pass1",     # Слишком короткий
        "12345678"   # Нет букв
    ]
    
    # Проверка валидных паролей
    for password in valid_passwords:
        is_valid, message = validate_password(password)
        if not is_valid:
            logger.error(f"Валидный пароль {password} не прошел валидацию: {message}")
            return False
    
    # Проверка невалидных паролей
    for password in invalid_passwords:
        is_valid, message = validate_password(password)
        if is_valid:
            logger.error(f"Невалидный пароль {password} прошел валидацию")
            return False
    
    logger.info("Валидация пароля работает корректно")
    return True

def run_tests():
    """
    Запуск всех тестов
    """
    logger.info("Запуск тестов...")
    
    tests = [
        test_ydb_connection,
        test_admin_creation,
        test_password_hashing,
        test_email_validation,
        test_password_validation
    ]
    
    results = []
    
    for test in tests:
        result = test()
        results.append(result)
    
    # Вывод результатов
    logger.info("Результаты тестов:")
    for i, test in enumerate(tests):
        status = "УСПЕШНО" if results[i] else "ОШИБКА"
        logger.info(f"{test.__name__}: {status}")
    
    # Общий результат
    if all(results):
        logger.info("Все тесты пройдены успешно")
    else:
        logger.error("Некоторые тесты завершились с ошибками")

if __name__ == "__main__":
    run_tests()

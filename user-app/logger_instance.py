import os
import logging
from dotenv import load_dotenv
from user_logger import UserLogger

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Проверка наличия необходимых переменных окружения
if not os.getenv('USER_LOGS_BUCKET'):
    logger.warning("Переменная окружения USER_LOGS_BUCKET не задана")

if not os.getenv('BUCKET'):
    logger.warning("Переменная окружения BUCKET не задана")

# Инициализация логгера пользовательских действий
user_logger = UserLogger(
    logs_bucket=os.getenv('USER_LOGS_BUCKET'),
    images_bucket=os.getenv('BUCKET'),
    local_logs_dir=os.getenv('LOCAL_LOGS_DIR', './logs'),
    max_local_logs_size_mb=int(os.getenv('MAX_LOCAL_LOGS_SIZE_MB', '500')),
    cleanup_interval_hours=int(os.getenv('CLEANUP_INTERVAL_HOURS', '24'))
)

logger.info(f"Логгер инициализирован с параметрами: logs_bucket={os.getenv('USER_LOGS_BUCKET')}, images_bucket={os.getenv('BUCKET')}")

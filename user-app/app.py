import streamlit as st
from streamlit_image_select import image_select

# Подавление предупреждений gRPC о fork() (должно быть до импорта YDB)
import os
import sys
import warnings

# Подавляем предупреждения gRPC
os.environ['GRPC_POLL_STRATEGY'] = 'poll'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '1'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Подавляем предупреждения Python
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from PIL import Image
import torch
from stqdm import stqdm
import io
import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import uuid
import json
from datetime import datetime, timezone
import gettext
import logging
import signal
import atexit
# BugFix SSL error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Локализация (должна быть до импорта utils, так как utils использует _)
language = 'ru'  # Устанавливаем русский язык по умолчанию
try:
    localizator = gettext.translation('base', localedir='locales', languages=[language])
    localizator.install()
    _ = localizator.gettext 
except:
    # Безопасный fallback - функция, которая возвращает исходную строку
    def _(s):
        return s

# Импорт utils (после определения _)
from utils import *

# Импорт модулей аутентификации
from ydb_client import init_ydb_client
from auth import create_admin_if_not_exists, is_authenticated
from auth_ui import auth_ui

# Импорт логгера из единого экземпляра
from logger_instance import user_logger

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
# Форма обратной связи
form_data = {}

# Загрузка переменных окружения из .env файла
load_dotenv()
BUCKET = os.getenv('BUCKET')
IFRAME = '<iframe src="https://ghbtns.com/github-btn.html?user=yandex-cloud-socialtech&repo=spina-bifida&type=watch&count=true&size=large" frameborder="0" scrolling="0" width="121" height="30" title="GitHub"></iframe>'

# Настройки страницы
st.set_page_config(
     page_title='Spina bifida',
     layout="wide"
)

# Конфигурация версий моделей
MODEL_VERSIONS = {
    "v1": {
        "name": _("Версия 1"),
        "description": _("Базовая версия модели — стабильный и проверенный вариант"),
        "paths": {
            "yolo_model": 'models/v1/best_object_detection.pt',
            "axial_quality": 'models/v1/axial_quality.pt',
            "axial_pathology": 'models/v1/axial_pathology.pt',
            "sagittal_quality": 'models/v1/sagittal_quality.pt',
            "sagittal_pathology": 'models/v1/sagittal_pathology.pt'
        }
    },
    "v2": {
        "name": _("Версия 2"),
        "description": _("Новые модели (бета) — экспериментальная версия с потенциально более высокой точностью."),
        "paths": {
            "yolo_model": 'models/v2/best_object_detection.pt',
            "axial_quality": 'models/v2/axial_quality.pt',
            "axial_pathology": 'models/v2/axial_pathology.pt',
            "sagittal_quality": 'models/v2/sagittal_quality.pt',
            "sagittal_pathology": 'models/v2/sagittal_pathology.pt'
        }
    }
}

torch.classes.__path__ = []
# Инициализация YDB клиента
ydb_client = init_ydb_client()

# Функция для корректного закрытия YDB клиента
def cleanup_ydb_client():
    """Корректное закрытие YDB клиента при завершении приложения"""
    global ydb_client
    if ydb_client:
        ydb_client.close()

# Регистрируем функцию очистки для вызова при завершении приложения
atexit.register(cleanup_ydb_client)

# Создание администратора при первом запуске
if ydb_client:
    admin_email = os.getenv('ADMIN_EMAIL')
    admin_password = os.getenv('ADMIN_PASSWORD')
    admin_organization = os.getenv('ADMIN_ORGANIZATION')
    
    if admin_email and admin_password and admin_organization:
        create_admin_if_not_exists(ydb_client, admin_email, admin_password, admin_organization)


###########
# Функции #
###########

# Функция для отображения модального окна с условиями использования
def show_terms_modal():
    st.write("### Spina Bifida")
    st.markdown(
        f"""
        ### GitHub {IFRAME}
        """,
        unsafe_allow_html=True,
    )
    st.write(_("Пожалуйста, прочитайте и примите наши условия использования данного сервиса, чтобы продолжить."))
    st.markdown("---")
    
    # Новый текст пользовательского соглашения
    st.markdown(_("### 1. Назначение сервиса"))
    st.write(_("Данный веб-сервис представляет собой инструмент на основе искусственного интеллекта, разработанный исключительно для научных, образовательных и исследовательских целей. Он не является зарегистрированным медицинским изделием, не имеет статуса клинического диагностического инструмента и не предназначен для постановки диагноза, мониторинга состояния пациентов или принятия клинических решений."))
    
    st.markdown(_("### 2. Целевая аудитория"))
    st.write(_("Сервис предназначен только для использования квалифицированными специалистами в области ультразвуковой диагностики, перинатологии и смежных медицинских дисциплин, обладающими соответствующим образованием и опытом. Доступ и использование сервиса лицами без медицинского образования запрещены."))
    
    st.markdown(_("### 3. Ограничение ответственности"))
    st.write(_("Авторы сервиса прилагают все разумные усилия для обеспечения точности и актуальности алгоритмов анализа. Однако результаты работы ИИ могут быть ограничены качеством входных данных, особенностями изображений и текущими технологическими возможностями."))
    st.write(_("Результаты анализа, предоставляемые сервисом, не являются медицинским заключением, диагнозом или рекомендацией."))
    st.write(_("Никакая информация из сервиса не должна использоваться как замена консультации с лечащим врачом или другим квалифицированным специалистом в области здравоохранения."))
    
    st.markdown(_("### 4. Обработка данных"))
    st.write(_("• Запрещается загружать в сервис изображения, содержащие персональные данные пациентов (ФИО, дата рождения, номера исследований и т.п.)."))
    st.write(_("• Сохранению подлежат только те изображения и данные, для которых пользователь добровольно предоставил обратную связь (фидбэк), с целью последующего улучшения качества модели ИИ."))
    st.write(_("• Все действия пользователей в системе логируются в целях обеспечения безопасности, аудита и технической поддержки."))
    st.write(_("• Никакие данные не передаются третьим лицам без согласия пользователя, за исключением случаев, предусмотренных законодательством."))
    
    st.markdown(_("### 5. Согласие пользователя"))
    st.write(_("Используя данный сервис, вы подтверждаете, что:"))
    st.write(_("• Ознакомлены с настоящей оговоркой;"))
    st.write(_("• Являетесь медицинским специалистом, имеющим право на работу с медицинскими данными;"))
    st.write(_("• Не используете сервис для принятия клинических решений;"))
    st.write(_("• Согласны с условиями обработки данных, указанными выше."))
    
    st.markdown(_("**❗ Важно:** Результат работы сервиса — это предварительная оценка, требующая интерпретации и подтверждения квалифицированным врачом. За окончательное заключение ответственность несёт лечащий специалист."))
    
    accept = st.button(_("Я прочитал(а) и соглашаюсь с вышеперечисленными условиями"))
    
    # Логирование принятия условий использования
    if accept and is_authenticated():
        user = st.session_state['user']
        user_logger.log_user_action(
            user_id=user['id'],
            user_email=user['email'],
            user_organization=user['organization'],
            action_type="accept_terms",
            action_details={
                "acceptance_time": datetime.now(timezone.utc).isoformat(),
                "language": language
            }
        )

    return accept

# Функция записи обратной связи на S3
def upload_to_yandex_cloud(file_name, bucket, object_name=None):
    ACCESS_KEY = os.getenv('ACCESS_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')
    ENDPOINT_URL = 'https://storage.yandexcloud.net'

    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    try:
        s3.upload_file(file_name, bucket, object_name or file_name)
        st.success(_("Файл {} успешно загружен в Yandex Object Storage.").format(file_name))
        # Удаление локального файла после успешной загрузки
        os.remove(file_name)
        return True, object_name or file_name
    except FileNotFoundError:
        st.error(_("Файл не найден."))
        return False, None
    except NoCredentialsError:
        st.error(_("Ошибка с учетными данными."))
        return False, None
    except Exception as e:
        st.error(_("Произошла ошибка: {}").format(e))
        return False, None

# Функция настройки моделей
@st.cache_resource(show_spinner = "Load model ...")
def get_processor(model_version="v1"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = MODEL_VERSIONS[model_version]
    return MedicalImageProcessor(
        yolo_model_path=model_config["paths"]["yolo_model"],
        axial_quality_model_path=model_config["paths"]["axial_quality"],
        axial_pathology_model_path=model_config["paths"]["axial_pathology"],
        sagittal_quality_model_path=model_config["paths"]["sagittal_quality"],
        sagittal_pathology_model_path=model_config["paths"]["sagittal_pathology"],
        device=device,
        model_version=model_version
    )

# Функции обработки изображений
@st.cache_data(show_spinner = "Image processing ...", ttl = 3600, max_entries = 100)
def cache_process_image(img_bytes, img_name):
    return processor.process_image(img_bytes, img_name)

# Функции обработки изображений
def process_uploaded_files(uploaded_files):
    with stqdm(uploaded_files, mininterval=1) as pbar:
        for uploaded_file in pbar:
            img = Image.open(uploaded_file)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            img_name = uploaded_file.name
            
            if img_name not in st.session_state['imgs']:
                # Логирование загрузки изображения (только при первой обработке)
                if is_authenticated():
                    user = st.session_state['user']
                    user_logger.log_user_action(
                        user_id=user['id'],
                        user_email=user['email'],
                        user_organization=user['organization'],
                        action_type="image_upload",
                        action_details={
                            "upload_time": datetime.now(timezone.utc).isoformat(),
                            "image_name": img_name,
                            "image_size_bytes": len(img_bytes),
                            "file_type": uploaded_file.type
                        }
                    )
                
                # Сохранение изображения в S3
                s3_object_name = None
                if is_authenticated():
                    user = st.session_state['user']
                    # Сохраняем имя объекта в S3 для использования при отправке обратной связи
                    st.session_state['s3_object_names'][img_name] = s3_object_name
                # Измерение времени обработки
                import time
                start_time = time.time()
                result = cache_process_image(img_bytes, img_name)
                processing_time = time.time() - start_time
                
                st.session_state['imgs'][img_name] = img
                st.session_state['processed_images'][img_name] = result
                
                # Логирование ответов моделей
                if is_authenticated():
                    user = st.session_state['user']
                    user_logger.log_model_response(
                        user_id=user['id'],
                        user_email=user['email'],
                        user_organization=user['organization'],
                        image_name=img_name,
                        model_responses={
                            "model_version": model_version,
                            "plane": {
                                "type": result["plane"]["type"],
                                "prediction_prob": float(result["plane"]["prediction_prob"])
                            },
                            "quality": {
                                "prediction_prob": float(result["quality"]["prediction_prob"])
                            },
                            "pathology": {
                                "prediction_prob": float(result["pathology"]["prediction_prob"])
                            }
                        },
                        processing_time=processing_time
                    )
    
def process_example_files(example_files):
    # Логирование действия просмотра примеров
    if is_authenticated():
        user = st.session_state['user']
        user_logger.log_user_action(
            user_id=user['id'],
            user_email=user['email'],
            user_organization=user['organization'],
            action_type="view_examples",
            action_details={
                "example_count": len(example_files),
                "example_names": example_files,
                "view_time": datetime.now(timezone.utc).isoformat()
            }
        )
    
    for example_file in example_files:
        img = Image.open(example_file)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        # Добавляем префикс "example_" к имени файла, чтобы отличить примеры от загруженных изображений
        img_name = f"example_{example_file}"
        if img_name not in st.session_state['imgs']:
            result = cache_process_image(img_bytes, img_name)
            st.session_state['imgs'][img_name] = img
            st.session_state['processed_images'][img_name] = result

# Функция генерации уникального идентификатора файла
def get_unique_id():
    unique_id = str(uuid.uuid4())
    return unique_id


##############
# Приложение #
##############

# Проверка авторизации
if ydb_client:
    is_authorized = auth_ui(ydb_client)
    
    # Если пользователь не авторизован, не показываем основное приложение
    if not is_authorized:
        st.stop()

# Проверяем, если пользователь уже принял условия
if 'accepted' not in st.session_state:
    st.session_state.accepted = False

if not st.session_state.accepted:
    if show_terms_modal():
        st.session_state.accepted = True
        st.rerun()
else:
    # Заголовок приложения
    st.title("Spina Bifida")
    st.markdown(
        f"""
        ### GitHub {IFRAME}
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
        <style>
        .main .block-container {
            max-width: 1200px;
        }
        button[data-baseweb="tab"] {
            font-size: 24px;
            margin: 0;
            width: 100%;
        }
        .stTabs [data-baseweb="tab"] {
            white-space: pre-wrap;
            gap: 0px;
            border-radius: 8px 8px 0px 0px;
            background-color: #262730;
        }   
        .stTabs [aria-selected="true"] {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)

    # Текстовое поле для ввода или отображения текста
    st.markdown(_("Представленный алгоритм на основе ИИ направлен на детектирование спектра патологий центральной нервной системы (в том числе Спина бифида) на эхографических снимках головного мозга плода в первом триместре беременности. Для проведения анализа необходимо загрузить один или несколько ключевых снимков."))
    st.markdown(_("После получения результатов анализа оставьте, пожалуйста, обратную связь."))

    # Выбор версии модели
    model_version = st.selectbox(
        _("На данном этапе мы исследуем несколько версий ИИ-моделей. Выберите версию, которая вам больше подходит:"),
        options=list(MODEL_VERSIONS.keys()),
        format_func=lambda x: f"{MODEL_VERSIONS[x]['name']} - {MODEL_VERSIONS[x]['description']}"
    )

    # Примеры изображений для выбора
    example_images = {
        "Example 1": "example_images/norm-sagittal.jpg",
        "Example 2": "example_images/norm-axial.jpg",
        "Example 3": "example_images/patology-sagittal.jpg",
        "Example 4": "example_images/patology-axial.jpg"
    }   
    
    # Элемент для загрузки файлов
    uploaded_files = st.file_uploader(
        label=_("Загрузите не более 2 изображений (желательно 2 ключевых кадра одного исследования в аксиальной и сагиттальной плоскости):"),
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, 
        label_visibility='visible'
    )

    if len(uploaded_files) > 2:
        st.warning(_("Могут быть обработаны только 2 файла!"))
        uploaded_files = uploaded_files[:2]
    
    # Инициализация процессора с выбранной версией модели
    processor = get_processor(model_version)
    
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = {}
    
    if 'imgs' not in st.session_state:
        st.session_state['imgs'] = {}
    
    if 'processed_images' not in st.session_state:
        st.session_state['processed_images'] = {}
    
    if 's3_object_names' not in st.session_state:
        st.session_state['s3_object_names'] = {}
    
    # Очистка кэша при смене версии модели
    if 'current_model_version' not in st.session_state:
        st.session_state['current_model_version'] = model_version
    elif st.session_state['current_model_version'] != model_version:
        st.session_state['current_model_version'] = model_version
        st.session_state['imgs'] = {}
        st.session_state['processed_images'] = {}
        st.session_state['s3_object_names'] = {}
        cache_process_image.clear()

    if uploaded_files:
        process_uploaded_files(uploaded_files)
    else:
        example_img = image_select(
            label=_("Или посмотрите примеры:"),
            images=list(example_images.values()),
            captions=[_("Норма (сагиттальная)"), _("Норма (аксиальная)"), _("Патология (сагиттальная)"), _("Патология (аксиальная)")]
        )
        process_example_files(list(example_images.values()))
    
    processed_images = st.session_state['processed_images']
    imgs = st.session_state['imgs']
    col1, col2 = st.columns(2)
    
    if uploaded_files:
        with col1:
            # Показываем только загруженные изображения (без префикса "example_")
            options = [key for key in processed_images.keys() if not key.startswith('example_')]
            option = st.selectbox(_('Выберите конкретный снимок:'), options, label_visibility='collapsed')
            if option:
                st.image(imgs[option], caption=_('Выбранное изображение'), width='stretch')
                col3, col4 = st.columns(2)
                with col3:
                    st.metric(label=_("Корректность"), value=int(processed_images[option]["quality"]["prediction_prob"]*100))
                with col4:
                    st.metric(label=_("Патология"), value=int(processed_images[option]["pathology"]["prediction_prob"]*100))
        
        with col2:
            if option:
                selected_image_data = processed_images[option]
        
                tabs = st.tabs([_("Зона интереса"), _("Корректность"), _("Патология")])
                
                with tabs[0]:
                    st.image(selected_image_data['cropped_img'], width='stretch')
                    st.markdown(_('Голова в {} плоскости с вероятностью **:red[{}%]**').format(selected_image_data["plane"]["type"], int(selected_image_data["plane"]["prediction_prob"]*100)))
                    
                with tabs[1]:
                    quality_image = selected_image_data["quality"]["heatmap"]
                    st.image(quality_image, width='stretch')
                    st.markdown(_('Изображение качественное с вероятностью **:red[{}%]**').format(int(selected_image_data["quality"]["prediction_prob"]*100)))
                    st.markdown("---")
                    st.markdown(_("**Пояснение использования цветов:**"))
                    st.markdown(_("**Красный:** Модель считает эти области важными для распознавания качества изображения."))
                    st.markdown(_("**Желтый/зеленый:** Умеренно важные области."))
                    st.markdown(_("**Синий/темный:** Области, незначительные для качества изображения."))
                
                with tabs[2]:
                    pathology_image = selected_image_data["pathology"]["heatmap"]
                    st.image(pathology_image, width='stretch')
                    st.markdown(_('На изображении присутствует патология с вероятностью **:red[{}%]**').format(int(selected_image_data["pathology"]["prediction_prob"]*100)))
                    st.markdown("---")
                    st.markdown(_("**Пояснение использования цветов:**"))
                    st.markdown(_("**Красный:** Модель считает эти области важными для распознавания патологических паттернов."))
                    st.markdown(_("**Желтый/зеленый:** Умеренно важные области."))
                    st.markdown(_("**Синий/темный:** Области, незначительные для распознавания патологических паттернов."))
    
    else:
        with col1:
            # Показываем только примеры изображений (с префиксом "example_")
            options = [key for key in processed_images.keys() if key.startswith('example_')]
            option = st.selectbox(_('Выберите конкретный снимок:'), options, label_visibility='collapsed', disabled=True)
            if option:
                st.image(imgs[option], caption=_('Выбранное изображение'), width='stretch')
                col3, col4 = st.columns(2)
                with col3:
                    st.metric(label=_("Корректность"), value=int(processed_images[option]["quality"]["prediction_prob"]*100))
                with col4:
                    st.metric(label=_("Патология"), value=int(processed_images[option]["pathology"]["prediction_prob"]*100))
    
        with col2:
            if option:
                selected_image_data = processed_images[option]
        
                tabs = st.tabs([_("Зона интереса"), _("Корректность"), _("Патология")])
                
                with tabs[0]:
                    st.image(selected_image_data['cropped_img'], width='stretch')
                    st.markdown(_('Голова в {} плоскости с вероятностью **:red[{}%]**').format(selected_image_data["plane"]["type"], int(selected_image_data["plane"]["prediction_prob"]*100)))
                    
                with tabs[1]:
                    quality_image = selected_image_data["quality"]["heatmap"]
                    st.image(quality_image, width='stretch')
                    st.markdown(_('Изображение качественное с вероятностью **:red[{}%]**').format(int(selected_image_data["quality"]["prediction_prob"]*100)))
                    st.markdown("---")
                    st.markdown(_("**Пояснение использования цветов:**"))
                    st.markdown(_("**Красный:** Модель считает эти области важными для распознавания качества изображения."))
                    st.markdown(_("**Желтый/зеленый:** Умеренно важные области."))
                    st.markdown(_("**Синий/темный:** Области, незначительные для качества изображения."))
                    
                with tabs[2]:
                    pathology_image = selected_image_data["pathology"]["heatmap"]
                    st.image(pathology_image, width='stretch')
                    st.markdown(_('На изображении присутствует патология с вероятностью **:red[{}%]**').format(int(selected_image_data["pathology"]["prediction_prob"]*100)))
                    st.markdown("---")
                    st.markdown(_("**Пояснение использования цветов:**"))
                    st.markdown(_("**Красный:** Модель считает эти области важными для распознавания патологических паттернов."))
                    st.markdown(_("**Желтый/зеленый:** Умеренно важные области."))
                    st.markdown(_("**Синий/темный:** Области, незначительные для распознавания патологических паттернов."))

    # Блок для отправки обратной связи

    # Активность кнопки Отправить (активна если загружены файлы)
    if uploaded_files:
        button_disable = False
    else:
        button_disable = True

    st.markdown("---")
    st.write(_("Оставьте обратную связь:"))
    
    action = st.radio(_("Вы согласны с работой сервиса?"), [_("Да"), _("Нет")])
    
    if action == _("Нет"):
        comment = st.text_area(_("Комментарий"))
    
    if action == _("Нет"):
        form_data["comment"] = comment
    
    if st.button(_("Отправить"), disabled=button_disable):
    
        # Генерация уникального имени файла с использованием UUID
        unique_id = get_unique_id()
        # Очищаем имя файла от слешей и других символов, которые могут вызвать проблемы с путями
        clean_option = option.replace('/', '_').replace('\\', '_').replace(':', '_')
        img_file_name = f'img_{unique_id}_{clean_option}'

        # Формирование JSON с фидбэком
        form_data["timestamp"] = str(datetime.now())
        form_data["old_image_name"] = option
        form_data["new_image_name"] = img_file_name
        form_data["patology_prediction"] = int(selected_image_data["pathology"]["prediction_prob"]*100)
        form_data["quality_prediction"] = int(selected_image_data["quality"]["prediction_prob"]*100)
        form_data["plane_type"] = selected_image_data["plane"]["type"]
        form_data["action"] = action
        form_data["model_version"] = model_version
        
        # Добавление информации о пользователе
        if is_authenticated():
            user = st.session_state['user']
            form_data["user_email"] = user['email']
            form_data["user_organization"] = user['organization']
            
            # Получение имени объекта в S3, если изображение уже было загружено
            s3_object_name = st.session_state['s3_object_names'].get(option)
            
            # Если изображение еще не было загружено в S3, загружаем его
            if not s3_object_name:
                # Проверяем, что изображение существует в session_state
                if option in st.session_state['imgs']:
                    # Сохранение выбранного изображения (временно)
                    st.session_state['imgs'][option].save(img_file_name)
                    
                    # Загрузка оригинального файла в Yandex Object Storage
                    success, s3_object_name = upload_to_yandex_cloud(img_file_name, BUCKET, img_file_name)
                else:
                    st.error(f"Изображение {option} не найдено в сессии. Пожалуйста, перезагрузите изображение.")
                    success = False
            
            # Логирование обратной связи
            if success:
                user_logger.log_user_action(
                    user_id=user['id'],
                    user_email=user['email'],
                    user_organization=user['organization'],
                    action_type="submit_feedback",
                    action_details={
                        "feedback_time": datetime.now(timezone.utc).isoformat(),
                        "image_name": option,
                        "image_s3_object_name": s3_object_name,
                        "feedback_data": form_data,
                        "model_version": model_version
                    }
                )
        else: 
            # Если пользователь не авторизован, используем старый метод
            # Проверяем, что изображение существует в session_state
            if option in st.session_state['imgs']:
                # Сохранение выбранного изображения (временно)
                st.session_state['imgs'][option].save(img_file_name)
                
                # Загрузка оригинального файла в Yandex Object Storage
                upload_to_yandex_cloud(img_file_name, BUCKET)
            else:
                st.error(f"Изображение {option} не найдено в сессии. Пожалуйста, перезагрузите изображение.")

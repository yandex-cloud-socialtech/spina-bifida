import streamlit as st
from streamlit_image_select import image_select

import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from stqdm import stqdm
import io
from utils import *
import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import uuid
import json
from datetime import datetime

# BugFix SSL error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Форма обратной связи
form_data = {}

# Загрузка переменных окружения из .env файла
load_dotenv()

# Функция для отображения модального окна с условиями использования
def show_terms_modal():
    st.write("### Spina Bifida")
    st.write("Пожалуйста, прочитайте и примите наши условия использования данного сервиса, чтобы продолжить.")
    st.markdown("---")
    st.write("""
        **Юридическая оговорка:** 
        Авторы данного веб-сервиса приложили максимум усилий для обеспечения точности анализа УЗИ первого триместра. Однако информация постоянно меняется вследствие непрерывных исследований и клинической деятельности, значительных расхождений во мнениях различных организаций, уникального характера отдельных случаев.  """)
    st.write("""**Информация в данном веб-сервисе предназначена исключительно для использования** **:red[специалистами с медицинским образованием, практикующими специалистами в области  УЗ диагностики, перинатологии]** в научных целях и **не должна рассматриваться как профессиональный совет или замена консультации с квалифицированным врачом или другим специалистом в области здравоохранения.**
        """)
    st.write("""В сервис **:red[запрещено]** загружать изображения, содержащие персональные данные пациента (ФИО, дата рождения). 
        """)
    st.write("""Исследования, загруженные в веб-сервис, будут сохраняться и использоваться для улучшения качества данной модели. 
        """)
    st.write("""**:red[Результат работы сервиса не является диагнозом, за консультацией обратитесь к врачу.]**
             """)
    
    accept = st.button("Я прочитал(а) и соглашаюсь с вышеперечисленными условиями")

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
        st.success(f"Файл {file_name} успешно загружен в Yandex Object Storage.")
        # Удаление локального файла после успешной загрузки
        os.remove(file_name)
    except FileNotFoundError:
        st.error("Файл не найден.")
    except NoCredentialsError:
        st.error("Ошибка с учетными данными.")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")

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
    st.markdown("Представленный алгоритм на основе ИИ направлен на детектирование спектра патологий центральной нервной системы (в том числе Спина бифида) на эхографических снимках головного мозга плода в первом триместре беременности. Для проведения анализа необходимо загрузить один или несколько ключевых снимков.")
    st.markdown("После получения результатов анализа оставьте, пожалуйста, обратную связь.")

# Примеры изображений для выбора
    example_images = {
        "Example 1": "example_images/norm-sagittal.jpg",
        "Example 2": "example_images/norm-axial.jpg",
        "Example 3": "example_images/patology-sagittal.jpg",
        "Example 4": "example_images/patology-axial.jpg"
    }   
    
    # Элемент для загрузки файлов
    uploaded_files = st.file_uploader(
        label="Загрузите не более 2 изображений (желательно 2 ключевых кадра одного исследования в аксиальной и сагиттальной плоскости):", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True, 
        label_visibility='visible'
    )

    if len(uploaded_files) > 2:
        st.warning("Могут быть обработаны только 2 файла!")
        uploaded_files = uploaded_files[:2]
    
    # Настройка модели и обработка изображений
    @st.cache_resource(show_spinner = "Загрузка модели ...")
    def get_processor():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return MedicalImageProcessor(
            yolo_model_path='models/best_object_detection.pt',
            axial_quality_model_path='models/axial_quality.pt',
            axial_pathology_model_path='models/axial_pathology.pt',
            sagittal_quality_model_path='models/sagittal_quality.pt',
            sagittal_pathology_model_path='models/sagittal_pathology.pt',
            device=device
        )
    
    processor = get_processor()
    
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = {}
    
    @st.cache_data(show_spinner = "Обработка изображения ...", ttl = 3600, max_entries = 10)
    def cache_process_image(img_bytes, img_name):
        return processor.process_image(img_bytes, img_name)
    
    def process_uploaded_files(uploaded_files):
        with stqdm(uploaded_files, mininterval=1) as pbar:
            st.session_state['imgs'] = {}
            st.session_state['processed_images'] = {}
            for uploaded_file in pbar:
                img = Image.open(uploaded_file)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                img_name = uploaded_file.name
                if img_name not in st.session_state['imgs']:
                    result = cache_process_image(img_bytes, img_name)
                    st.session_state['imgs'][img_name] = img
                    st.session_state['processed_images'][img_name] = result
    
    def process_example_files(example_files):
        st.session_state['imgs'] = {}
        st.session_state['processed_images'] = {}
        for example_file in example_files:
            img = Image.open(example_file)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            img_name = example_file
            if img_name not in st.session_state['imgs']:
                result = cache_process_image(img_bytes, img_name)
                st.session_state['imgs'][img_name] = img
                st.session_state['processed_images'][img_name] = result
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    else:
        example_img = image_select(
            label="Или посмотрите примеры:",
            images=list(example_images.values()),
            captions=["Норма (сагиттальная)", "Норма (аксиальная)", "Патология (сагиттальная)", "Патология (аксиальная)"]
        )
        process_example_files(list(example_images.values()))
    
    processed_images = st.session_state['processed_images']
    imgs = st.session_state['imgs']
    col1, col2 = st.columns(2)
    
    if uploaded_files:
        with col1:
            options = list(processed_images.keys())
            option = st.selectbox('Выберите конкретный снимок:', options, label_visibility='collapsed')
            if option:
                st.image(imgs[option], caption='Выбранное изображение', use_column_width=True)
        
        with col2:
            if option:
                selected_image_data = processed_images[option]
        
                tabs = st.tabs(["Зона интереса", "Корректность", "Патология"])
                
                with tabs[0]:
                    st.image(selected_image_data['cropped_img'], use_column_width=True)
                    st.markdown(f'Голова в {selected_image_data["plane"]["type"]} плоскости с вероятностью **:red[{int(selected_image_data["plane"]["prediction_prob"]*100)}%]**')
                    
                with tabs[1]:
                    quality_image = selected_image_data["quality"]["heatmap"]
                    st.image(quality_image, use_column_width=True)
                    st.markdown(f'Изображение качественное с вероятностью **:red[{int(selected_image_data["quality"]["prediction_prob"]*100)}%]**')
                    st.markdown("---")
                    st.markdown("**Пояснение использования цветов:**")
                    st.markdown("**Красный:** Модель считает эти области важными для распознавания качества изображения.")
                    st.markdown("**Желтый/зеленый:** Умеренно важные области.")
                    st.markdown("**Синий/темный:** Области, незначительные для качества изображения.")
                
                with tabs[2]:
                    pathology_image = selected_image_data["pathology"]["heatmap"]
                    st.image(pathology_image, use_column_width=True)
                    st.markdown(f'На изображении присутствует патология с вероятностью **:red[{int(selected_image_data["pathology"]["prediction_prob"]*100)}%]**')
                    st.markdown("---")
                    st.markdown("**Пояснение использования цветов:**")
                    st.markdown("**Красный:** Модель считает эти области важными для распознавания патологических паттернов.")
                    st.markdown("**Желтый/зеленый:** Умеренно важные области.")
                    st.markdown("**Синий/темный:** Области, незначительные для распознавания патологических паттернов.")
    
    else:
        with col1:
            options = list(processed_images.keys())
            option = st.selectbox('Выберите конкретный снимок:', options, label_visibility='collapsed', disabled=True)
            if option:
                st.image(imgs[str(example_img)], caption='Выбранное изображение', use_column_width=True)
    
        with col2:
            if option:
                selected_image_data = processed_images[str(example_img)]
        
                tabs = st.tabs(["Зона интереса", "Корректность", "Патология"])
                
                with tabs[0]:
                    st.image(selected_image_data['cropped_img'], use_column_width=True)
                    st.markdown(f'Голова в {selected_image_data["plane"]["type"]} плоскости с вероятностью **:red[{int(selected_image_data["plane"]["prediction_prob"]*100)}%]**')
                    
                with tabs[1]:
                    quality_image = selected_image_data["quality"]["heatmap"]
                    st.image(quality_image, use_column_width=True)
                    st.markdown(f'Изображение качественное с вероятностью **:red[{int(selected_image_data["quality"]["prediction_prob"]*100)}%]**')
                    st.markdown("---")
                    st.markdown("**Пояснение использования цветов:**")
                    st.markdown("**Красный:** Модель считает эти области важными для распознавания качества изображения.")
                    st.markdown("**Желтый/зеленый:** Умеренно важные области.")
                    st.markdown("**Синий/темный:** Области, незначительные для качества изображения.")
                    
                with tabs[2]:
                    pathology_image = selected_image_data["pathology"]["heatmap"]
                    st.image(pathology_image, use_column_width=True)
                    st.markdown(f'На изображении присутствует патология с вероятностью **:red[{int(selected_image_data["pathology"]["prediction_prob"]*100)}%]**')
                    st.markdown("---")
                    st.markdown("**Пояснение использования цветов:**")
                    st.markdown("**Красный:** Модель считает эти области важными для распознавания патологических паттернов.")
                    st.markdown("**Желтый/зеленый:** Умеренно важные области.")
                    st.markdown("**Синий/темный:** Области, незначительные для распознавания патологических паттернов.")

    # Блок для отправки обратной связи

    # Активность кнопки Отправить (активна если загружены файлы)
    if uploaded_files:
        button_disable = False
    else:
        button_disable = True

    st.markdown("---")
    st.write("Оставьте обратную связь:")
    
    action = st.radio("Вы согласны с работой сервиса?", ["Да", "Нет"])
    
    if action == "Нет":
        comment = st.text_area("Комментарий")
    
    if action == "Нет":
        form_data["comment"] = comment
    
    if st.button("Отправить", disabled=button_disable):
    
        # Генерация уникального имени файла с использованием UUID
        unique_id = str(uuid.uuid4())
        json_file_name = f'form_data_{unique_id}.json'
        img_file_name = f'img_{unique_id}_{option}'

        # Формирование JSON с фидбэком
        form_data["timestamp"] = str(datetime.now())
        form_data["old_image_name"] = option
        form_data["new_image_name"] = img_file_name
        form_data["patology_prediction"] = int(selected_image_data["pathology"]["prediction_prob"]*100)
        form_data["quality_prediction"] = int(selected_image_data["quality"]["prediction_prob"]*100)
        form_data["plane_type"] = selected_image_data["plane"]["type"]
        form_data["action"] = action
    
        # Сохранение данных формы в JSON файл (временно)
        with open(json_file_name, 'w') as f:
            json.dump(form_data, f)

        # Сохранение выбранного изображения (временно)
        st.session_state['imgs'][option].save(img_file_name)
    
        BUCKET = os.getenv('BUCKET')

        # Загрузка JSON файла в Yandex Object Storage
        upload_to_yandex_cloud(json_file_name, BUCKET)
        
        # Загрузка оригинального файла в Yandex Object Storage
        upload_to_yandex_cloud(img_file_name, BUCKET)
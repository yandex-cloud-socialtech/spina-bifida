import streamlit as st
import logging
import os
from datetime import datetime, timezone
from auth import (
    register_user, login_user, logout_user, 
    is_authenticated, is_admin, 
    approve_user, reject_user
)

# Импорт логгера из единого экземпляра
from logger_instance import user_logger

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def login_page(ydb_client):
    """
    Страница входа в систему
    
    Args:
        ydb_client: Клиент YDB
    """
    st.title("Вход в систему")
    
    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Пароль", type="password", key="login_password")
        submit_button = st.form_submit_button("Войти")
        
        if submit_button:
            if not email or not password:
                st.error("Пожалуйста, заполните все поля")
            else:
                success, message = login_user(ydb_client, email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    st.markdown("Нет аккаунта? [Зарегистрироваться](#register)")
    
    # Кнопка для перехода на страницу регистрации
    if st.button("Зарегистрироваться", key="goto_register"):
        st.session_state['auth_page'] = 'register'
        st.rerun()

def register_page(ydb_client):
    """
    Страница регистрации
    
    Args:
        ydb_client: Клиент YDB
    """
    st.title("Регистрация")
    
    with st.form("register_form"):
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Пароль", type="password", key="register_password")
        password_confirm = st.text_input("Подтверждение пароля", type="password", key="register_password_confirm")
        organization = st.text_input("Организация", key="register_organization")
        
        submit_button = st.form_submit_button("Зарегистрироваться")
        
        if submit_button:
            if not email or not password or not password_confirm or not organization:
                st.error("Пожалуйста, заполните все поля")
            elif password != password_confirm:
                st.error("Пароли не совпадают")
            else:
                success, message = register_user(ydb_client, email, password, organization)
                if success:
                    st.success(message)
                    # Переход на страницу входа после успешной регистрации
                    st.session_state['auth_page'] = 'login'
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    st.markdown("Уже есть аккаунт? [Войти](#login)")
    
    # Кнопка для перехода на страницу входа
    if st.button("Войти", key="goto_login"):
        st.session_state['auth_page'] = 'login'
        st.rerun()

def admin_panel(ydb_client):
    """
    Панель администратора
    
    Args:
        ydb_client: Клиент YDB
    """
    st.title("Панель администратора")
    
    # Модерация отключена - пользователи регистрируются с активным статусом
    st.info("Модерация пользователей отключена. Новые пользователи регистрируются с активным статусом и могут сразу войти в систему.")
    
    # Можно добавить другую административную функциональность здесь
    st.subheader("Доступные функции")
    st.write("• Просмотр логов системы")
    st.write("• Управление настройками")
    st.write("• Статистика использования")

def user_info():
    """
    Отображение информации о текущем пользователе
    """
    if is_authenticated():
        user = st.session_state['user']
        
        with st.sidebar:
            # Декодирование байтовых строк
            email = user['email']
            if isinstance(email, bytes):
                email = email.decode('utf-8')
            
            organization = user['organization']
            if isinstance(organization, bytes):
                organization = organization.decode('utf-8')
            
            st.write(f"**Пользователь:** {email}")
            st.write(f"**Организация:** {organization}")
            
            if is_admin():
                st.write("**Роль:** Администратор")
                
                # Кнопка для перехода в панель администратора
                if st.button("Панель администратора"):
                    
                    st.session_state['show_admin_panel'] = True
                    st.rerun()
            else:
                st.write("**Роль:** Пользователь")
            
            # Кнопка выхода
            if st.button("Выйти"):
                logout_user()
                st.rerun()

def auth_ui(ydb_client):
    """
    Основной интерфейс аутентификации
    
    Args:
        ydb_client: Клиент YDB
        
    Returns:
        bool: True, если пользователь авторизован, иначе False
    """
    # Инициализация состояния
    if 'auth_page' not in st.session_state:
        st.session_state['auth_page'] = 'login'
    
    if 'show_admin_panel' not in st.session_state:
        st.session_state['show_admin_panel'] = False
    
    # Если пользователь авторизован
    if is_authenticated():
        # Отображение информации о пользователе
        user_info()
        
        # Если пользователь администратор и нужно показать панель администратора
        if is_admin() and st.session_state.get('show_admin_panel', False):
            admin_panel(ydb_client)
            
            # Кнопка для возврата к основному приложению
            if st.button("Вернуться к основному приложению"):
                
                st.session_state['show_admin_panel'] = False
                st.rerun()
            
            # Не показываем основное приложение
            return False
        
        # Показываем основное приложение
        return True
    
    # Если пользователь не авторизован
    if st.session_state['auth_page'] == 'login':
        login_page(ydb_client)
    else:
        register_page(ydb_client)
    
    # Не показываем основное приложение
    return False

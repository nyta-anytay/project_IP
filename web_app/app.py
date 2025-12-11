"""
Streamlit веб-приложение для детекции масок
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ===== ПУТИ К ФАЙЛАМ =====
# ВАЖНО: все файлы должны быть в той же папке, что и app.py
MODEL1_PATH = 'model1_hog_svm.pkl'    # Первый .pkl файл
MODEL2_PATH = 'model2_haar_rf.pkl'    # Второй .pkl файл  
MODEL3_PATH = 'model3_cnn.h5'         # .h5 файл
LABELS_MAP_PATH = 'labels_map.json'  # JSON с метками

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== КАСТОМНЫЕ СТИЛИ =====
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ФАЙЛОВ =====
def check_files_exist():
    """Проверяем наличие файлов моделей и labels_map"""
    files_needed = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABELS_MAP_PATH]
    existing_files = []
    missing_files = []
    
    for file in files_needed:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    return existing_files, missing_files

# ===== ЗАГРУЗКА МОДЕЛЕЙ С ОБРАБОТКОЙ ОШИБОК =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей и labels_map"""
    # Проверяем файлы
    existing_files, missing_files = check_files_exist()
    
    if missing_files:
        st.warning(f"⚠️ Отсутствуют файлы: {', '.join(missing_files)}")
        st.info(f"✅ Найдены файлы: {', '.join(existing_files) if existing_files else 'нет'}")
    
    try:
        # ===== labels_map =====
        if os.path.exists(LABELS_MAP_PATH):
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
                st.sidebar.success(f"✅ labels_map загружен: {labels_map}")
        else:
            labels_map = {0: "Без маски", 1: "С маской"}
            st.sidebar.info(f"ℹ️ Используется стандартный labels_map: {labels_map}")

        models_loaded = []
        model1, model2, model3 = None, None, None

        # ===== Модель 1: HOG + SVM =====
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            models_loaded.append(("model1_hog_svm", True, ""))
            st.sidebar.success(f"✅ {MODEL1_PATH} загружена")
        except Exception as e:
            models_loaded.append(("model1_hog_svm", False, str(e)))
            st.sidebar.error(f"❌ Ошибка загрузки {MODEL1_PATH}: {str(e)[:50]}...")

        # ===== Модель 2: Haar + RF =====
        try:
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
            models_loaded.append(("model2_haar_rf", True, ""))
            st.sidebar.success(f"✅ {MODEL2_PATH} загружена")
        except Exception as e:
            models_loaded.append(("model2_haar_rf", False, str(e)))
            st.sidebar.error(f"❌ Ошибка загрузки {MODEL2_PATH}: {str(e)[:50]}...")

        # ===== Модель 3: CNN =====
        try:
            if os.path.exists(MODEL3_PATH):
                model3_keras = load_model(
                    MODEL3_PATH,
                    compile=False,
                    custom_objects={'BatchNormalization': BatchNormalization}
                )
                
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    def predict_proba(self, X):
                        return self.model.predict(X, verbose=0)
                
                model3 = CNNWrapper(model3_keras)
                models_loaded.append(("model3_cnn", True, ""))
                st.sidebar.success(f"✅ {MODEL3_PATH} загружена")
            else:
                models_loaded.append(("model3_cnn", False, f"Файл не найден: {MODEL3_PATH}"))
                st.sidebar.error(f"❌ Файл не найден: {MODEL3_PATH}")
        except Exception as e:
            models_loaded.append(("model_cnn3", False, str(e)))
            st.sidebar.error(f"❌ Ошибка загрузки {MODEL3_PATH}: {str(e)[:50]}...")

        # Проверяем, есть ли хотя бы одна модель
        any_loaded = any(status for _, status, _ in models_loaded)
        
        # Формируем сообщение об ошибках
        error_msg = ""
        if not any_loaded:
            error_details = [f"{name}: {msg}" for name, status, msg in models_loaded if not status and msg]
            error_msg = f"Ошибки загрузки: {'; '.join(error_details)}"

        return model1, model2, model3, labels_map, any_loaded, error_msg

    except Exception as e:
        return None, None, None, {}, False, f"Общая ошибка загрузки: {str(e)}"

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR: ОСНОВНЫЕ НАСТРОЙКИ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Информация о файлах
    if st.checkbox("📁 Показать информацию о файлах", True):
        existing, missing = check_files_exist()
        st.write("**Найдены файлы:**")
        for file in existing:
            size_kb = os.path.getsize(file) / 1024
            st.success(f"✅ {file} ({size_kb:.1f} KB)")
        
        if missing:
            st.write("**Отсутствуют файлы:**")
            for file in missing:
                st.error(f"❌ {file}")
    
    # Выбор модели
    available_models = []
    if model1:
        available_models.append("HOG + SVM")
    if model2:
        available_models.append("Haar Cascade + RF")
    if model3:
        available_models.append("CNN (Deep Learning)")
    
    if available_models:
        model_choice = st.selectbox(
            "Выберите модель:",
            ["Все модели"] + available_models,
            help="Выберите модель для предсказания"
        )
    else:
        model_choice = "Нет доступных моделей"
        st.error("❌ Нет доступных моделей для выбора")
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Минимальная уверенность для положительного предсказания"
    )
    
    # Кнопка перезагрузки
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Статус моделей
    st.markdown("### 📊 Статус моделей")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.metric("HOG+SVM", "✅" if model1 else "❌")
    with status_col2:
        st.metric("Haar+RF", "✅" if model2 else "❌")
    with status_col3:
        st.metric("CNN", "✅" if model3 else "❌")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Если нет ни одной модели
if not models_loaded:
    st.error("⚠️ Не удалось загрузить ни одной модели!")
    st.warning(error_msg)
    
    st.info("""
    ## 🚀 Решение для Streamlit Cloud:
    
    ### 1. **Создайте правильную структуру файлов:**
    ```
    ваша-папка/
    ├── app.py                    # Этот файл
    ├── model_hog_svm.pkl        # Ваш первый .pkl
    ├── model_haar_rf.pkl        # Ваш второй .pkl  
    ├── model_cnn.h5             # Ваш .h5 файл
    ├── labels_map.json          # JSON с метками (или будет создан)
    └── requirements.txt         # Список зависимостей
    ```
    
    ### 2. **Переименуйте ваши файлы:**
    ```bash
    # Ваши текущие файлы должны называться так:
    mv ваш_файл1.pkl model_hog_svm.pkl
    mv ваш_файл2.pkl model_haar_rf.pkl  
    mv ваш_файл.h5 model_cnn.h5
    ```
    
    ### 3. **Создайте requirements.txt:**
    ```txt
    streamlit==1.29.0
    tensorflow==2.15.0
    opencv-python-headless==4.8.1  # Используйте headless версию!
    numpy==1.24.3
    Pillow==10.1.0
    scikit-learn==1.3.2
    pandas==2.1.4
    ```
    
    ### 4. **Загрузите ВСЕ файлы на GitHub** (не только код!)
    """)
    
    # Показываем текущую директорию
    if st.checkbox("Показать содержимое текущей директории"):
        st.write("Текущая рабочая директория:", os.getcwd())
        st.write("Содержимое:", os.listdir('.'))
    
    st.stop()

# Создание колонок
col1, col2 = st.columns([1, 1], gap="large")

# ===== ЛЕВАЯ КОЛОНКА: ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio(
        "Выберите способ:",
        ["Загрузить файл", "Использовать камеру"],
        horizontal=True
    )
    
    uploaded_file = None
    
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Поддерживаемые форматы: JPG, JPEG, PNG, BMP"
        )
    else:
        camera_image = st.camera_input("Сделайте фото")
        if camera_image is not None:
            uploaded_file = camera_image

# ===== ПРАВАЯ КОЛОНКА: РЕЗУЛЬТАТЫ =====
with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        try:
            # Загружаем и обрабатываем изображение
            image = Image.open(uploaded_file)
            
            # Показываем изображение в левой колонке
            with col1:
                st.image(image, caption='Загруженное изображение', use_column_width=True)
                img_array = np.array(image)
                st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")
            
            # Обработка изображения для моделей
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Ресайз для модели (убедитесь, что размер соответствует вашим моделям)
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0  # Нормализация
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("📊 Сравнение всех моделей")
                
                models = []
                if model1:
                    models.append((model1, "HOG + SVM", "🔵", "#1f77b4"))
                if model2:
                    models.append((model2, "Haar Cascade + RF", "🟢", "#2ca02c"))
                if model3:
                    models.append((model3, "CNN (Deep Learning)", "🔴", "#d62728"))
                
                for model, name, icon, color in models:
                    with st.container():
                        st.markdown(f"### {icon} {name}")
                        
                        try:
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            # Определяем класс
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                            
                            # Отображение результата
                            result_col1, result_col2 = st.columns([2, 1])
                            with result_col1:
                                if confidence >= confidence_threshold:
                                    if prediction == "С маской" or pred_class == 1:
                                        st.success(f"✅ **{prediction}**")
                                    else:
                                        st.error(f"❌ **{prediction}**")
                                else:
                                    st.warning(f"⚠️ **{prediction}** (низкая уверенность)")
                            
                            with result_col2:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            st.progress(float(confidence))
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)[:100]}")
                        
                        st.markdown("---")
            
            elif model_choice in ["HOG + SVM", "Haar Cascade + RF", "CNN (Deep Learning)"]:
                # Одна модель
                model_map = {
                    "HOG + SVM": (model1, "🔵"),
                    "Haar Cascade + RF": (model2, "🟢"),
                    "CNN (Deep Learning)": (model3, "🔴")
                }
                
                model, icon = model_map[model_choice]
                
                if model is None:
                    st.error(f"Модель {model_choice} не загружена")
                else:
                    with st.spinner(f'Обработка {model_choice}...'):
                        try:
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                            
                            # Большой результат
                            st.markdown(f"## {icon} {prediction}")
                            
                            if confidence >= confidence_threshold:
                                if prediction == "С маской" or pred_class == 1:
                                    st.success("✅ Маска обнаружена!")
                                else:
                                    st.error("❌ Маска не обнаружена!")
                            else:
                                st.warning("⚠️ Низкая уверенность в предсказании")
                            
                            # Метрики
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Класс", prediction)
                            
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            with col_c:
                                status = "✅" if confidence >= confidence_threshold else "⚠️"
                                st.metric("Статус", status)
                            
                            st.progress(float(confidence))
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)}")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
    else:
        # Когда нет изображения
        st.info("👆 Загрузите изображение для начала детекции")
        
        st.markdown("""
        ### 💡 Как использовать:
        
        1. **Загрузите фото** человека (лицо должно быть видно)
        2. **Выберите модель** для предсказания
        3. **Получите результат** детекции маски
        
        ### 🎯 Рекомендации:
        - Четкое, хорошо освещенное лицо
        - Портретная ориентация
        - Минимум посторонних объектов
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("📋 Инструкция по деплою"):
    st.markdown("""
    ### Для успешного деплоя на Streamlit Cloud:
    
    1. **Создайте репозиторий со следующей структурой:**
    ```
    mask-detection-app/
    ├── app.py                    # Этот файл
    ├── model_hog_svm.pkl        # Ваш HOG+SVM .pkl
    ├── model_haar_rf.pkl        # Ваш Haar+RF .pkl
    ├── model_cnn.h5             # Ваша CNN .h5
    ├── labels_map.json          # (опционально)
    └── requirements.txt         # Важные!
    ```
    
    2. **requirements.txt должен содержать:**
    ```txt
    streamlit==1.29.0
    tensorflow==2.15.0
    opencv-python-headless==4.8.1
    numpy==1.24.3
    Pillow==10.1.0
    scikit-learn==1.3.2
    ```
    
    3. **На Streamlit Cloud:**
       - Подключите GitHub репозиторий
       - Main file path: `app.py`
       - Нажмите Deploy
    """)

# Copyright
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System</p>
    </div>
""", unsafe_allow_html=True)
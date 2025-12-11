"""
Streamlit веб-приложение для детекции масок
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('src') 




# ===== ВАЖНО: ДЛЯ STREAMLIT CLOUD УБИРАЕМ ПУТИ ИЗ КОНФИГА =====
# Вместо импорта из config, задаем пути напрямую
MODEL1_PATH = 'web_app/model1_hog_svm.pkl'  # или другое имя вашего первого .pkl файла
MODEL2_PATH = 'web_app/model2_haar_rf.pkl'  # или другое имя вашего второго .pkl файла
MODEL3_PATH = 'web_app/model3_cnn.h5'   # имя вашего .h5 файла
LABELS_MAP_PATH = 'web_app/labels_map.json'  # если есть, или задаем вручную

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
        return None, None, None, None, False, f"Отсутствуют файлы: {', '.join(missing_files)}"

    try:
        # ===== labels_map =====
        if os.path.exists(LABELS_MAP_PATH):
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
        else:
            labels_map = {0: "Без маски", 1: "С маской"}
            os.makedirs(os.path.dirname(LABELS_MAP_PATH), exist_ok=True)
            with open(LABELS_MAP_PATH, 'w') as f:
                json.dump(labels_map, f)

        models_loaded = []

        # ===== Модель 1: HOG + SVM =====
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            models_loaded.append(("model1_hog_svm", True, ""))
        except Exception as e:
            model1 = None
            models_loaded.append(("model1_hog_svm", False, str(e)))

        # ===== Модель 2: Haar + RF =====
        try:
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
            models_loaded.append(("model2_haar_rf", True, ""))
        except Exception as e:
            model2 = None
            models_loaded.append(("model2_haar_rf", False, str(e)))

        # ===== Модель 3: CNN =====
        try:
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
        except Exception as e:
            model3 = None
            models_loaded.append(("model3_cnn", False, str(e)))

        # Проверяем, все ли модели загружены
        all_loaded = all(status for _, status, _ in models_loaded)

        # Формируем сообщение об ошибках
        error_msg = ""
        if not all_loaded:
            error_details = [f"{name}: {msg}" for name, status, msg in models_loaded if not status and msg]
            error_msg = f"Ошибки загрузки: {'; '.join(error_details)}"

        return model1, model2, model3, labels_map, all_loaded, error_msg

    except Exception as e:
        return None, None, None, None, False, f"Ошибка загрузки: {str(e)}"

# ===== ОТЛАДОЧНАЯ ИНФОРМАЦИЯ В SIDEBAR =====
with st.sidebar:
    st.header("🔍 Отладка")
    
    if st.checkbox("Показать информацию о файлах"):
        existing, missing = check_files_exist()
        st.write("**Найдены файлы:**")
        for file in existing:
            st.success(f"✅ {file} ({os.path.getsize(file)} байт)")
        
        if missing:
            st.write("**Отсутствуют файлы:**")
            for file in missing:
                st.error(f"❌ {file}")
    
    # Принудительная перезагрузка моделей
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()

# Загрузка моделей
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR: ОСНОВНЫЕ НАСТРОЙКИ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
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
        st.error("Нет доступных моделей для выбора")
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Минимальная уверенность для положительного предсказания"
    )
    
    st.markdown("---")
    
    # Информация о загруженных моделях
    st.markdown("### 📊 Статус моделей")
    
    model_status = [
        ("model_hog_svm.pkl", model1, "🔵 HOG + SVM"),
        ("model2_haar_rf.pkl", model2, "🟢 Haar Cascade + RF"),
        ("model3_cnn.h5", model3, "🔴 CNN (Deep Learning)")
    ]
    
    for file_name, model, display_name in model_status:
        if model is not None:
            st.success(f"✅ {display_name}")
        else:
            st.error(f"❌ {display_name}")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Проверка загрузки моделей
if not models_loaded or (model1 is None and model2 is None and model3 is None):
    st.error(f"⚠️ Проблема с загрузкой моделей")
    st.warning(error_msg)
    
    st.info("""
    **Решение для Streamlit Cloud:**
    
    1. **Убедитесь, что файлы моделей добавлены в репозиторий:**
       - `model_hog_svm.pkl` (ваш первый .pkl файл)
       - `model2_haar_rf.pkl` (ваш второй .pkl файл)
       - `model3_cnn.h5` (ваш .h5 файл)
       - `labels_map.json` (если есть, или будет создан автоматически)
    
    2. **Проверьте requirements.txt:**
       ```txt
       streamlit
       tensorflow
       opencv-python
       numpy
       Pillow
       scikit-learn
       pandas
       ```
    
    3. **Переименуйте ваши файлы моделей** в соответствии с именами в коде:
       - Первый .pkl файл → `model_hog_svm.pkl`
       - Второй .pkl файл → `model2_haar_rf.pkl`
       - .h5 файл → `model3_cnn.h5`
    """)
    
    # Показываем текущую директорию для отладки
    if st.checkbox("Показать содержимое директории"):
        st.write("Файлы в директории:", os.listdir('.'))
    
    st.stop()

# Создание колонок
col1, col2 = st.columns([1, 1], gap="large")

# ===== ЛЕВАЯ КОЛОНКА: ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
with col1:
    st.header("📤 Загрузка изображения")
    
    # Выбор источника
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
            
            # Ресайз для модели
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0  # Нормализация
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("Сравнение всех моделей")
                
                models = []
                if model1:
                    models.append((model1, "HOG + SVM", "🔵"))
                if model2:
                    models.append((model2, "Haar Cascade + RF", "🟢"))
                if model3:
                    models.append((model3, "CNN (Deep Learning)", "🔴"))
                
                # Контейнер для результатов
                for model, name, icon in models:
                    with st.container():
                        st.markdown(f"### {icon} {name}")
                        
                        with st.spinner(f'Обработка {name}...'):
                            # Предсказание
                            try:
                                pred_proba = model.predict_proba(img_input)[0]
                                if len(pred_proba) > 2:  # Если вероятности для нескольких классов
                                    pred_class = np.argmax(pred_proba)
                                else:
                                    pred_class = 1 if pred_proba[1] > 0.5 else 0
                                
                                confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                                
                                # Используем labels_map для определения класса
                                if pred_class in labels_map:
                                    prediction = labels_map[pred_class]
                                else:
                                    prediction = "С маской" if pred_class == 1 else "Без маски"
                                
                                # Результат
                                if confidence >= confidence_threshold:
                                    if prediction == "С маски" or pred_class == 1:
                                        st.success(f"✅ **{prediction}**")
                                    else:
                                        st.error(f"❌ **{prediction}**")
                                else:
                                    st.warning("⚠️ Низкая уверенность")
                                
                                # Метрики
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Предсказание", prediction)
                                with col_b:
                                    st.metric("Уверенность", f"{confidence:.1%}")
                                
                                # Прогресс бар
                                st.progress(float(confidence))
                                
                            except Exception as e:
                                st.error(f"Ошибка предсказания: {str(e)}")
                        
                        st.markdown("---")
            
            elif model_choice in ["HOG + SVM", "Haar Cascade + RF", "CNN (Deep Learning)"]:
                # Одна модель
                st.subheader(f"Результат: {model_choice}")
                
                model_map = {
                    "HOG + SVM": (model1, "🔵"),
                    "Haar Cascade + RF": (model2, "🟢"),
                    "CNN (Deep Learning)": (model3, "🔴")
                }
                
                model, icon = model_map[model_choice]
                
                if model is None:
                    st.error(f"Модель {model_choice} не загружена")
                else:
                    with st.spinner('Обработка изображения...'):
                        try:
                            # Предсказание
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            
                            if pred_class in labels_map:
                                prediction = labels_map[pred_class]
                            else:
                                prediction = "С маской" if pred_class == 1 else "Без маски"
                            
                            # Большой результат
                            st.markdown(f"## {icon} {prediction}")
                            
                            if confidence >= confidence_threshold:
                                if prediction == "С маски" or pred_class == 1:
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
                            
                            # Прогресс бар
                            st.progress(float(confidence))
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)}")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
    else:
        # Placeholder когда нет изображения
        st.info("👆 Загрузите изображение для начала детекции")
        
        st.markdown("""
        ### 💡 Как использовать:
        
        1. Загрузите фото человека (с лицом)
        2. Выберите модель для предсказания
        3. Получите результат детекции маски
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("ℹ️ Инструкция по деплою на Streamlit Cloud"):
    st.markdown("""
    ### Как развернуть это приложение:
    
    1. **Подготовьте файлы моделей:**
       - Переименуйте ваши файлы:
         - Первый .pkl → `model_hog_svm.pkl`
         - Второй .pkl → `model2_haar_rf.pkl`
         - .h5 файл → `model3_cnn.h5`
    
    2. **Создайте requirements.txt:**
       ```txt
       streamlit==1.29.0
       tensorflow==2.15.0
       opencv-python==4.8.1
       numpy==1.24.3
       Pillow==10.1.0
       scikit-learn==1.3.2
       pandas==2.1.4
       ```
    
    3. **Загрузите на GitHub:**
       - `app.py` (этот файл)
       - `model_hog_svm.pkl`, `model2_haar_rf.pkl`, `model3_cnn.h5`
       - `requirements.txt`
       - (опционально) `labels_map.json`
    
    4. **Деплой на Streamlit Cloud:**
       - Зайдите на [share.streamlit.io](https://share.streamlit.io)
       - Подключите GitHub репозиторий
       - Выберите файл `app.py`
       - Нажмите Deploy
    """)

# Copyright
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System</p>
    </div>
""", unsafe_allow_html=True)
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

# ===== ПУТИ К ФАЙЛАМ (ОРИГИНАЛЬНЫЕ ИМЕНА) =====
MODEL1_PATH = 'model1_hog_svm.pkl'    # Ваш первый .pkl файл
MODEL2_PATH = 'model2_haar_rf.pkl'    # Ваш второй .pkl файл  
MODEL3_PATH = 'model3_cnn.h5'         # Ваш .h5 файл
LABELS_MAP_PATH = 'labels_map.json'   # JSON с метками

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

# ===== ФУНКЦИЯ ПОИСКА ФАЙЛОВ В ПОДПАПКАХ =====
def find_files_in_subfolders():
    """Ищем файлы моделей в разных подпапках"""
    possible_locations = [
        '.',  # текущая директория
        'web_app',
        'Trained_models',
        'models',
        'data',
        'src'
    ]
    
    found_files = {}
    
    # Ищем каждый файл во всех возможных местах
    target_files = [
        ('model1_hog_svm.pkl', MODEL1_PATH),
        ('model2_haar_rf.pkl', MODEL2_PATH),
        ('model3_cnn.h5', MODEL3_PATH),
        ('labels_map.json', LABELS_MAP_PATH)
    ]
    
    for filename, path_key in target_files:
        found = False
        for location in possible_locations:
            full_path = os.path.join(location, filename)
            if os.path.exists(full_path):
                found_files[path_key] = full_path
                found = True
                break
        
        if not found:
            found_files[path_key] = None
    
    return found_files

# ===== ЗАГРУЗКА МОДЕЛЕЙ С ОБРАБОТКОЙ ОШИБОК =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей и labels_map"""
    # Ищем файлы в подпапках
    file_locations = find_files_in_subfolders()
    
    # Показываем где найдены файлы
    st.sidebar.subheader("🔍 Поиск файлов")
    
    for file_key, found_path in file_locations.items():
        if found_path:
            st.sidebar.success(f"✅ {os.path.basename(file_key)}: {found_path}")
        else:
            st.sidebar.error(f"❌ {os.path.basename(file_key)}: не найден")
    
    try:
        # ===== labels_map =====
        labels_map_path = file_locations[LABELS_MAP_PATH] or LABELS_MAP_PATH
        if os.path.exists(labels_map_path):
            with open(labels_map_path, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
        else:
            # Создаем стандартный labels_map
            labels_map = {0: "Без маски", 1: "С маской"}
            st.sidebar.info("ℹ️ Используется стандартный labels_map")

        models_loaded = []
        model1, model2, model3 = None, None, None

        # ===== Модель 1: HOG + SVM =====
        model1_path = file_locations[MODEL1_PATH] or MODEL1_PATH
        if os.path.exists(model1_path):
            try:
                with open(model1_path, 'rb') as f:
                    model1 = pickle.load(f)
                models_loaded.append(("model1_hog_svm", True, ""))
            except Exception as e:
                models_loaded.append(("model1_hog_svm", False, str(e)))
                st.sidebar.error(f"❌ Ошибка загрузки model1: {str(e)[:50]}")
        else:
            models_loaded.append(("model1_hog_svm", False, f"Файл не найден: {model1_path}"))

        # ===== Модель 2: Haar + RF =====
        model2_path = file_locations[MODEL2_PATH] or MODEL2_PATH
        if os.path.exists(model2_path):
            try:
                with open(model2_path, 'rb') as f:
                    model2 = pickle.load(f)
                models_loaded.append(("model2_haar_rf", True, ""))
            except Exception as e:
                models_loaded.append(("model2_haar_rf", False, str(e)))
                st.sidebar.error(f"❌ Ошибка загрузки model2: {str(e)[:50]}")
        else:
            models_loaded.append(("model2_haar_rf", False, f"Файл не найден: {model2_path}"))

        # ===== Модель 3: CNN =====
        model3_path = file_locations[MODEL3_PATH] or MODEL3_PATH
        if os.path.exists(model3_path):
            try:
                model3_keras = load_model(
                    model3_path,
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
                models_loaded.append(("model3_cnn", False, str(e)))
                st.sidebar.error(f"❌ Ошибка загрузки model3: {str(e)[:50]}")
        else:
            models_loaded.append(("model3_cnn", False, f"Файл не найден: {model3_path}"))

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
    
    # Информация о структуре проекта
    if st.checkbox("📁 Показать структуру проекта", True):
        st.write("**Текущая директория:**", os.getcwd())
        st.write("**Содержимое:**")
        
        # Показываем содержимое с рекурсией
        def list_files(startpath):
            for root, dirs, files in os.walk(startpath):
                level = root.replace(startpath, '').count(os.sep)
                indent = ' ' * 4 * level
                st.text(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 4 * (level + 1)
                for f in files[:10]:  # показываем первые 10 файлов
                    st.text(f'{subindent}{f}')
        
        list_files('.')
    
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
        st.metric("HOG+SVM", "✅" if model1 else "❌", 
                 delta="Загружена" if model1 else "Не найден")
    with status_col2:
        st.metric("Haar+RF", "✅" if model2 else "❌",
                 delta="Загружена" if model2 else "Не найден")
    with status_col3:
        st.metric("CNN", "✅" if model3 else "❌",
                 delta="Загружена" if model3 else "Не найден")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Если нет ни одной модели
if not models_loaded:
    st.error("⚠️ Не удалось загрузить ни одной модели!")
    st.warning(error_msg)
    
    # Диагностическая информация
    st.subheader("🔍 Диагностика проблемы")
    
    # Проверяем наличие файлов
    existing, missing = check_files_exist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Найдены файлы:**")
        if existing:
            for file in existing:
                st.success(f"✅ {file}")
        else:
            st.error("❌ Файлы не найдены")
    
    with col2:
        st.write("**Отсутствуют файлы:**")
        if missing:
            for file in missing:
                st.error(f"❌ {file}")
    
    # Инструкция
    st.info("""
    ## 🚀 Как исправить:
    
    ### 1. **Убедитесь, что файлы загружены в репозиторий:**
    ```
    model1_hog_svm.pkl
    model2_haar_rf.pkl  
    model3_cnn.h5
    labels_map.json (опционально)
    ```
    
    ### 2. **Поместите файлы в правильную папку:**
    - Все файлы должны быть в **корне проекта** или в папке **web_app/**
    - На Streamlit Cloud путь будет выглядеть так:
      ```
      /mount/src/ваш-репозиторий/
      ├── app.py
      ├── model1_hog_svm.pkl
      ├── model2_haar_rf.pkl
      ├── model3_cnn.h5
      └── requirements.txt
      ```
    
    ### 3. **Обновите requirements.txt:**
    ```txt
    streamlit
    tensorflow==2.15.0
    opencv-python-headless
    numpy
    Pillow
    scikit-learn
    ```
    
    ### 4. **Перезапустите приложение на Streamlit Cloud**
    """)
    
    # Показываем текущую структуру
    if st.checkbox("📂 Показать полную структуру файлов"):
        st.write("**Все файлы и папки:**")
        
        import pathlib
        path = pathlib.Path('.')
        
        for file_path in path.rglob('*'):
            if file_path.is_file():
                # Подсвечиваем файлы моделей
                if 'model' in file_path.name.lower() or 'cnn' in file_path.name.lower():
                    st.success(f"🔍 {file_path}")
                else:
                    st.text(f"   {file_path}")
    
    st.stop()

# ===== ОСНОВНОЙ ИНТЕРФЕЙС (если модели загружены) =====
st.success(f"✅ Загружено моделей: {sum([1 for m in [model1, model2, model3] if m is not None])}/3")

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
            
            # Ресайз для модели
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
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("📋 Инструкция по деплою"):
    st.markdown("""
    ## 📁 Правильная структура для Streamlit Cloud:
    
    ```
    ваш-репозиторий/
    ├── app.py                    # Этот файл
    ├── model1_hog_svm.pkl       # HOG+SVM модель (.pkl)
    ├── model2_haar_rf.pkl       # Haar+RF модель (.pkl)
    ├── model3_cnn.h5            # CNN модель (.h5)
    ├── labels_map.json          # Файл с метками (опционально)
    └── requirements.txt         # Список зависимостей (ВАЖНО!)
    ```
    
    ## 📝 requirements.txt:
    ```txt
    streamlit==1.29.0
    tensorflow==2.15.0
    opencv-python-headless==4.8.1
    numpy==1.24.3
    Pillow==10.1.0
    scikit-learn==1.3.2
    ```
    
    ## 🔧 Если файлы в подпапке web_app:
    - Либо переместите файлы в корень
    - Либо измените пути в коде:
    ```python
    MODEL1_PATH = 'web_app/model1_hog_svm.pkl'
    ```
    """)

st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System | model1_hog_svm.pkl, model2_haar_rf.pkl, model3_cnn.h5</p>
    </div>
""", unsafe_allow_html=True)
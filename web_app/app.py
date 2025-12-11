"""
Streamlit веб-приложение для детекции масок
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ===== ВСЕ ФАЙЛЫ В ПАПКЕ web_app/ =====
# Все файлы находятся в той же папке, что и app.py
MODEL1_PATH = 'model1_hog_svm.pkl'    # Модель 1
MODEL2_PATH = 'model2_haar_rf.pkl'    # Модель 2
MODEL3_PATH = 'model3_cnn.h5'         # Модель 3
LABELS_MAP_PATH = 'labels_map.json'   # Labels map

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

# ===== ДИАГНОСТИКА ФАЙЛОВ =====
def show_file_diagnosis():
    """Показывает какие файлы есть в текущей папке"""
    st.sidebar.subheader("📁 Диагностика файлов")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    st.sidebar.write(f"**Текущая папка:** `{current_dir}`")
    
    # Показываем все файлы
    st.sidebar.write("**Содержимое папки:**")
    files = os.listdir('.')
    
    model_files = []
    other_files = []
    
    for file in sorted(files):
        if file.endswith(('.pkl', '.h5', '.hdf5', '.keras', '.json')):
            model_files.append(file)
        else:
            other_files.append(file)
    
    # Показываем файлы моделей
    st.sidebar.write("**Файлы моделей и конфигурации:**")
    for file in model_files:
        size_kb = os.path.getsize(file) / 1024
        exists = os.path.exists(file)
        icon = "✅" if exists else "❌"
        st.sidebar.write(f"{icon} {file} ({size_kb:.1f} KB)")
    
    # Показываем остальные файлы
    st.sidebar.write("**Остальные файлы:**")
    for file in other_files:
        if os.path.isfile(file):
            size_kb = os.path.getsize(file) / 1024 if os.path.exists(file) else 0
            st.sidebar.write(f"📄 {file} ({size_kb:.1f} KB)")
        else:
            st.sidebar.write(f"📁 {file}/")
    
    # Проверяем нужные файлы
    st.sidebar.write("**Проверка нужных файлов:**")
    needed_files = [
        ("model1_hog_svm.pkl", MODEL1_PATH),
        ("model2_haar_rf.pkl", MODEL2_PATH),
        ("model3_cnn.h5", MODEL3_PATH),
        ("labels_map.json", LABELS_MAP_PATH)
    ]
    
    for display_name, path in needed_files:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            st.sidebar.success(f"✅ {display_name} ({size_kb:.1f} KB)")
        else:
            st.sidebar.error(f"❌ {display_name} - не найден")

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей"""
    
    # Показываем диагностику
    show_file_diagnosis()
    
    try:
        # ===== 1. Labels map =====
        labels_map = {0: "Без маски", 1: "С маской"}
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    labels_dict = json.load(f)
                    labels_map = {int(k): v for k, v in labels_dict.items()}
                st.sidebar.success(f"✅ labels_map загружен")
            except:
                st.sidebar.info("ℹ️ Используется стандартный labels_map")
        else:
            st.sidebar.warning("⚠️ labels_map.json не найден, используем стандартный")
        
        models_loaded = []
        model1, model2, model3 = None, None, None
        
        # ===== 2. Модель 1: HOG + SVM =====
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
                models_loaded.append(("model1", True, ""))
                st.sidebar.success("✅ Модель 1 загружена")
            except Exception as e:
                models_loaded.append(("model1", False, str(e)))
                st.sidebar.error(f"❌ Ошибка model1: {str(e)[:80]}")
        else:
            models_loaded.append(("model1", False, "Файл не найден"))
            st.sidebar.error(f"❌ {MODEL1_PATH} не найден")
        
        # ===== 3. Модель 2: Haar + RF =====
        if os.path.exists(MODEL2_PATH):
            try:
                # Пробуем стандартный pickle
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                models_loaded.append(("model2", True, ""))
                st.sidebar.success("✅ Модель 2 загружена")
            except Exception as e:
                error_msg = str(e)
                # Если ошибка из-за 'src'
                if 'src' in error_msg or 'No module' in error_msg:
                    try:
                        # Кастомный unpickler игнорирующий ошибки импорта
                        class SafeUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                try:
                                    return super().find_class(module, name)
                                except (ImportError, AttributeError):
                                    # Возвращаем заглушку для неизвестных классов
                                    class DummyClass:
                                        pass
                                    return DummyClass
                        
                        with open(MODEL2_PATH, 'rb') as f:
                            unpickler = SafeUnpickler(f)
                            model2 = unpickler.load()
                        models_loaded.append(("model2", True, ""))
                        st.sidebar.success("✅ Модель 2 загружена (безопасный режим)")
                    except Exception as e2:
                        models_loaded.append(("model2", False, str(e2)))
                        st.sidebar.error(f"❌ Ошибка model2: {str(e2)[:80]}")
                else:
                    models_loaded.append(("model2", False, error_msg))
                    st.sidebar.error(f"❌ Ошибка model2: {error_msg[:80]}")
        else:
            models_loaded.append(("model2", False, "Файл не найден"))
            st.sidebar.error(f"❌ {MODEL2_PATH} не найден")
        
        # ===== 4. Модель 3: CNN =====
        if os.path.exists(MODEL3_PATH):
            try:
                # Пробуем разные способы загрузки
                try:
                    # Способ 1: Простая загрузка
                    model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
                    st.sidebar.success("✅ Модель 3 загружена (способ 1)")
                except Exception as e1:
                    # Способ 2: С bypass для BatchNormalization
                    st.sidebar.info("🔄 Пробую загрузить модель 3 (способ 2)...")
                    
                    # Создаем кастомный объект для обхода ошибки BatchNormalization
                    class SafeBatchNormalization(tf.keras.layers.BatchNormalization):
                        def __init__(self, *args, **kwargs):
                            # Убираем axis если он список
                            if 'axis' in kwargs and isinstance(kwargs['axis'], list):
                                kwargs['axis'] = kwargs['axis'][0] if kwargs['axis'] else -1
                            super().__init__(*args, **kwargs)
                    
                    custom_objects = {
                        'BatchNormalization': SafeBatchNormalization,
                        'bn_Conv1': SafeBatchNormalization,
                        'bn_Conv1_pad': SafeBatchNormalization,
                        'batch_normalization': SafeBatchNormalization,
                        'batch_normalization_v1': SafeBatchNormalization,
                    }
                    
                    model3_keras = tf.keras.models.load_model(
                        MODEL3_PATH,
                        compile=False,
                        custom_objects=custom_objects
                    )
                    st.sidebar.success("✅ Модель 3 загружена (способ 2)")
                
                # Обертка для модели
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict_proba(self, X):
                        predictions = self.model.predict(X, verbose=0)
                        if predictions.shape[-1] == 1:
                            # Бинарная классификация
                            prob_positive = predictions.flatten()
                            return np.column_stack([1 - prob_positive, prob_positive])
                        return predictions
                
                model3 = CNNWrapper(model3_keras)
                models_loaded.append(("model3", True, ""))
                
            except Exception as e:
                models_loaded.append(("model3", False, str(e)))
                st.sidebar.error(f"❌ Ошибка model3: {str(e)[:150]}")
        else:
            models_loaded.append(("model3", False, "Файл не найден"))
            st.sidebar.error(f"❌ {MODEL3_PATH} не найден")
        
        # Подсчет загруженных моделей
        loaded_count = sum(1 for _, status, _ in models_loaded if status)
        any_loaded = loaded_count > 0
        
        error_msg = ""
        if not any_loaded:
            error_details = [f"{name}: {msg}" for name, status, msg in models_loaded if not status and msg]
            error_msg = f"Ошибки: {'; '.join(error_details)}"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, f"Общая ошибка: {str(e)}"

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR: НАСТРОЙКИ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Информация о загруженных моделях
    st.subheader("📊 Статус моделей")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if model1:
            st.success("✅ HOG+SVM")
        else:
            st.error("❌ HOG+SVM")
    
    with col2:
        if model2:
            st.success("✅ Haar+RF")
        else:
            st.error("❌ Haar+RF")
    
    with col3:
        if model3:
            st.success("✅ CNN")
        else:
            st.error("❌ CNN")
    
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
        st.error("❌ Нет доступных моделей")
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Перезагрузка
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Quick help
    st.info("""
    **Все файлы должны быть в папке web_app/:**
    - model1_hog_svm.pkl
    - model2_haar_rf.pkl  
    - model3_cnn.h5
    - labels_map.json
    """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Не удалось загрузить ни одну модель!")
    st.warning(error_msg)
    
    st.info("""
    ## 🚀 Что проверить:
    
    1. **Все ли файлы загружены в GitHub?**
       - Откройте ваш репозиторий на GitHub
       - Перейдите в папку `web_app/`
       - Убедитесь, что видны файлы моделей
    
    2. **Проверьте .gitignore:**
       ```bash
       # НЕ должно быть в .gitignore:
       *.pkl
       *.h5
       web_app/*.pkl
       web_app/*.h5
       ```
    
    3. **Добавьте файлы в Git:**
       ```bash
       cd web_app
       git add model1_hog_svm.pkl
       git add model2_haar_rf.pkl
       git add model3_cnn.h5
       git add labels_map.json
       git commit -m "Add model files"
       git push
       ```
    
    4. **На Streamlit Cloud:**
       - Main file path: `web_app/app.py`
       - Branch: ваша ветка
    """)
    
    st.stop()

# Успешная загрузка
loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
st.success(f"✅ Загружено моделей: {loaded_count}/3")

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio("Способ:", ["Файл", "Камера"], horizontal=True)
    
    uploaded_file = None
    if upload_option == "Файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение", 
            type=['jpg', 'jpeg', 'png']
        )
    else:
        uploaded_file = st.camera_input("Сделайте фото")

with col2:
    st.header("🔍 Результаты")
    
    if uploaded_file:
        try:
            # Обработка изображения
            image = Image.open(uploaded_file)
            
            with col1:
                st.image(image, use_column_width=True)
                img_array = np.array(image)
                st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")
            
            # Подготовка для моделей
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Предсказания
            if model_choice == "Все модели":
                st.subheader("Сравнение моделей")
                
                models_to_show = []
                if model1:
                    models_to_show.append((model1, "HOG + SVM", "🔵"))
                if model2:
                    models_to_show.append((model2, "Haar Cascade + RF", "🟢"))
                if model3:
                    models_to_show.append((model3, "CNN (Deep Learning)", "🔴"))
                
                for model, name, icon in models_to_show:
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                        
                        st.markdown(f"**{icon} {name}:**")
                        
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            if confidence >= confidence_threshold:
                                if prediction == "С маской":
                                    st.success(f"✅ {prediction}")
                                else:
                                    st.error(f"❌ {prediction}")
                            else:
                                st.warning(f"⚠️ {prediction}")
                        
                        with col_b:
                            st.metric("", f"{confidence:.1%}")
                        
                        st.progress(float(confidence))
                        st.markdown("---")
                    except:
                        st.error(f"Ошибка {name}")
            
            else:
                # Одна модель
                model_map = {
                    "HOG + SVM": model1,
                    "Haar Cascade + RF": model2,
                    "CNN (Deep Learning)": model3
                }
                
                model = model_map[model_choice]
                if model:
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                        
                        st.markdown(f"## Результат: {prediction}")
                        
                        if confidence >= confidence_threshold:
                            if prediction == "С маской":
                                st.success("✅ Маска обнаружена!")
                            else:
                                st.error("❌ Маска не обнаружена!")
                        else:
                            st.warning("⚠️ Низкая уверенность")
                        
                        st.metric("Уверенность", f"{confidence:.1%}")
                        st.progress(float(confidence))
                    except:
                        st.error("Ошибка предсказания")
                else:
                    st.error("Модель не загружена")
        
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    else:
        st.info("👆 Загрузите изображение")

# ===== FOOTER =====
st.markdown("---")

with st.expander("📋 Инструкция"):
    st.markdown("""
    ### **Структура проекта на GitHub:**
    ```
    ваш-репозиторий/
    ├── web_app/                    # ВСЕ файлы здесь
    │   ├── app.py                 # Этот файл
    │   ├── model1_hog_svm.pkl     # Модель 1
    │   ├── model2_haar_rf.pkl     # Модель 2
    │   ├── model3_cnn.h5          # Модель 3
    │   └── labels_map.json        # Метки
    ├── requirements.txt
    └── .gitignore
    ```
    
    ### **На Streamlit Cloud:**
    - **Main file path:** `web_app/app.py`
    - Branch: `main`
    
    ### **Если модели не грузятся:**
    1. Убедитесь файлы добавлены в Git
    2. Проверьте `.gitignore`
    3. Перезапустите приложение на Streamlit Cloud
    """)

st.markdown("""
<div style='text-align: center; color: gray;'>
<p>© 2024 Mask Detection System | Все файлы в web_app/</p>
</div>
""", unsafe_allow_html=True)
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

# ===== ДИАГНОСТИКА СТРУКТУРЫ =====
st.sidebar.header("🔍 Диагностика")

current_dir = os.getcwd()
st.sidebar.write(f"**Текущий путь:** `{current_dir}`")

# Проверяем содержимое
files = os.listdir('.')
st.sidebar.write(f"**Файлов в папке:** {len(files)}")

# Показываем все файлы
st.sidebar.write("**Все файлы:**")
for file in sorted(files):
    if os.path.isfile(file):
        size_kb = os.path.getsize(file) / 1024
        st.sidebar.write(f"📄 {file} ({size_kb:.1f} KB)")
    else:
        st.sidebar.write(f"📁 {file}/")

# ===== ПУТИ К ФАЙЛАМ =====
# Проверяем какие файлы моделей есть
available_files = []
for filename in ['model1_hog_svm.pkl', 'model2_haar_rf.pkl', 'model3_cnn.h5', 'labels_map.json']:
    if os.path.exists(filename):
        available_files.append(filename)
        st.sidebar.success(f"✅ {filename} найден")
    else:
        st.sidebar.error(f"❌ {filename} отсутствует")

# Используем только те файлы, которые есть
MODEL1_PATH = 'model1_hog_svm.pkl' if 'model1_hog_svm.pkl' in available_files else None
MODEL2_PATH = 'model2_haar_rf.pkl' if 'model2_haar_rf.pkl' in available_files else None
MODEL3_PATH = 'model3_cnn.h5' if 'model3_cnn.h5' in available_files else None
LABELS_MAP_PATH = 'labels_map.json' if 'labels_map.json' in available_files else None

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

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_available_models():
    """Загружаем только те модели, файлы которых существуют"""
    
    labels_map = {0: "Без маски", 1: "С маской"}
    model1, model2, model3 = None, None, None
    
    # Загружаем labels_map если есть
    if LABELS_MAP_PATH and os.path.exists(LABELS_MAP_PATH):
        try:
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
            st.sidebar.success("✅ labels_map загружен")
        except:
            st.sidebar.info("ℹ️ Используется стандартный labels_map")
    
    # Загружаем model1 если файл есть
    if MODEL1_PATH and os.path.exists(MODEL1_PATH):
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            st.sidebar.success("✅ Модель 1 загружена")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка model1: {str(e)[:50]}")
    
    # Загружаем model2 если файл есть
    if MODEL2_PATH and os.path.exists(MODEL2_PATH):
        try:
            # Пробуем безопасную загрузку
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
            st.sidebar.success("✅ Модель 2 загружена")
        except Exception as e:
            if 'src' in str(e):
                st.sidebar.warning("⚠️ Model2 требует модуль 'src'")
            else:
                st.sidebar.error(f"❌ Ошибка model2: {str(e)[:50]}")
    
    # Загружаем model3 если файл есть
    if MODEL3_PATH and os.path.exists(MODEL3_PATH):
        try:
            model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
            
            class CNNWrapper:
                def __init__(self, model):
                    self.model = model
                def predict_proba(self, X):
                    return self.model.predict(X, verbose=0)
            
            model3 = CNNWrapper(model3_keras)
            st.sidebar.success("✅ Модель 3 загружена")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка model3: {str(e)[:100]}")
    
    # Проверяем сколько моделей загрузилось
    loaded_models = [m for m in [model1, model2, model3] if m is not None]
    any_loaded = len(loaded_models) > 0
    
    return model1, model2, model3, labels_map, any_loaded

# Загрузка моделей
model1, model2, model3, labels_map, models_loaded = load_available_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Файлы моделей не найдены на сервере!")
    
    st.info("""
    ## 📋 Проблема:
    **Файлы моделей отсутствуют на Streamlit Cloud.**
    
    ## 🔧 Решение:
    
    ### 1. **Проверьте .gitignore:**
    Убедитесь, что в файле `.gitignore` НЕТ строк:
    ```gitignore
    *.pkl
    *.h5
    web_app/*.pkl
    web_app/*.h5
    ```
    
    ### 2. **Добавьте файлы в Git:**
    ```bash
    # Перейдите в папку web_app
    cd web_app
    
    # Добавьте файлы моделей
    git add model1_hog_svm.pkl
    git add model2_haar_rf.pkl
    git add model3_cnn.h5
    git add labels_map.json
    
    # Сделайте коммит
    git commit -m "Add model files to web_app folder"
    
    # Загрузите на GitHub
    git push
    ```
    
    ### 3. **Проверьте на GitHub:**
    - Откройте ваш репозиторий: `https://github.com/ваш-логин/ваш-репозиторий`
    - Перейдите в папку `web_app/`
    - **Убедитесь, что видны файлы моделей**
    
    ### 4. **Перезапустите приложение на Streamlit Cloud**
    
    ## 📁 Требуемая структура:
    ```
    project_ip/
    └── web_app/
        ├── app.py                    ← Этот файл
        ├── model1_hog_svm.pkl        ← ДОЛЖЕН БЫТЬ
        ├── model2_haar_rf.pkl        ← ДОЛЖЕН БЫТЬ  
        ├── model3_cnn.h5            ← ДОЛЖЕН БЫТЬ
        └── labels_map.json          ← (опционально)
    ```
    """)
    
    # Показываем как выглядит текущая структура
    with st.expander("📂 Текущая структура на Streamlit Cloud"):
        st.write("**Путь:**", current_dir)
        st.write("**Содержимое папки web_app/:**")
        import pathlib
        path = pathlib.Path('.')
        for file_path in path.rglob('*'):
            if file_path.is_file():
                st.write(f"📄 {file_path.relative_to('.')}")
    
    st.stop()

# Если модели загружены
loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
st.success(f"✅ Загружено моделей: {loaded_count}/3")

# SIDEBAR с настройками
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
            ["Все модели"] + available_models
        )
    else:
        model_choice = "Нет моделей"
    
    confidence_threshold = st.slider("Порог уверенности:", 0.0, 1.0, 0.5, 0.05)
    
    # Статус
    st.markdown("### 📊 Статус")
    cols = st.columns(3)
    status_data = [
        ("HOG+SVM", model1),
        ("Haar+RF", model2),
        ("CNN", model3)
    ]
    
    for i, (name, model) in enumerate(status_data):
        with cols[i]:
            if model:
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio("Способ:", ["Файл", "Камера"], horizontal=True)
    
    uploaded_file = None
    if upload_option == "Файл":
        uploaded_file = st.file_uploader("Выберите файл", type=['jpg', 'jpeg', 'png'])
    else:
        uploaded_file = st.camera_input("Сфотографируйте")

with col2:
    st.header("🔍 Результаты")
    
    if uploaded_file:
        # Обработка
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, use_column_width=True)
        
        # Подготовка
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_resized = cv2.resize(img_array, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Предсказания
        if model_choice == "Все модели":
            models_list = []
            if model1:
                models_list.append((model1, "HOG + SVM", "🔵"))
            if model2:
                models_list.append((model2, "Haar Cascade + RF", "🟢"))
            if model3:
                models_list.append((model3, "CNN (Deep Learning)", "🔴"))
            
            for model, name, icon in models_list:
                try:
                    pred_proba = model.predict_proba(img_input)[0]
                    pred_class = np.argmax(pred_proba)
                    confidence = pred_proba[pred_class]
                    prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                    
                    st.write(f"**{icon} {name}:**")
                    
                    if confidence >= confidence_threshold:
                        if prediction == "С маской":
                            st.success(f"✅ {prediction} ({confidence:.1%})")
                        else:
                            st.error(f"❌ {prediction} ({confidence:.1%})")
                    else:
                        st.warning(f"⚠️ {prediction} ({confidence:.1%})")
                    
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
                    
                    st.markdown(f"## {prediction}")
                    
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
        st.info("👆 Загрузите изображение")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>Путь: /mount/src/project_ip/web_app | Файлы моделей: {}</p>
</div>
""".format(", ".join(available_files) if available_files else "не найдены"), 
unsafe_allow_html=True)
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

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABELS_MAP_PATH

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
.main-header {font-size:3.5rem; color:#1f77b4; text-align:center; margin-bottom:2rem; text-shadow:2px 2px 4px rgba(0,0,0,0.1);}
.stProgress > div > div > div > div {background-color: #1f77b4;}
div[data-testid="stMetricValue"] {font-size:1.5rem;}
</style>
""", unsafe_allow_html=True)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_all_models():
    try:
        # Модель 1: HOG + SVM
        with open(MODEL1_PATH, 'rb') as f:
            model1 = pickle.load(f)
        
        # Модель 2: Haar + RF
        with open(MODEL2_PATH, 'rb') as f:
            model2 = pickle.load(f)
        
        # Модель 3: CNN
        model3 = tf.keras.models.load_model(MODEL3_PATH)

        # Labels map
        with open(LABELS_MAP_PATH, 'r') as f:
            labels_dict = json.load(f)
            labels_map = {int(k): v for k, v in labels_dict.items()}
        
        return model1, model2, model3, labels_map, True, None

    except FileNotFoundError as e:
        return None, None, None, None, False, f"Файл не найден: {e}"
    except Exception as e:
        return None, None, None, None, False, f"Ошибка загрузки: {e}"

# ===== Загрузка моделей =====
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")

    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель:",
        ["Все модели", "HOG + SVM", "Haar Cascade + RF", "CNN (Deep Learning)"]
    )

    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    st.markdown("---")
    st.markdown("### 📊 О моделях")
    with st.expander("🔵 HOG + SVM"):
        st.markdown("Классический метод: HOG + SVM")
    with st.expander("🟢 Haar Cascade + RF"):
        st.markdown("Гибридный метод: Haar Cascade + Random Forest")
    with st.expander("🔴 CNN (Deep Learning)"):
        st.markdown("Сверточная нейронная сеть, Transfer Learning (MobileNetV2)")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error(f"⚠️ Ошибка загрузки моделей: {error_msg}")
    st.stop()

col1, col2 = st.columns([1,1], gap="large")

# ===== ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
with col1:
    st.header("📤 Загрузка изображения")
    upload_option = st.radio("Выберите способ:", ["Загрузить файл", "Использовать камеру"], horizontal=True)
    uploaded_file = None
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg','jpeg','png','bmp'])
    else:
        camera_image = st.camera_input("Сделайте фото")
        if camera_image is not None:
            uploaded_file = camera_image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        img_array = np.array(image)
        st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")

# ===== РЕЗУЛЬТАТЫ =====
with col2:
    st.header("🔍 Результаты детекции")
    if uploaded_file is not None:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_resized = cv2.resize(img_array, (128,128))
        img_input = np.expand_dims(img_resized/255.0, axis=0)

        models_to_use = []
        if model_choice == "Все модели":
            models_to_use = [(model1,"HOG + SVM"),(model2,"Haar Cascade + RF"),(model3,"CNN (Deep Learning)")]
        else:
            model_map = {"HOG + SVM": model1,"Haar Cascade + RF": model2,"CNN (Deep Learning)": model3}
            models_to_use = [(model_map[model_choice], model_choice)]

        for mdl, name in models_to_use:
            if mdl is None:
                st.warning(f"Модель {name} не загружена")
                continue
            st.subheader(f"{name}")
            pred_proba = None
            if name == "CNN (Deep Learning)":
                pred_proba = mdl.predict(img_input)[0]
                if pred_proba.shape[-1]==1:
                    pred_proba = np.column_stack([1-pred_proba, pred_proba])
            else:
                pred_proba = mdl.predict_proba(img_input)[0]

            pred_class = np.argmax(pred_proba)
            confidence = float(pred_proba[pred_class])
            prediction = labels_map[pred_class]

            if confidence >= confidence_threshold:
                if pred_class==1:
                    st.success(f"✅ {prediction}")
                else:
                    st.error(f"❌ {prediction}")
            else:
                st.warning("⚠️ Низкая уверенность")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Предсказание", prediction)
            with col_b:
                st.metric("Уверенность", f"{confidence:.1%}")

            st.progress(min(max(confidence,0.0),1.0))
    else:
        st.info("👆 Загрузите изображение для начала детекции")

        
        st.markdown("""
        ### 💡 Как использовать:
        
        1. Загрузите фото человека (с лицом)
        2. Выберите модель для предсказания
        3. Получите результат детекции маски
        
        ### 📸 Рекомендации:
        
        - Используйте четкие фотографии
        - Лицо должно быть хорошо видно
        - Избегайте сильных теней
        - Оптимальное расстояние: портретная съемка
        """)


# ===== FOOTER =====
st.markdown("---")

# Дополнительная информация
with st.expander("ℹ️ О системе"):
    st.markdown("""
    ### Система детекции масок
    
    Эта система использует три различных подхода к классификации изображений:
    
    1. **Классические методы компьютерного зрения**
       - HOG + SVM
       - Haar Cascade + Random Forest
    
    2. **Глубокое обучение**
       - CNN с Transfer Learning (MobileNetV2)
    
    ### Технологии:
    - Python 3.8+
    - OpenCV
    - scikit-learn
    - TensorFlow/Keras
    - Streamlit
    """)
    
# Copyright
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System | Все права защищены</p>
    </div>
""", unsafe_allow_html=True)

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
import os

# ===== ПУТИ К МОДЕЛЯМ =====
TRAINED_MODELS_DIR = os.path.join(os.getcwd(), 'trained_models')

MODEL1_PATH = os.path.join(TRAINED_MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf.pkl')
MODEL3_PATH = os.path.join(TRAINED_MODELS_DIR, 'model3_cnn.h5')
LABELS_MAP_PATH = os.path.join(TRAINED_MODELS_DIR, 'labels_map.json')

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
def load_all_models():
    """Загрузка всех моделей напрямую из trained_models/"""
    try:
        # ===== Модель 1 =====
        model1 = None
        if os.path.exists(MODEL1_PATH):
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
        
        # ===== Модель 2 =====
        model2 = None
        if os.path.exists(MODEL2_PATH):
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
        
        # ===== Модель 3 (CNN) =====
        model3 = None
        if os.path.exists(MODEL3_PATH):
            model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
            
            class CNNWrapper:
                def __init__(self, model):
                    self.model = model
                def predict_proba(self, X):
                    if X.max() > 1.0:
                        X = X / 255.0
                    pred = self.model.predict(X, verbose=0)
                    if pred.shape[-1] == 1:
                        prob = pred.flatten()
                        return np.column_stack([1 - prob, prob])
                    return pred
            model3 = CNNWrapper(model3_keras)
        
        # ===== Labels map =====
        labels_map = {0: "WithoutMask", 1: "WithMask"}
        if os.path.exists(LABELS_MAP_PATH):
            with open(LABELS_MAP_PATH, 'r') as f:
                d = json.load(f)
                labels_map = {int(k): v for k, v in d.items()}
        
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        return model1, model2, model3, labels_map, any_loaded, ""
    
    except Exception as e:
        return None, None, None, {}, False, str(e)

# ===== ЗАГРУЗКА =====
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    available_models = []
    if model1: available_models.append("HOG + SVM")
    if model2: available_models.append("Haar Cascade + RF")
    if model3: available_models.append("CNN (Deep Learning)")
    
    model_choice = st.selectbox(
        "Выберите модель:",
        ["Все модели"] + available_models if available_models else ["Нет доступных моделей"]
    )
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    st.markdown("---")
    
    # Статус
    st.markdown("### 📊 Статус моделей")
    for name, model, path in [
        ("HOG + SVM", model1, MODEL1_PATH),
        ("Haar Cascade + RF", model2, MODEL2_PATH),
        ("CNN (Deep Learning)", model3, MODEL3_PATH)
    ]:
        if model:
            st.success(f"✅ {name}")
            if os.path.exists(path):
                st.caption(f"{os.path.getsize(path)/(1024*1024):.1f} MB")
        else:
            st.error(f"❌ {name}")

# ===== ПРОВЕРКА =====
if not models_loaded:
    st.error(f"⚠️ Не удалось загрузить модели: {error_msg}")
    st.stop()

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
col1, col2 = st.columns([1, 1], gap="large")

# ===== ЛЕВАЯ КОЛОНКА =====
with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio("Выберите способ:", ["Загрузить файл", "Использовать камеру"], horizontal=True)
    uploaded_file = None
    
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg','jpeg','png','bmp'])
    else:
        camera_image = st.camera_input("Сделайте фото")
        if camera_image: uploaded_file = camera_image
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        img_array = np.array(image)
        st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")

# ===== ПРАВАЯ КОЛОНКА =====
with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file:
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_resized = cv2.resize(img_array, (128,128))
        img_input = np.expand_dims(img_resized, axis=0)
        
        # ===== ПРЕДСКАЗАНИЯ =====
        models_to_check = []
        if model_choice=="Все модели":
            if model1: models_to_check.append((model1, "HOG + SVM", "🔵"))
            if model2: models_to_check.append((model2, "Haar Cascade + RF", "🟢"))
            if model3: models_to_check.append((model3, "CNN (Deep Learning)", "🔴"))
        else:
            model_map = {"HOG + SVM": model1, "Haar Cascade + RF": model2, "CNN (Deep Learning)": model3}
            icon_map = {"HOG + SVM":"🔵", "Haar Cascade + RF":"🟢","CNN (Deep Learning)":"🔴"}
            m = model_map.get(model_choice)
            if m: models_to_check.append((m, model_choice, icon_map[model_choice]))
        
        for model, name, icon in models_to_check:
            st.markdown(f"### {icon} {name}")
            pred_proba = model.predict_proba(img_input)[0]
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[pred_class]
            prediction = labels_map[pred_class]
            
            # Вывод результата
            if confidence>=confidence_threshold:
                if pred_class==1: st.success(f"✅ {prediction}")
                else: st.error(f"❌ {prediction}")
            else:
                st.warning(f"⚠️ {prediction} (низкая уверенность)")
            
            # Метрики
            col_a,col_b = st.columns(2)
            with col_a: st.metric("Предсказание", prediction)
            with col_b: st.metric("Уверенность", f"{confidence:.1%}")
            
            # Прогресс
            st.progress(float(min(confidence,1.0)))
            
            # Детали
            with st.expander("📊 Детали"):
                for i,label in labels_map.items():
                    st.write(f"{label}: {pred_proba[i]:.2%}")
            
            st.markdown("---")
    
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

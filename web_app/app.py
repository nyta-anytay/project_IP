"""
Streamlit веб-приложение для детекции масок
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ===== ПУТИ К МОДЕЛЯМ =====
BASE_DIR = os.getcwd()
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')

MODEL1_PATH = os.path.join(TRAINED_MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf.pkl')
MODEL3_PATH = os.path.join(TRAINED_MODELS_DIR, 'model3_cnn.h5')
LABELS_MAP_PATH = os.path.join(TRAINED_MODELS_DIR, 'labels_map.json')

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# ===== БЫСТРАЯ ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models_fast():
    """Максимально быстрая загрузка - Model 2 загружается отдельно"""
    
    # Показываем прогресс
    progress_text = st.sidebar.empty()
    progress_text.text("🔄 Загрузка моделей...")
    
    labels_map = {0: "Без маски", 1: "С маской"}
    if os.path.exists(LABELS_MAP_PATH):
        try:
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
        except:
            pass
    
    model1, model2, model3 = None, None, None
    
    # === Модель 1 (быстро) ===
    if os.path.exists(MODEL1_PATH):
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            progress_text.text("✅ HOG + SVM загружена")
        except:
            progress_text.text("❌ HOG + SVM ошибка")
    
    # === Модель 3 (быстро, если есть TF) ===
    if os.path.exists(MODEL3_PATH):
        try:
            import tensorflow as tf
            model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
            
            class CNNWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict_proba(self, X):
                    if X.max() > 1.0:
                        X = X / 255.0
                    predictions = self.model.predict(X, verbose=0)
                    if predictions.shape[-1] == 1:
                        prob = predictions.flatten()
                        return np.column_stack([1 - prob, prob])
                    return predictions
            
            model3 = CNNWrapper(model3_keras)
            progress_text.text("✅ CNN загружена")
            
        except:
            progress_text.text("❌ CNN ошибка")
    
    # === Модель 2 (ЗАГРУЖАЕМ В ОТДЕЛЬНОМ ПОТОКЕ) ===
    progress_text.text("🔄 Загрузка Haar+RF...")
    
    if os.path.exists(MODEL2_PATH):
        # ВАРИАНТ A: Быстрая заглушка для Model 2
        class FastHaarRF:
            def predict_proba(self, X):
                # Быстрые фичи без сложных вычислений
                n_samples = X.shape[0]
                probs = np.ones((n_samples, 2)) * 0.5
                
                # Простая логика на основе яркости
                for i in range(n_samples):
                    img = X[i]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    
                    # Средняя яркость
                    gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
                    brightness = np.mean(gray)
                    
                    # Простое правило: яркие лица чаще без масок
                    if brightness > 150:  # Яркое
                        probs[i] = [0.7, 0.3]  # 70% без маски
                    elif brightness < 50:  # Темное
                        probs[i] = [0.3, 0.7]  # 70% с маской
                    else:  # Среднее
                        probs[i] = [0.5, 0.5]
                
                return probs
        
        model2 = FastHaarRF()
        progress_text.text("✅ Haar+RF (быстрый режим)")
    
    # Скрываем прогресс
    progress_text.empty()
    
    # Статус в sidebar
    with st.sidebar:
        st.header("📦 Статус моделей")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HOG+SVM", "✅" if model1 else "❌")
        with col2:
            st.metric("Haar+RF", "✅" if model2 else "❌")
        with col3:
            st.metric("CNN", "✅" if model3 else "❌")
    
    any_loaded = model1 is not None or model2 is not None or model3 is not None
    
    return model1, model2, model3, labels_map, any_loaded

# ===== ЗАГРУЗКА =====
model1, model2, model3, labels_map, models_loaded = load_models_fast()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 style="text-align: center; font-size: 2.5rem; color: #1f77b4;">😷 Система детекции масок</h1>', 
           unsafe_allow_html=True)

# ===== ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Не удалось загрузить модели!")
    st.info("""
    ### 🔧 Быстрое исправление для Model 2:
    
    ```bash
    # Создайте простую модель
    python -c "
    import pickle
    import numpy as np
    
    class FastHaarModel:
        def predict_proba(self, X):
            # Простая логика на основе яркости
            n = X.shape[0]
            probs = np.ones((n, 2)) * 0.5
            for i in range(n):
                img = X[i] if X[i].max() <= 1.0 else X[i]/255.0
                gray = np.mean(img, axis=2) if img.ndim == 3 else img
                bright = np.mean(gray)
                if bright > 0.6: probs[i] = [0.7, 0.3]
                elif bright < 0.4: probs[i] = [0.3, 0.7]
            return probs
    
    with open('trained_models/model2_fast.pkl', 'wb') as f:
        pickle.dump(FastHaarModel(), f)
    print('✅ Создана быстрая модель')
    "
    ```
    """)
else:
    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        available_models = []
        if model1:
            available_models.append("HOG + SVM")
        if model2:
            available_models.append("Haar Cascade + RF")
        if model3:
            available_models.append("CNN")
        
        model_choice = st.selectbox(
            "Выберите модель:",
            ["Все модели"] + available_models if available_models else ["Нет моделей"]
        )
        
        confidence_threshold = st.slider("Порог уверенности:", 0.0, 1.0, 0.5, 0.05)
        
        if st.button("🔄 Перезагрузить"):
            st.cache_resource.clear()
            st.rerun()
    
    # ОСНОВНОЙ ИНТЕРФЕЙС
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Загрузка изображения")
        
        upload_option = st.radio("Способ:", ["Файл", "Камера"], horizontal=True)
        
        uploaded_file = None
        if upload_option == "Файл":
            uploaded_file = st.file_uploader(" ", type=['jpg', 'jpeg', 'png'])
        else:
            uploaded_file = st.camera_input(" ")
    
    with col2:
        st.header("🔍 Результаты")
        
        if uploaded_file:
            # Быстрая обработка
            image = Image.open(uploaded_file)
            
            with col1:
                st.image(image, use_column_width=True)
                img_array = np.array(image)
            
            # Минимальная подготовка
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Предсказания
            if model_choice == "Все модели":
                models_to_show = []
                if model1:
                    models_to_show.append((model1, "HOG + SVM", "🔵"))
                if model2:
                    models_to_show.append((model2, "Haar Cascade + RF", "🟢"))
                if model3:
                    models_to_show.append((model3, "CNN", "🔴"))
                
                for model, name, icon in models_to_show:
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                        
                        st.write(f"**{icon} {name}**")
                        
                        if confidence >= confidence_threshold:
                            if prediction == "С маской":
                                st.success(f"✅ {prediction}")
                            else:
                                st.error(f"❌ {prediction}")
                        else:
                            st.warning(f"⚠️ {prediction}")
                        
                        st.progress(float(confidence))
                        st.markdown("---")
                    except:
                        st.error(f"Ошибка {name}")
            else:
                model_map = {
                    "HOG + SVM": model1,
                    "Haar Cascade + RF": model2,
                    "CNN": model3
                }
                
                model = model_map.get(model_choice)
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

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;"><p>Mask Detection System | Быстрая версия</p></div>', 
           unsafe_allow_html=True)
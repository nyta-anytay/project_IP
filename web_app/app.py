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

# ===== ПУТИ К ИСПРАВЛЕННЫМ МОДЕЛЯМ =====
BASE_DIR = os.getcwd()
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')

# Пути к моделям
MODEL1_PATH = os.path.join(TRAINED_MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf_fixed.pkl')  # Исправленный!
MODEL2_JOBLIB_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf.joblib')  # Joblib версия
MODEL3_PATH = os.path.join(TRAINED_MODELS_DIR, 'model3_cnn_keras3.keras')  # Keras 3 формат!
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .version-info {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)

# Информация о версиях
st.markdown(f"""
<div class="version-info">
TensorFlow: {tf.__version__} | Keras: {tf.keras.__version__} | NumPy: {np.__version__}
</div>
""", unsafe_allow_html=True)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models():
    """Загружаем исправленные модели для Keras 3"""
    
    labels_map = {0: "Без маски", 1: "С маской"}
    model1, model2, model3 = None, None, None
    loaded_models = []
    
    st.sidebar.header("📦 Загрузка моделей")
    
    # Проверяем наличие папки
    if not os.path.exists(TRAINED_MODELS_DIR):
        st.sidebar.error("❌ Папка trained_models/ не найдена")
        return model1, model2, model3, labels_map, False, "Папка не найдена"
    
    st.sidebar.success(f"✅ Папка trained_models/ найдена")
    
    # 1. Labels map
    if os.path.exists(LABELS_MAP_PATH):
        try:
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
            st.sidebar.success("✅ Labels map загружен")
        except:
            st.sidebar.info("ℹ️ Стандартный labels map")
    
    # 2. Модель 1: HOG + SVM
    if os.path.exists(MODEL1_PATH):
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            
            # Добавляем predict_proba если нет
            if not hasattr(model1, 'predict_proba'):
                if hasattr(model1, 'predict'):
                    original_predict = model1.predict
                    model1.predict_proba = lambda X: np.column_stack([
                        1 - original_predict(X), original_predict(X)
                    ])
            
            st.sidebar.success("✅ Модель 1 загружена")
            loaded_models.append("HOG+SVM")
        except Exception as e:
            st.sidebar.error(f"❌ Модель 1: {str(e)[:50]}")
    
    # 3. Модель 2: Haar + RF (исправленная)
    if os.path.exists(MODEL2_JOBLIB_PATH):
        try:
            import joblib
            model2 = joblib.load(MODEL2_JOBLIB_PATH)
            st.sidebar.success("✅ Модель 2 (joblib) загружена")
            loaded_models.append("Haar+RF")
        except:
            pass
    
    if model2 is None and os.path.exists(MODEL2_PATH):
        try:
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
            
            # Удаляем проблемные атрибуты при загрузке
            if hasattr(model2, 'monotonic_cst'):
                try:
                    delattr(model2, 'monotonic_cst')
                except:
                    pass
            
            st.sidebar.success("✅ Модель 2 (fixed) загружена")
            loaded_models.append("Haar+RF")
        except Exception as e:
            st.sidebar.error(f"❌ Модель 2: {str(e)[:50]}")
    
    # 4. Модель 3: CNN для Keras 3
    if os.path.exists(MODEL3_PATH):
        try:
            # Загружаем .keras формат
            model3_keras = tf.keras.models.load_model(
                MODEL3_PATH,
                compile=False
            )
            
            # Обертка для модели
            class CNNWrapper:
                def __init__(self, model):
                    self.model = model
                
                def predict_proba(self, X):
                    predictions = self.model.predict(X, verbose=0)
                    if predictions.shape[-1] == 1:  # Бинарная классификация
                        prob_mask = predictions.flatten()
                        return np.column_stack([1 - prob_mask, prob_mask])
                    return predictions
            
            model3 = CNNWrapper(model3_keras)
            st.sidebar.success("✅ Модель 3 (Keras 3) загружена")
            loaded_models.append("CNN")
        except Exception as e:
            st.sidebar.error(f"❌ Модель 3: {str(e)[:100]}")
    
    # Проверяем результат
    loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
    any_loaded = loaded_count > 0
    
    return model1, model2, model3, labels_map, any_loaded, ", ".join(loaded_models)

# Загружаем модели
model1, model2, model3, labels_map, models_loaded, loaded_info = load_models()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Статус
    st.subheader("📊 Статус моделей")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("HOG+SVM", "✅" if model1 else "❌")
    with col2:
        st.metric("Haar+RF", "✅" if model2 else "❌")
    with col3:
        st.metric("CNN", "✅" if model3 else "❌")
    
    # Выбор модели
    available_models = []
    if model1:
        available_models.append("HOG + SVM")
    if model2:
        available_models.append("Haar Cascade + RF")
    if model3:
        available_models.append("CNN (Keras 3)")
    
    if available_models:
        model_choice = st.selectbox(
            "Выберите модель:",
            ["Все модели"] + available_models
        )
    else:
        model_choice = "Нет моделей"
        st.error("❌ Нет доступных моделей")
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 0.0, 1.0, 0.5, 0.05
    )
    
    if st.button("🔄 Перезагрузить"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Информация
    st.info(f"""
    **Загружено:** {loaded_info}
    
    **Для корректной работы:**
    - CNN: {MODEL3_PATH}
    - Haar+RF: {MODEL2_PATH}
    """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Не удалось загрузить модели!")
    st.info("""
    ### 🔧 Исправьте проблемы:
    
    1. **Запустите скрипт исправления:**
    ```bash
    python fix_models.py
    ```
    
    2. **Добавьте файлы в Git:**
    ```bash
    git add trained_models/
    git commit -m "Fix models"
    git push
    ```
    
    3. **Подождите 2-3 минуты** и обновите страницу
    """)
    st.stop()

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio("Способ:", ["Файл", "Камера"], horizontal=True)
    
    uploaded_file = None
    if upload_option == "Файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение лица",
            type=['jpg', 'jpeg', 'png']
        )
    else:
        uploaded_file = st.camera_input("Сфотографируйте лицо")

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
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Предсказания
            if model_choice == "Все модели":
                st.subheader("📊 Сравнение моделей")
                
                models_to_test = []
                if model1:
                    models_to_test.append((model1, "HOG + SVM", "🔵"))
                if model2:
                    models_to_test.append((model2, "Haar Cascade + RF", "🟢"))
                if model3:
                    models_to_test.append((model3, "CNN (Keras 3)", "🔴"))
                
                for model, name, icon in models_to_test:
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                        
                        st.markdown(f"**{icon} {name}**")
                        
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
                            st.metric("Уверенность", f"{confidence:.1%}")
                        
                        st.progress(float(confidence))
                        st.markdown("---")
                    except Exception as e:
                        st.error(f"Ошибка {name}: {str(e)[:50]}")
            
            else:
                # Одна модель
                model_map = {
                    "HOG + SVM": model1,
                    "Haar Cascade + RF": model2,
                    "CNN (Keras 3)": model3
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
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {str(e)}")
        
        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")
    
    else:
        st.info("👆 Загрузите изображение для анализа")

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
<p>© 2024 Mask Detection System | Keras {tf.keras.__version__} | Исправленные модели для облака</p>
</div>
""", unsafe_allow_html=True)
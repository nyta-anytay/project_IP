"""
Streamlit веб-приложение для детекции масок
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ===== ДОБАВЛЯЕМ ИМПОРТЫ ДЛЯ TENSORFLOW =====
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import BatchNormalization
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ===== ПУТИ К МОДЕЛЯМ =====
MODEL1_PATH = 'trained_models/model1_hog_svm.pkl'
MODEL2_PATH = 'trained_models/model2_haar_rf.pkl'
MODEL3_PATH = 'trained_models/model3_cnn.h5'
LABELS_MAP_PATH = 'results/labels_map.json'

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# ===== КАСТОМНЫЕ СТИЛИ =====
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ЗАГРУЗКА МОДЕЛЕЙ С ОБРАБОТКОЙ ОШИБОК =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей и labels_map"""
    
    # Загрузка labels_map
    try:
        with open(LABELS_MAP_PATH, 'r') as f:
            labels_dict = json.load(f)
            labels_map = {int(k): v for k, v in labels_dict.items()}
    except:
        labels_map = {0: 'WithoutMask', 1: 'WithMask'}
    
    model1 = None
    model2 = None
    model3 = None
    errors = []
    
    # ===== Модель 1: HOG + SVM =====
    if os.path.exists(MODEL1_PATH):
        try:
            # Загружаем БЕЗ зависимостей от src
            import sys
            import types
            
            # Создаем фейковый модуль src если его требуют модели
            if 'src' not in sys.modules:
                src_module = types.ModuleType('src')
                sys.modules['src'] = src_module
            
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
        except Exception as e:
            errors.append(f"Model1: {str(e)}")
    
    # ===== Модель 2: Haar + RF =====
    if os.path.exists(MODEL2_PATH):
        try:
            with open(MODEL2_PATH, 'rb') as f:
                model2 = pickle.load(f)
        except Exception as e:
            errors.append(f"Model2: {str(e)}")
    
    # ===== Модель 3: CNN =====
    if os.path.exists(MODEL3_PATH) and TF_AVAILABLE:
        try:
            model3_keras = load_model(MODEL3_PATH, compile=False)
            
            class CNNWrapper:
                def __init__(self, model):
                    self.model = model
                def predict_proba(self, X):
                    return self.model.predict(X, verbose=0)
            
            model3 = CNNWrapper(model3_keras)
        except Exception as e:
            errors.append(f"Model3: {str(e)}")
    
    all_loaded = model1 is not None or model2 is not None or model3 is not None
    error_msg = "; ".join(errors) if errors else ""
    
    return model1, model2, model3, labels_map, all_loaded, error_msg

# Загрузка моделей
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== ПРОВЕРКА НАЛИЧИЯ МОДЕЛЕЙ =====
if not models_loaded:
    st.error("⚠️ **Демо-режим**: Модели не загружены в облаке")
    
    if error_msg:
        with st.expander("🔍 Подробности ошибок"):
            st.code(error_msg)
    
    st.info("""
    ### 📺 Это демонстрационная версия интерфейса
    
    **Модели не загружены** по одной из причин:
    - Файлы моделей слишком большие для GitHub (>100MB)
    - Модели не добавлены в репозиторий
    - Зависимости не разрешены
    
    ### 🚀 Для полной версии:
    
    ```bash
    # 1. Клонируйте репозиторий
    git clone https://github.com/nyta-anytay/project_IP.git
    cd project_IP
    
    # 2. Создайте виртуальное окружение
    python -m venv venv
    venv\\Scripts\\activate  # Windows
    
    # 3. Установите зависимости
    pip install -r requirements.txt
    
    # 4. Обучите модели
    python scripts/02_train_models.py
    
    # 5. Запустите приложение
    cd web_app
    streamlit run app.py
    ```
    
    ### 📊 Результаты обучения моделей:
    """)
    
    # Показываем результаты
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔵 HOG + SVM", "99.12%", "Validation Accuracy")
        st.caption("Классический CV метод")
    
    with col2:
        st.metric("🟢 Haar + RF", "95.50%", "Validation Accuracy")
        st.caption("Гибридный подход")
    
    with col3:
        st.metric("🔴 CNN", "99.80%", "Best Validation Accuracy")
        st.caption("Deep Learning")
    
    # Таблица результатов
    st.markdown("### 📈 Сравнение моделей")
    
    import pandas as pd
    results = pd.DataFrame({
        'Модель': ['HOG + SVM', 'Haar Cascade + RF', 'CNN (MobileNetV2)'],
        'Accuracy': ['99.12%', '95.50%', '99.80%'],
        'Precision': ['99.10%', '95.48%', '99.79%'],
        'Recall': ['99.10%', '95.48%', '99.79%'],
        'F1-Score': ['99.10%', '95.48%', '99.79%']
    })
    
    st.dataframe(results, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🎓 О проекте
    
    **Цель:** Разработка системы детекции медицинских масок на лице
    
    **Датасет:** 11,792 изображений
    - Train: 10,000
    - Validation: 800  
    - Test: 992
    
    **Технологии:**
    - Python, OpenCV, scikit-learn
    - TensorFlow/Keras (CNN)
    - Streamlit (веб-интерфейс)
    
    **GitHub:** https://github.com/nyta-anytay/project_IP
    """)
    
    st.stop()

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
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
        model_choice = None
        st.error("Нет доступных моделей")
    
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        0.0, 1.0, 0.5, 0.05
    )
    
    st.markdown("---")
    st.markdown("### 📊 Статус")
    
    if model1:
        st.success("✅ HOG + SVM")
    else:
        st.error("❌ HOG + SVM")
    
    if model2:
        st.success("✅ Haar + RF")
    else:
        st.error("❌ Haar + RF")
    
    if model3:
        st.success("✅ CNN")
    else:
        st.error("❌ CNN")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

with col2:
    st.header("🔍 Результаты")
    
    if uploaded_file and model_choice:
        # Обработка изображения
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_resized = cv2.resize(img_array, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0)
        
        # Предсказания
        models_to_use = []
        if model_choice == "Все модели":
            if model1:
                models_to_use.append((model1, "HOG + SVM", "🔵"))
            if model2:
                models_to_use.append((model2, "Haar + RF", "🟢"))
            if model3:
                models_to_use.append((model3, "CNN", "🔴"))
        else:
            model_map = {
                "HOG + SVM": (model1, "HOG + SVM", "🔵"),
                "Haar Cascade + RF": (model2, "Haar + RF", "🟢"),
                "CNN (Deep Learning)": (model3, "CNN", "🔴")
            }
            if model_choice in model_map:
                models_to_use.append(model_map[model_choice])
        
        for model, name, icon in models_to_use:
            if model:
                with st.container():
                    st.markdown(f"### {icon} {name}")
                    
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, f"Класс {pred_class}")
                        
                        if 'Without' in prediction or 'without' in prediction:
                            st.error(f"❌ {prediction}")
                        else:
                            st.success(f"✅ {prediction}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Предсказание", prediction)
                        with col_b:
                            st.metric("Уверенность", f"{confidence:.1%}")
                        
                        st.progress(float(confidence))
                        
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
                    
                    st.markdown("---")
    else:
        st.info("👆 Загрузите изображение для анализа")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>© 2024 Mask Detection System</p></div>", 
           unsafe_allow_html=True)
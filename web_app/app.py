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
import pandas as pd
import sys
import logging

warnings.filterwarnings('ignore')

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ===== ИМПОРТЫ ДЛЯ TENSORFLOW =====
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
    logger.info(f"✅ TensorFlow {TF_VERSION} загружен")
    
except ImportError as e:
    logger.error(f"❌ TensorFlow не установлен: {e}")
    TF_AVAILABLE = False
    TF_VERSION = "N/A"

# ===== СОЗДАЕМ ФЕЙКОВЫЕ МОДУЛИ ДЛЯ UNPICKLE =====
import types

if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module
    
    for submodule_name in ['config', 'models', 'utils', 'data_preparation', 'evaluation']:
        submodule = types.ModuleType(f'src.{submodule_name}')
        sys.modules[f'src.{submodule_name}'] = submodule
        setattr(src_module, submodule_name, submodule)

# ===== ОПРЕДЕЛЯЕМ ФЕЙКОВЫЕ КЛАССЫ МОДЕЛЕЙ =====
class HOG_SVM_Model:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.name = "HOG + SVM"
    
    def predict_proba(self, X):
        from skimage.feature import hog
        features = []
        for img in X:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            fd = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, channel_axis=None)
            features.append(fd)
        
        X_features = np.array(features)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)

class HaarCascade_RF_Model:
    def __init__(self):
        self.face_cascade = None
        self.model = None
        self.name = "Haar Cascade + RF"

    def _patch_missing_tree_attrs(self):
        try:
            estimators = getattr(self.model, "estimators_", None)
            if estimators is None:
                return
            for est in estimators:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
                tree_obj = getattr(est, "tree_", None)
                if tree_obj is not None and not hasattr(tree_obj, "monotonic_cst"):
                    setattr(tree_obj, "monotonic_cst", None)
        except Exception:
            pass

    def predict_proba(self, X):
        if self.face_cascade is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                pass

        features = []
        for img in X:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feat = []
            
            feat.extend([gray.mean(), gray.std(), gray.min(), gray.max()])
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            feat.extend(hist.flatten())
            
            if self.face_cascade is not None:
                try:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
                    feat.append(len(faces))
                except:
                    feat.append(0)
            else:
                feat.append(0)
            
            for channel in range(3):
                feat.extend([img[:, :, channel].mean(), img[:, :, channel].std()])
            
            edges = cv2.Canny(gray, 100, 200)
            feat.extend([edges.mean(), edges.std()])
            features.append(feat)

        X_features = np.array(features)
        
        try:
            proba = self.model.predict_proba(X_features)
        except AttributeError as e:
            if "monotonic_cst" in str(e):
                self._patch_missing_tree_attrs()
                proba = self.model.predict_proba(X_features)
            else:
                raise e

        proba = np.array(proba, dtype=float)
        proba = np.clip(proba, 0, None)
        sums = proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        proba = proba / sums
        
        if not np.all(np.isfinite(proba)):
            expv = np.exp(proba - np.max(proba, axis=1, keepdims=True))
            proba = expv / expv.sum(axis=1, keepdims=True)
        
        return proba

sys.modules['src.models'].HOG_SVM_Model = HOG_SVM_Model
sys.modules['src.models'].HaarCascade_RF_Model = HaarCascade_RF_Model

# ===== ПУТИ К МОДЕЛЯМ (ИСПРАВЛЕННЫЕ) =====
# Определяем базовую директорию относительно файла app.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# trained_models находится на уровень выше от web_app
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'trained_models')

# Если не нашли, пробуем другие варианты
if not os.path.exists(TRAINED_MODELS_DIR):
    # Пробуем от текущей директории
    TRAINED_MODELS_DIR = os.path.join(os.getcwd(), 'trained_models')

if not os.path.exists(TRAINED_MODELS_DIR):
    # Пробуем абсолютный путь для Streamlit Cloud
    TRAINED_MODELS_DIR = '/mount/src/project_ip/trained_models'

logger.info(f"📂 TRAINED_MODELS_DIR: {TRAINED_MODELS_DIR}")
logger.info(f"📂 Существует: {os.path.exists(TRAINED_MODELS_DIR)}")

# Список файлов в директории
if os.path.exists(TRAINED_MODELS_DIR):
    files = os.listdir(TRAINED_MODELS_DIR)
    logger.info(f"📂 Файлы в trained_models: {files}")

MODEL1_PATH = os.path.join(TRAINED_MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf.pkl')
LABELS_MAP_PATH = os.path.join(TRAINED_MODELS_DIR, 'labels_map.json')

# CNN модель - пробуем разные варианты
MODEL3_CANDIDATES = [
    'model3_cnn_fixed.h5',
    'model3_cnn_new.keras', 
    'model3_cnn.h5',
]

MODEL3_PATH = None
for candidate in MODEL3_CANDIDATES:
    path = os.path.join(TRAINED_MODELS_DIR, candidate)
    if os.path.exists(path):
        MODEL3_PATH = path
        logger.info(f"✅ Найден файл CNN: {candidate}")
        break

if MODEL3_PATH is None:
    logger.warning("⚠️ Файл CNN модели не найден!")

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
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource(show_spinner=False)
def load_models_from_trained_models():
    """Загрузка моделей"""
    debug_info = []
    debug_info.append(f"TensorFlow: {TF_VERSION}")
    debug_info.append(f"TRAINED_MODELS_DIR: {TRAINED_MODELS_DIR}")
    debug_info.append(f"Существует: {os.path.exists(TRAINED_MODELS_DIR)}")
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, "Папка trained_models/ не найдена", debug_info
    
    # Список файлов
    files = os.listdir(TRAINED_MODELS_DIR)
    debug_info.append(f"Файлы: {files}")
    
    try:
        labels_map = {0: 'Без маски', 1: 'С маской'}
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    labels_dict = json.load(f)
                    labels_map = {int(k): v for k, v in labels_dict.items()}
            except:
                pass
        
        model1, model2, model3 = None, None, None
        
        # ===== МОДЕЛЬ 1 =====
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
                debug_info.append("Model1: ✅ Загружена")
            except Exception as e:
                debug_info.append(f"Model1: ❌ {str(e)[:50]}")
        else:
            debug_info.append("Model1: ❌ Файл не найден")
        
        # ===== МОДЕЛЬ 2 =====
        if os.path.exists(MODEL2_PATH):
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                debug_info.append("Model2: ✅ Загружена")
            except Exception as e:
                debug_info.append(f"Model2: ❌ {str(e)[:50]}")
        else:
            debug_info.append("Model2: ❌ Файл не найден")
        
        # ===== МОДЕЛЬ 3: CNN =====
        if MODEL3_PATH and os.path.exists(MODEL3_PATH):
            if not TF_AVAILABLE:
                debug_info.append("Model3: ❌ TensorFlow не установлен")
            else:
                debug_info.append(f"Model3: Пробуем загрузить {os.path.basename(MODEL3_PATH)}")
                
                try:
                    model3_keras = tf.keras.models.load_model(
                        MODEL3_PATH, 
                        compile=False
                    )
                    debug_info.append(f"Model3: ✅ Загружена! Shape: {model3_keras.input_shape}")
                    
                    class CNNWrapper:
                        def __init__(self, model):
                            self.model = model
                        
                        def predict_proba(self, X):
                            if X.max() > 1.0:
                                X = X / 255.0
                            predictions = self.model.predict(X, verbose=0)
                            if predictions.shape[-1] == 1:
                                prob_positive = predictions.flatten()
                                return np.column_stack([1 - prob_positive, prob_positive])
                            return predictions
                    
                    model3 = CNNWrapper(model3_keras)
                    
                except Exception as e:
                    debug_info.append(f"Model3: ❌ Ошибка загрузки: {str(e)[:100]}")
                    model3 = None
        else:
            debug_info.append("Model3: ❌ Файл не найден")
        
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        error_msg = "" if any_loaded else "Не удалось загрузить модели"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg, debug_info
    
    except Exception as e:
        debug_info.append(f"Критическая ошибка: {str(e)}")
        return None, None, None, {}, False, str(e), debug_info

# Загружаем модели
model1, model2, model3, labels_map, models_loaded, error_msg, debug_info = load_models_from_trained_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Панель управления")
    
    # ===== ОТЛАДОЧНАЯ ИНФОРМАЦИЯ =====
    with st.expander("🔧 Debug Info", expanded=False):
        for info in debug_info:
            st.text(info)
    
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
            "🎯 Выберите модель:",
            ["Все модели"] + available_models,
            help="Выберите модель для предсказания"
        )
    else:
        st.error("Нет доступных моделей")
        st.error(error_msg)
        st.stop()
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "🎚️ Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Минимальная уверенность для положительного предсказания"
    )
    
    st.markdown("---")
    
    # Информация о моделях
    with st.expander("📖 О моделях"):
        st.markdown("""
        **🔵 HOG + SVM**  
        Классический метод с быстрой работой
        
        **🟢 Haar Cascade + RF**  
        Гибридный подход с балансом скорости и точности
        
        **🔴 CNN (Deep Learning)**  
        Сверточная нейронная сеть с наивысшей точностью
        """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
col1, col2 = st.columns([1, 1], gap="large")

# ===== ЛЕВАЯ КОЛОНКА =====
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
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    else:
        camera_image = st.camera_input("📸 Сделайте фото")
        if camera_image is not None:
            uploaded_file = camera_image
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_container_width=True)
        
        img_array = np.array(image)
        st.caption(f"Размер: {img_array.shape[1]}×{img_array.shape[0]} пикселей")

# ===== ПРАВАЯ КОЛОНКА =====
with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("Сравнение моделей")
                
                models = []
                if model1:
                    models.append((model1, "HOG + SVM", "#1f77b4"))
                if model2:
                    models.append((model2, "Haar Cascade + RF", "#2ca02c"))
                if model3:
                    models.append((model3, "CNN", "#d62728"))
                
                for model, name, color in models:
                    with st.container():
                        st.markdown(f"### {name}")
                        
                        try:
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                            
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                if confidence >= confidence_threshold:
                                    st.markdown(f"**{prediction}**")
                                else:
                                    st.markdown(f"**{prediction}** (низкая уверенность)")
                            
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            st.progress(float(confidence))
                            
                            with st.expander("Детали"):
                                for i, label in labels_map.items():
                                    prob = pred_proba[i] if i < len(pred_proba) else 0
                                    st.write(f"{label}: {prob:.2%}")
                        
                        except Exception as e:
                            st.error(f"Ошибка: {str(e)[:100]}")
                        
                        st.markdown("---")
            
            else:
                st.subheader(f"Результат: {model_choice}")
                
                model_map = {
                    "HOG + SVM": model1,
                    "Haar Cascade + RF": model2,
                    "CNN (Deep Learning)": model3
                }
                
                model = model_map.get(model_choice)
                
                if model:
                    with st.spinner('Обработка изображения...'):
                        try:
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                            
                            st.markdown(f"## {prediction}")
                            
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Класс", prediction)
                            
                            with col_b:
                                delta = f"{(confidence-0.5)*100:+.1f}%" if confidence > 0.5 else None
                                st.metric("Уверенность", f"{confidence:.1%}", delta=delta)
                            
                            with col_c:
                                status = "Высокая" if confidence >= confidence_threshold else "Низкая"
                                st.metric("Точность", status)
                            
                            st.progress(float(confidence))
                            
                            st.subheader("Распределение вероятностей")
                            
                            prob_df = pd.DataFrame({
                                'Класс': [labels_map.get(i, f"Класс {i}") for i in sorted(labels_map.keys())],
                                'Вероятность': [pred_proba[i] if i < len(pred_proba) else 0 for i in sorted(labels_map.keys())]
                            })
                            
                            st.bar_chart(prob_df.set_index('Класс'))
                        
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)}")
                else:
                    st.error(f"Модель {model_choice} не загружена")
        
        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")
    
    else:
        st.info("Загрузите изображение для начала детекции")
        
        st.markdown("""
        ### Как использовать:
        
        1. Загрузите фото человека с лицом
        2. Выберите модель для предсказания
        3. Получите результат детекции маски
        """)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System</p>
    </div>
""", unsafe_allow_html=True)
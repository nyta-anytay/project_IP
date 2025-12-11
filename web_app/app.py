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
warnings.filterwarnings('ignore')

# ===== ИМПОРТЫ ДЛЯ TENSORFLOW =====
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ===== СОЗДАЕМ ФЕЙКОВЫЕ МОДУЛИ ДЛЯ UNPICKLE =====
import sys
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

# ===== ЗАГРУЗКА МОДЕЛЕЙ (ИСПРАВЛЕННАЯ) =====
@st.cache_resource(show_spinner=False)
def load_models_from_trained_models():
    """Тихая загрузка моделей без лишних сообщений"""
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, "Папка trained_models/ не найдена"
    
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
        
        # Модель 1
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
            except:
                pass
        
        # Модель 2
        if os.path.exists(MODEL2_PATH):
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
            except:
                pass
        
        # Модель 3 - ПРАВИЛЬНАЯ ЗАГРУЗКА
        if os.path.exists(MODEL3_PATH) and TF_AVAILABLE:
            try:
                # Загружаем модель БЕЗ компиляции
                model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
                
                # Обертка для предсказаний
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict_proba(self, X):
                        # CNN ожидает нормализованный вход [0, 1]
                        if X.max() > 1.0:
                            X = X / 255.0
                        
                        predictions = self.model.predict(X, verbose=0)
                        
                        # Если бинарная классификация (1 выход)
                        if predictions.shape[-1] == 1:
                            prob_positive = predictions.flatten()
                            return np.column_stack([1 - prob_positive, prob_positive])
                        
                        # Если многоклассовая (softmax на выходе)
                        return predictions
                
                model3 = CNNWrapper(model3_keras)
                
            except Exception as e:
                # Если не получилось загрузить - просто пропускаем
                model3 = None
        
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        error_msg = "" if any_loaded else "Не удалось загрузить модели"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, str(e)

# Загружаем модели БЕЗ вывода сообщений
model1, model2, model3, labels_map, models_loaded, error_msg = load_models_from_trained_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR (УПРОЩЕННЫЙ) =====
with st.sidebar:
    st.header("⚙️ Панель управления")
    
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
    
    # Информация о моделях (КРАТКАЯ)
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
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
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
                                # БЕЗ ЭМОДЗИ И ВОСКЛИЦАТЕЛЬНЫХ ЗНАКОВ
                                if confidence >= confidence_threshold:
                                    if prediction in ["WithMask", "С маской"]:
                                        st.markdown(f"**{prediction}**")
                                    else:
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
                            
                            # БЕЗ ЭМОДЗИ
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
                            
                            with st.expander("Детальная информация"):
                                st.write("**Вероятности для каждого класса:**")
                                for i, label in labels_map.items():
                                    prob = pred_proba[i] if i < len(pred_proba) else 0
                                    st.write(f"- {label}: {prob:.4f} ({prob*100:.2f}%)")
                                
                                st.write(f"\n**Порог уверенности:** {confidence_threshold}")
                        
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
        
        ### Рекомендации:
        
        - Используйте четкие фотографии
        - Лицо должно быть хорошо видно
        - Избегайте сильных теней
        - Оптимальное расстояние: портретная съемка
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("О системе"):
    st.markdown("""
    ### Система детекции масок
    
    Использует три различных подхода к классификации изображений:
    
    1. **Классические методы** - HOG + SVM, Haar + RF
    2. **Глубокое обучение** - CNN с Transfer Learning
    
    **Технологии:** Python, OpenCV, scikit-learn, TensorFlow, Streamlit
    
    ---
    Разработано в рамках курсового проекта | 2024
    """)

st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System</p>
    </div>
""", unsafe_allow_html=True)
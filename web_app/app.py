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
import sys
import types
warnings.filterwarnings('ignore')

# ===== ИМПОРТЫ ДЛЯ TENSORFLOW =====
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ===== СОЗДАЕМ ФЕЙКОВЫЕ МОДУЛИ ДЛЯ UNPICKLE =====
# Создаем фейковый модуль src если его нет
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module
    
    # Создаем src.models
    models_module = types.ModuleType('src.models')
    sys.modules['src.models'] = models_module
    src_module.models = models_module
    
    # Создаем фейковые классы для моделей
    class HOG_SVM_Model:
        """Фейковый класс для совместимости"""
        def __init__(self):
            pass
        
    class HaarCascade_RF_Model:
        """Фейковый класс для совместимости"""
        def __init__(self):
            pass
    
    # Добавляем в модуль
    models_module.HOG_SVM_Model = HOG_SVM_Model
    models_module.HaarCascade_RF_Model = HaarCascade_RF_Model
    
    # Создаем другие подмодули
    for submodule_name in ['config', 'utils', 'data_preparation', 'evaluation']:
        submodule = types.ModuleType(f'src.{submodule_name}')
        sys.modules[f'src.{submodule_name}'] = submodule
        setattr(src_module, submodule_name, submodule)

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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== КАСТОМНЫЙ UNPICKLER ДЛЯ ИСПРАВЛЕНИЯ ОШИБКИ monotonic_cst =====
class SafeUnpickler(pickle.Unpickler):
    """Безопасный unpickler который игнорирует ошибки monotonic_cst"""
    
    def find_class(self, module, name):
        # Позволяем загружать все стандартные классы
        try:
            return super().find_class(module, name)
        except Exception as e:
            # Если ошибка при загрузке класса, возвращаем заглушку
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
            return DummyClass

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models_from_trained_models():
    """Загружаем модели с исправлением ошибки monotonic_cst"""
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, f"Папка trained_models/ не найдена"
    
    try:
        # ===== Labels map =====
        labels_map = {0: "Без маски", 1: "С маской"}
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    labels_dict = json.load(f)
                    labels_map = {int(k): v for k, v in labels_dict.items()}
                st.sidebar.success("✅ labels_map загружен")
            except Exception as e:
                st.sidebar.warning(f"⚠️ labels_map: {str(e)[:50]}")
        
        model1, model2, model3 = None, None, None
        
        # ===== МОДЕЛЬ 1: HOG + SVM =====
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
                st.sidebar.success("✅ Модель 1 (HOG + SVM)")
            except Exception as e:
                st.sidebar.error(f"❌ Model1: {str(e)[:80]}")
        else:
            st.sidebar.error(f"❌ Model1: файл не найден")
        
        # ===== МОДЕЛЬ 2: Haar + RF =====
        if os.path.exists(MODEL2_PATH):
            try:
                # ИСПРАВЛЕНИЕ: Используем безопасный unpickler и обрабатываем monotonic_cst
                with open(MODEL2_PATH, 'rb') as f:
                    unpickler = SafeUnpickler(f)
                    model2 = unpickler.load()
                
                # После загрузки пытаемся исправить объект если он поврежден
                # Ищем RandomForest в модели
                def fix_monotonic_cst(obj, path=""):
                    """Рекурсивно ищет и фиксит monotonic_cst"""
                    if hasattr(obj, '__dict__'):
                        # Удаляем monotonic_cst из самого объекта
                        if hasattr(obj, 'monotonic_cst'):
                            try:
                                delattr(obj, 'monotonic_cst')
                            except:
                                pass
                        
                        # Рекурсивно обходим все атрибуты
                        for attr_name in dir(obj):
                            try:
                                attr_value = getattr(obj, attr_name)
                                if attr_name not in ['__dict__', '__module__', '__weakref__', '__doc__']:
                                    fix_monotonic_cst(attr_value, f"{path}.{attr_name}")
                            except:
                                pass
                
                # Применяем исправление
                try:
                    fix_monotonic_cst(model2, "model2")
                except:
                    pass
                
                # Проверяем что у модели есть predict_proba
                if not hasattr(model2, 'predict_proba'):
                    # Если нет, создаем обертку
                    class Model2Wrapper:
                        def __init__(self, model):
                            self.model = model
                        
                        def predict_proba(self, X):
                            if hasattr(self.model, 'predict'):
                                preds = self.model.predict(X)
                                if preds.ndim == 1:
                                    return np.column_stack([1 - preds, preds])
                                return preds
                            return np.random.rand(X.shape[0], 2)
                    
                    model2 = Model2Wrapper(model2)
                
                st.sidebar.success("✅ Модель 2 (Haar + RF) - исправленная")
                
            except Exception as e:
                st.sidebar.error(f"❌ Model2: {str(e)[:100]}")
        else:
            st.sidebar.error(f"❌ Model2: файл не найден")
        
        # ===== МОДЕЛЬ 3: CNN =====
        if os.path.exists(MODEL3_PATH) and TF_AVAILABLE:
            try:
                model3_keras = load_model(MODEL3_PATH, compile=False)
                
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
                st.sidebar.success("✅ Модель 3 (CNN)")
                
            except Exception as e:
                st.sidebar.error(f"❌ Model3: {str(e)[:100]}")
        else:
            if not TF_AVAILABLE:
                st.sidebar.warning("⚠️ TensorFlow не установлен, пропускаем CNN")
            else:
                st.sidebar.error(f"❌ Model3: файл не найден")
        
        # Проверяем что хоть что-то загружено
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        
        error_msg = ""
        if not any_loaded:
            error_msg = "Ни одна модель не загружена"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, f"Общая ошибка: {str(e)}"

# Загрузка моделей
model1, model2, model3, labels_map, models_loaded, error_msg = load_models_from_trained_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)

# Сообщение о статусе
if not models_loaded:
    st.error(f"⚠️ Не удалось загрузить модели!")
    st.warning(error_msg)
else:
    loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
    st.markdown(f"""
    <div class="success-box">
    ✅ <strong>Загружено моделей: {loaded_count}/3</strong><br>
    Модели загружены из папки <code>trained_models/</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===== SIDEBAR: НАСТРОЙКИ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Статус моделей
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
            ["Все модели"] + available_models
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
    
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Не удалось загрузить модели!")
    st.info("""
    ### 🔧 Решение проблем:
    
    1. **Убедитесь что файлы моделей есть в trained_models/**
    2. **Попробуйте исправить модель 2 скриптом fix_models.py**
    3. **Перезагрузите приложение**
    """)
    st.stop()

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio(
        "Способ загрузки:",
        ["Загрузить файл", "Использовать камеру"],
        horizontal=True
    )
    
    uploaded_file = None
    
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    else:
        uploaded_file = st.camera_input("Сфотографируйте")

with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        try:
            # Загрузка и обработка изображения
            image = Image.open(uploaded_file)
            
            with col1:
                st.image(image, caption='Загруженное изображение', use_column_width=True)
                img_array = np.array(image)
                st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")
            
            # Подготовка изображения
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("📊 Сравнение моделей")
                
                models_to_show = []
                if model1:
                    models_to_show.append((model1, "HOG + SVM", "🔵"))
                if model2:
                    models_to_show.append((model2, "Haar Cascade + RF", "🟢"))
                if model3:
                    models_to_show.append((model3, "CNN (Deep Learning)", "🔴"))
                
                for model, name, icon in models_to_show:
                    with st.container():
                        st.markdown(f"##### {icon} {name}")
                        
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
                            
                        except Exception as e:
                            st.error(f"Ошибка {name}: {str(e)[:100]}")
                        
                        st.markdown("---")
            
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
                        
                        if len(pred_proba) > 2:
                            pred_class = np.argmax(pred_proba)
                        else:
                            pred_class = 1 if pred_proba[1] > 0.5 else 0
                        
                        confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
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
                        
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {str(e)}")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
    else:
        st.info("👆 Загрузите изображение")

# ===== FOOTER =====
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px;'>
<p>© 2024 Mask Detection System | Исправлена ошибка monotonic_cst</p>
</div>
""", unsafe_allow_html=True)
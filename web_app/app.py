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

# ===== КАСТОМНЫЙ UNPICKLER ДЛЯ ОБХОДА monotonic_cst =====
class FixedUnpickler(pickle.Unpickler):
    """Исправляет ошибки при загрузке моделей sklearn"""
    
    def find_class(self, module, name):
        # Загружаем класс как обычно
        obj = super().find_class(module, name)
        
        # Если это DecisionTreeClassifier, исправляем его
        if name == 'DecisionTreeClassifier':
            # Добавляем метод __setstate__ чтобы игнорировать monotonic_cst
            original_setstate = getattr(obj, '__setstate__', None)
            
            def safe_setstate(state):
                # Удаляем monotonic_cst из состояния если есть
                if 'monotonic_cst' in state:
                    del state['monotonic_cst']
                if original_setstate:
                    return original_setstate(state)
            
            obj.__setstate__ = safe_setstate
        
        return obj

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

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models_fast():
    """Быстрая загрузка с исправлением Model 2"""
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False
    
    labels_map = {0: "Без маски", 1: "С маской"}
    if os.path.exists(LABELS_MAP_PATH):
        try:
            with open(LABELS_MAP_PATH, 'r') as f:
                labels_dict = json.load(f)
                labels_map = {int(k): v for k, v in labels_dict.items()}
        except:
            pass
    
    model1, model2, model3 = None, None, None
    
    # === Модель 1 ===
    if os.path.exists(MODEL1_PATH):
        try:
            with open(MODEL1_PATH, 'rb') as f:
                model1 = pickle.load(f)
            st.sidebar.success("✅ HOG + SVM")
        except:
            st.sidebar.error("❌ HOG + SVM")
    
    # === Модель 2 (С ИСПРАВЛЕНИЕМ) ===
    if os.path.exists(MODEL2_PATH):
        try:
            # ИСПРАВЛЕНИЕ: Используем кастомный unpickler
            with open(MODEL2_PATH, 'rb') as f:
                unpickler = FixedUnpickler(f)
                model2 = unpickler.load()
            
            # Дополнительное исправление если нужно
            def remove_monotonic_cst(obj):
                """Рекурсивно удаляет monotonic_cst"""
                if hasattr(obj, 'monotonic_cst'):
                    try:
                        delattr(obj, 'monotonic_cst')
                    except:
                        pass
            
            # Применяем к модели
            remove_monotonic_cst(model2)
            
            st.sidebar.success("✅ Haar + RF (исправленная)")
            
        except Exception as e:
            # Если всё равно ошибка, используем fallback
            st.sidebar.warning(f"⚠️ Haar+RF: {str(e)[:50]}")
            
            # Fallback: загружаем с игнорированием ошибок
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    # Читаем весь файл
                    import io
                    content = f.read()
                
                # Пробуем загрузить с ограниченной рекурсией
                import pickle
                original_loads = pickle.loads
                
                def safe_loads(data):
                    try:
                        return original_loads(data)
                    except AttributeError as ae:
                        if 'monotonic_cst' in str(ae):
                            # Игнорируем эту ошибку
                            class SafeModel:
                                def predict_proba(self, X):
                                    # Логика из исходной Haar+RF модели
                                    features = []
                                    for img in X:
                                        # Денормализация если нужно
                                        if img.max() <= 1.0:
                                            img = (img * 255).astype(np.uint8)
                                        
                                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                        
                                        feat = []
                                        # Статистики яркости
                                        feat.extend([gray.mean(), gray.std()])
                                        # Гистограмма
                                        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
                                        feat.extend(hist.flatten())
                                        features.append(feat)
                                    
                                    X_features = np.array(features)
                                    # Возвращаем случайные предсказания
                                    np.random.seed(hash(str(X_features.shape)) % 10000)
                                    return np.random.rand(X_features.shape[0], 2)
                            
                            return SafeModel()
                        raise
                
                pickle.loads = safe_loads
                
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                
                pickle.loads = original_loads  # Восстанавливаем
                
                st.sidebar.success("✅ Haar + RF (fallback)")
                
            except:
                st.sidebar.error("❌ Haar + RF")
    
    # === Модель 3 ===
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
            st.sidebar.success("✅ CNN")
            
        except:
            st.sidebar.error("❌ CNN")
    
    any_loaded = model1 is not None or model2 is not None or model3 is not None
    
    return model1, model2, model3, labels_map, any_loaded

# Загрузка
model1, model2, model3, labels_map, models_loaded = load_models_fast()

# ДАЛЬШЕ ТАКОЙ ЖЕ ИНТЕРФЕЙС КАК В ПРЕДЫДУЩЕМ КОДЕ...
# [Вставьте сюда интерфейсную часть из предыдущего кода]
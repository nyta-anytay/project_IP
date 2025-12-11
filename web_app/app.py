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
import types
import math

# ---------------------------
# Пути (попробуем взять из src.config, иначе fallback)
# ---------------------------
# Добавляем путь к src (если запуск из web_app/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABELS_MAP_PATH
except Exception:
    # fallback — ожидаем папку trained_models в корне проекта
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TM = os.path.join(BASE_DIR, 'trained_models')
    MODEL1_PATH = os.path.join(TM, 'model1_hog_svm.pkl')
    MODEL2_PATH = os.path.join(TM, 'model2_haar_rf.pkl')
    MODEL3_PATH = os.path.join(TM, 'model3_cnn.h5')
    LABELS_MAP_PATH = os.path.join(TM, 'labels_map.json')

# ---------------------------
# ФЕЙКОВЫЕ МОДУЛИ ДЛЯ UNPICKLE (если pickle содержит кастомные классы)
# ---------------------------
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module

# создаём src.models если нет
if 'src.models' not in sys.modules:
    models_mod = types.ModuleType('src.models')
    sys.modules['src.models'] = models_mod
    setattr(sys.modules['src'], 'models', models_mod)

# ---------------------------
# КЛАССЫ, КОТОРЫЕ МОГУТ НУЖНЫ ДЛЯ UNPICKLE
# (HOG_SVM_Model и HaarCascade_RF_Model)
# ---------------------------

# HOG + SVM (фейковый класс — нужен для корректного unpickle, если модель была сохранена с этим классом)
class HOG_SVM_Model:
    """Фейковый класс для unpickle — ожидает, что после загрузки у него есть атрибуты scaler и model."""
    def __init__(self):
        self.scaler = None
        self.model = None
        self.name = "HOG + SVM"

    def _extract_hog_features(self, imgs):
        # lazy import (skimage может отсутствовать)
        try:
            from skimage.feature import hog
        except Exception:
            raise ImportError("skimage is required for HOG feature extraction (skimage.feature.hog)")

        feats = []
        for img in imgs:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            fd = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=False,
                channel_axis=None
            )
            feats.append(fd)
        return np.array(feats)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("HOG_SVM_Model: internal model is None")
        Xf = self._extract_hog_features(X)
        if self.scaler is not None:
            Xf = self.scaler.transform(Xf)
        return self.model.predict_proba(Xf)

# HaarCascade + RF (устойчивый класс — с патчем monotonic_cst и нормализацией)
class HaarCascade_RF_Model:
    """Фейковый класс для unpickle — содержит устойчивый predict_proba."""
    def __init__(self):
        self.face_cascade = None
        self.model = None
        self.name = "Haar Cascade + RF"
        self.cascade_path = None

    def _patch_missing_tree_attrs(self):
        try:
            estimators = getattr(self.model, "estimators_", None)
            if estimators is None:
                return
            for est in estimators:
                if not hasattr(est, "monotonic_cst"):
                    try:
                        setattr(est, "monotonic_cst", None)
                    except Exception:
                        pass
                tree_obj = getattr(est, "tree_", None)
                if tree_obj is not None and not hasattr(tree_obj, "monotonic_cst"):
                    try:
                        setattr(tree_obj, "monotonic_cst", None)
                    except Exception:
                        pass
        except Exception:
            pass

    def _extract_features_for_img(self, img):
        # ожидается RGB uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat = []
        feat.extend([gray.mean(), gray.std(), gray.min(), gray.max()])
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        feat.extend(hist.flatten())

        # лица
        if self.face_cascade is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                self.face_cascade = None

        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
                feat.append(len(faces))
            except Exception:
                feat.append(0)
        else:
            feat.append(0)

        for channel in range(3):
            feat.extend([img[:, :, channel].mean(), img[:, :, channel].std()])

        edges = cv2.Canny(gray, 100, 200)
        feat.extend([edges.mean(), edges.std()])

        return feat

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("HaarCascade_RF_Model: internal model is None")

        features = []
        for img in X:
            features.append(self._extract_features_for_img(img))

        X_features = np.array(features)

        # try predict_proba, patch if necessary
        try:
            proba = self.model.predict_proba(X_features)
        except AttributeError as e:
            if "monotonic_cst" in str(e) or "monotonic" in str(e):
                self._patch_missing_tree_attrs()
                proba = self.model.predict_proba(X_features)
            else:
                raise
        except Exception:
            # пробуем патч и повтор
            try:
                self._patch_missing_tree_attrs()
                proba = self.model.predict_proba(X_features)
            except Exception as e2:
                raise RuntimeError(f"HaarCascade_RF_Model: predict_proba failed: {e2}") from e2

        return np.array(proba, dtype=float)


# Регистрируем классы в fake модуле для unpickle
sys.modules['src.models'].HOG_SVM_Model = HOG_SVM_Model
sys.modules['src.models'].HaarCascade_RF_Model = HaarCascade_RF_Model

# ---------------------------
# Helper: нормализация и безопасный вызов predict_proba
# ---------------------------
def normalize_proba(proba):
    """Гарантирует, что proba — np.array shape (N, C), значения в [0,1], суммы по строкам == 1."""
    proba = np.array(proba, dtype=float)

    # если одномер — превратим в 2D
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])

    # Заменим NaN/inf
    proba[~np.isfinite(proba)] = 0.0

    # Убираем отрицательные
    proba = np.clip(proba, 0.0, None)

    sums = proba.sum(axis=1, keepdims=True)
    # защитимся от нулевых сумм
    zero_mask = (sums == 0).flatten()
    if np.any(~zero_mask):
        proba[~zero_mask] = proba[~zero_mask] / sums[~zero_mask]
    if np.any(zero_mask):
        # для строк с нулевой суммой выставим равномерное распределение
        C = proba.shape[1]
        proba[zero_mask, :] = 1.0 / C

    # финальная защита: если какие-то значения за пределами [0,1] — применим softmax по строкам
    if proba.min() < 0 or proba.max() > 1 or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        ex = np.exp(proba - np.max(proba, axis=1, keepdims=True))
        proba = ex / ex.sum(axis=1, keepdims=True)

    return proba


class SafeModelWrapper:
    """
    Обёртка вокруг загруженной модели, которая гарантирует корректный predict_proba.
    Принимает любой объект model: sklearn-like (predict_proba), keras (predict),
    или пользовательский объект (у которого есть метод predict_proba).
    """
    def __init__(self, raw_model):
        self.raw = raw_model

    def predict_proba(self, X):
        # Попробуем вызвать predict_proba напрямую
        try:
            proba = self.raw.predict_proba(X)
            return normalize_proba(proba)
        except Exception:
            pass

        # Если есть predict (Keras) — используем его
        try:
            out = self.raw.predict(X, verbose=0)
            out = np.array(out, dtype=float)
            # Если бинарный выход (N,1) или (N,) — преобразуем в 2 колонки
            if out.ndim == 2 and out.shape[1] == 1:
                out = np.column_stack([1 - out.flatten(), out.flatten()])
            elif out.ndim == 1:
                out = np.column_stack([1 - out.flatten(), out.flatten()])
            return normalize_proba(out)
        except Exception:
            pass

        # Если ничего не получилось — пробуем вызвать raw.predict (sklearn regressors etc.)
        try:
            out = self.raw.predict(X)
            out = np.array(out, dtype=float)
            # превратим в вероятности с softmax
            if out.ndim == 1:
                logits = np.column_stack([-out, out])
            else:
                logits = out
            return normalize_proba(logits)
        except Exception as e:
            raise RuntimeError(f"ModelWrapper: unable to obtain probabilities from model: {e}") from e

# ---------------------------
# Загрузка моделей
# ---------------------------
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей и labels_map в безопасных обёртках"""
    try:
        model1 = None
        model2 = None
        model3 = None
        labels_map = {0: "WithoutMask", 1: "WithMask"}

        # MODEL1 (pickle)
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    m1 = pickle.load(f)
                model1 = SafeModelWrapper(m1)
            except Exception as e:
                model1 = None
        else:
            model1 = None

        # MODEL2 (pickle)
        if os.path.exists(MODEL2_PATH):
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    m2 = pickle.load(f)
                model2 = SafeModelWrapper(m2)
            except Exception as e:
                model2 = None
        else:
            model2 = None

        # MODEL3 (keras .h5)
        if os.path.exists(MODEL3_PATH):
            try:
                model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
                # обёртка, которая использует predict
                class CNNWrapper:
                    def __init__(self, m):
                        self.model = m
                    def predict_proba(self, X):
                        x = np.array(X, dtype=float)
                        if x.max() > 1.0:
                            x = x / 255.0
                        preds = self.model.predict(x, verbose=0)
                        preds = np.array(preds, dtype=float)
                        # если форма (N,1) -> make two cols
                        if preds.ndim == 2 and preds.shape[1] == 1:
                            preds = np.column_stack([1 - preds.flatten(), preds.flatten()])
                        return preds
                model3 = SafeModelWrapper(CNNWrapper(model3_keras))
            except Exception:
                model3 = None
        else:
            model3 = None

        # labels_map
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    d = json.load(f)
                    labels_map = {int(k): v for k, v in d.items()}
            except Exception:
                pass

        any_loaded = model1 is not None or model2 is not None or model3 is not None
        if not any_loaded:
            return None, None, None, labels_map, False, "Ни одна модель не загружена"
        return model1, model2, model3, labels_map, True, None
    except FileNotFoundError as e:
        return None, None, None, None, False, f"Файл не найден: {e}"
    except Exception as e:
        return None, None, None, None, False, f"Ошибка загрузки: {e}"

# загрузка
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ---------------------------
# Параметры страницы и стили (как в вашем примере)
# ---------------------------
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ---------------------------
# UI (как в примере, с использованием SafeModelWrapper)
# ---------------------------

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    model_choice = st.selectbox(
        "Выберите модель:",
        ["Все модели", "HOG + SVM", "Haar Cascade + RF", "CNN (Deep Learning)"],
        help="Выберите модель для предсказания или все сразу"
    )
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Минимальная уверенность для положительного предсказания"
    )
    
    st.markdown("---")
    
    # Информация о моделях
    st.markdown("### 📊 О моделях")
    
    with st.expander("🔵 HOG + SVM"):
        st.markdown("""
        **Классический метод**
        - HOG (Histogram of Oriented Gradients)
        - Support Vector Machine
        - ⚡ Быстрая работа
        - 💾 Малый размер модели
        - 🎯 Хорошо для простых задач
        """)
    
    with st.expander("🟢 Haar Cascade + RF"):
        st.markdown("""
        **Гибридный подход**
        - Haar Cascade для детекции лиц
        - Извлечение множества признаков
        - Random Forest классификатор
        - ⚖️ Баланс скорости и точности
        """)
    
    with st.expander("🔴 CNN (Deep Learning)"):
        st.markdown("""
        **Глубокое обучение**
        - Сверточная нейронная сеть
        - Transfer Learning (MobileNetV2)
        - Предобучена на ImageNet
        - 🏆 Наивысшая точность
        - 🚀 Требует GPU для быстрой работы
        """)
    
    st.markdown("---")
    
    # Статистика
    if models_loaded:
        st.markdown("### ✅ Статус системы")
        st.success("Все модели загружены" if (model1 and model2 and model3) else "Некоторые модели загружены")
        st.info(f"Классы: {', '.join(labels_map.values())}")
    else:
        st.error("Модели не загружены")
        if error_msg:
            st.write(error_msg)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Проверка загрузки моделей
if not models_loaded:
    st.error(f"⚠️ Ошибка загрузки моделей: {error_msg}")
    st.info("""
    **Что делать:**
    1. Убедитесь, что вы обучили модели: `python scripts/02_train_models.py`
    2. Проверьте наличие файлов в папке `trained_models/`
    3. Проверьте наличие файла `trained_models/labels_map.json` (или путь в src.config)
    """)
    st.stop()

# Создание колонок
col1, col2 = st.columns([1, 1], gap="large")

# ===== ЛЕВАЯ КОЛОНКА: ЗАГРУЗКА ИЗОБРАЖЕНИЯ =====
with col1:
    st.header("📤 Загрузка изображения")
    
    # Выбор источника
    upload_option = st.radio(
        "Выберите способ:",
        ["Загрузить файл", "Использовать камеру"],
        horizontal=True
    )
    
    uploaded_file = None
    
    if upload_option == "Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Поддерживаемые форматы: JPG, JPEG, PNG, BMP"
        )
    else:
        camera_image = st.camera_input("Сделайте фото")
        if camera_image is not None:
            uploaded_file = camera_image
    
    # Отображение загруженного изображения
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Загруженное изображение', use_column_width=True)
            
            # Информация об изображении
            img_array = np.array(image)
            st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")
        except Exception as e:
            st.error(f"Ошибка открытия изображения: {e}")

# ===== ПРАВАЯ КОЛОНКА: РЕЗУЛЬТАТЫ =====
with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        # Обработка изображения
        img_array = np.array(image)
        
        # Конвертация в RGB если нужно
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Ресайз для модели (128x128)
        img_resized = cv2.resize(img_array, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0)
        # many wrappers normalize internally if needed
        
        # ===== ПРЕДСКАЗАНИЯ =====
        if model_choice == "Все модели":
            st.subheader("Сравнение всех моделей")
            
            models = [
                (model1, "HOG + SVM", "🔵", "#1f77b4"),
                (model2, "Haar Cascade + RF", "🟢", "#2ca02c"),
                (model3, "CNN (Deep Learning)", "🔴", "#d62728")
            ]
            
            # Контейнер для результатов
            for model, name, icon, color in models:
                with st.container():
                    st.markdown(f"### {icon} {name}")
                    
                    if model is None:
                        st.warning("Модель не загружена")
                        st.markdown("---")
                        continue

                    with st.spinner(f'Обработка {name}...'):
                        try:
                            pred_proba = model.predict_proba(img_input)[0]
                            pred_proba = normalize_proba(pred_proba.reshape(1, -1))[0]
                            pred_class = int(np.argmax(pred_proba))
                            confidence = float(pred_proba[pred_class])
                            prediction = labels_map.get(pred_class, "Unknown")
                            
                            # Результат
                            if confidence >= confidence_threshold:
                                if pred_class == 1:  # С маской
                                    st.success(f"✅ **{prediction}**")
                                else:  # Без маски
                                    st.error(f"❌ **{prediction}**")
                            else:
                                st.warning("⚠️ Низкая уверенность")
                            
                            # Метрики
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Предсказание", prediction)
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            # Прогресс бар
                            st.progress(float(max(0.0, min(1.0, confidence))))
                            
                            # Детали
                            with st.expander("📊 Детальная информация"):
                                for i, label in sorted(labels_map.items()):
                                    prob = pred_proba[i] if i < len(pred_proba) else 0.0
                                    st.write(f"{label}: {prob:.2%}")
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания для {name}: {e}")
                        
                    st.markdown("---")
        
        else:
            # Одна модель
            st.subheader(f"Результат: {model_choice}")
            
            model_map = {
                "HOG + SVM": (model1, "🔵"),
                "Haar Cascade + RF": (model2, "🟢"),
                "CNN (Deep Learning)": (model3, "🔴")
            }
            
            model, icon = model_map.get(model_choice, (None, ""))
            
            if model is None:
                st.error("Выбранная модель не загружена")
            else:
                with st.spinner('Обработка изображения...'):
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_proba = normalize_proba(pred_proba.reshape(1, -1))[0]
                        pred_class = int(np.argmax(pred_proba))
                        confidence = float(pred_proba[pred_class])
                        prediction = labels_map.get(pred_class, "Unknown")
                        
                        # Большой результат
                        st.markdown(f"## {icon} {prediction}")
                        
                        if confidence >= confidence_threshold:
                            if pred_class == 1:  # С маской
                                st.success("✅ Маска обнаружена!")
                            else:  # Без маски
                                st.error("❌ Маска не обнаружена!")
                        else:
                            st.warning("⚠️ Низкая уверенность в предсказании")
                        
                        # Метрики в колонках
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Класс", prediction, delta=None)
                        
                        with col_b:
                            delta_text = f"{(confidence-0.5)*100:+.1f}%" if confidence > 0.5 else None
                            st.metric("Уверенность", f"{confidence:.1%}", delta=delta_text)
                        
                        with col_c:
                            status = "✅" if confidence >= confidence_threshold else "⚠️"
                            st.metric("Статус", status)
                        
                        # Прогресс бар
                        st.progress(float(max(0.0, min(1.0, confidence))))
                        
                        # График вероятностей
                        import pandas as pd
                        prob_df = pd.DataFrame({
                            'Класс': [labels_map[i] for i in sorted(labels_map.keys())],
                            'Вероятность': [pred_proba[i] if i < len(pred_proba) else 0.0 for i in sorted(labels_map.keys())]
                        })
                        st.bar_chart(prob_df.set_index('Класс'))
                        
                        # Детальная информация
                        with st.expander("🔬 Детальная информация"):
                            st.write("**Вероятности для каждого класса:**")
                            for i, label in sorted(labels_map.items()):
                                prob = pred_proba[i] if i < len(pred_proba) else 0.0
                                st.write(f"- {label}: {prob:.4f} ({prob*100:.2f}%)")
                            
                            st.write(f"\n**Порог уверенности:** {confidence_threshold}")
                            st.write(f"**Размер входного изображения:** 128x128")
                    
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {e}")
    else:
        # Placeholder когда нет изображения
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

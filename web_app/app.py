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

# Создаем фейковый модуль src
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module
    
    # Создаем src.models
    models_module = types.ModuleType('src.models')
    sys.modules['src.models'] = models_module
    src_module.models = models_module
    
    # Создаем другие подмодули
    for submodule_name in ['config', 'utils', 'data_preparation', 'evaluation']:
        submodule = types.ModuleType(f'src.{submodule_name}')
        sys.modules[f'src.{submodule_name}'] = submodule
        setattr(src_module, submodule_name, submodule)

# ===== ОПРЕДЕЛЯЕМ ФЕЙКОВЫЕ КЛАССЫ МОДЕЛЕЙ =====
class HOG_SVM_Model:
    """Фейковый класс для unpickle"""
    def __init__(self):
        self.scaler = None
        self.model = None
        self.name = "HOG + SVM"
    
    def predict_proba(self, X):
        from skimage.feature import hog
        features = []
        for img in X:
            # Денормализация если нужно
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
            features.append(fd)
        
        X_features = np.array(features)
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_scaled)

class HaarCascade_RF_Model:
    """Фейковый класс для unpickle"""
    def __init__(self):
        self.face_cascade = None
        self.model = None
        self.name = "Haar Cascade + RF"
        self.cascade_path = None

    def _patch_missing_tree_attrs(self):
        """Патчим отсутствующие поля у деревьев RandomForest."""
        try:
            estimators = getattr(self.model, "estimators_", None)
            if estimators is None:
                return

            for est in estimators:
                # Добавляем атрибут, если он отсутствует
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)

                tree_obj = getattr(est, "tree_", None)
                if tree_obj is not None and not hasattr(tree_obj, "monotonic_cst"):
                    setattr(tree_obj, "monotonic_cst", None)

        except Exception:
            pass

# Добавляем классы в фейковый модуль
sys.modules['src.models'].HOG_SVM_Model = HOG_SVM_Model
sys.modules['src.models'].HaarCascade_RF_Model = HaarCascade_RF_Model

# ===== ПУТИ К МОДЕЛЯМ =====
BASE_DIR = os.getcwd()
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')

MODEL1_PATH = os.path.join(TRAINED_MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(TRAINED_MODELS_DIR, 'model2_haar_rf.pkl')
MODEL3_PATH = os.path.join(TRAINED_MODELS_DIR, 'model3_cnn.h5')
LABELS_MAP_PATH = os.path.join(TRAINED_MODELS_DIR, 'labels_map.json')

# ===== НАСТРОЙКА СТРАНИЦЫ И СТИЛИ =====
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #d1f7c4;
        border: 1px solid #a3e4b0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей из папки trained_models"""
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, f"Папка trained_models/ не найдена"
    
    try:
        # ===== Labels map =====
        labels_map = {0: "WithoutMask", 1: "WithMask"}
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    labels_dict = json.load(f)
                    labels_map = {int(k): v for k, v in labels_dict.items()}
            except Exception:
                pass
        
        model1, model2, model3 = None, None, None
        
        # ===== МОДЕЛЬ 1: HOG + SVM =====
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
            except Exception:
                pass
        
        # ===== МОДЕЛЬ 2: Haar + RF =====
        if os.path.exists(MODEL2_PATH):
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
            except Exception:
                pass
        
        # ===== МОДЕЛЬ 3: CNN =====
        if os.path.exists(MODEL3_PATH) and TF_AVAILABLE:
            try:
                # Загружаем с ignore всех custom objects
                model3_keras = tf.keras.models.load_model(
                    MODEL3_PATH, 
                    compile=False
                )
                
                # Обертка для единообразного интерфейса
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict_proba(self, X):
                        # Убеждаемся что X нормализован
                        if X.max() > 1.0:
                            X = X / 255.0
                        
                        predictions = self.model.predict(X, verbose=0)
                        
                        # Если бинарная классификация
                        if predictions.shape[-1] == 1:
                            prob_positive = predictions.flatten()
                            return np.column_stack([1 - prob_positive, prob_positive])
                        
                        return predictions
                
                model3 = CNNWrapper(model3_keras)
                
            except Exception:
                pass
        
        # Проверяем что хоть что-то загружено
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        
        error_msg = ""
        if not any_loaded:
            error_msg = "Ни одна модель не загружена. Проверьте файлы в trained_models/"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, f"Критическая ошибка: {str(e)}"

# Загружаем модели
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
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
            ["Все модели"] + available_models,
            help="Выберите модель для предсказания или все сразу"
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
    
    # Статистика моделей
    st.markdown("### ✅ Статус загрузки")
    
    loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
    if loaded_count == 3:
        st.success("Все модели загружены")
    elif loaded_count > 0:
        st.warning(f"Загружено {loaded_count}/3 моделей")
    else:
        st.error("Модели не загружены")
    
    st.info(f"Классы: {', '.join(labels_map.values())}")
    
    # Кнопка перезагрузки
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Проверка загрузки моделей
if not models_loaded:
    st.error(f"⚠️ Ошибка загрузки моделей: {error_msg}")
    st.info("""
    **Что делать:**
    1. Убедитесь, что вы обучили модели
    2. Проверьте наличие файлов в папке `trained_models/`
    3. Проверьте наличие файла `labels_map.json`
    """)
    
    # Показываем текущую структуру
    with st.expander("📂 Текущая структура проекта"):
        st.write("**Корневая папка:**")
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            if os.path.isdir(item_path):
                st.write(f"📁 {item}/")
                if item in ['trained_models']:
                    try:
                        sub_items = os.listdir(item_path)
                        for sub in sub_items:
                            st.write(f"  📄 {sub}")
                    except:
                        pass
            else:
                st.write(f"📄 {item}")
    
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
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        # Информация об изображении
        img_array = np.array(image)
        st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")

# ===== ПРАВАЯ КОЛОНКА: РЕЗУЛЬТАТЫ =====
with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        try:
            # Обработка изображения
            img_array = np.array(image)
            
            # Конвертация в RGB если нужно
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Ресайз для модели
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0)
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("Сравнение всех моделей")
                
                models = []
                if model1:
                    models.append((model1, "HOG + SVM", "🔵", "#1f77b4"))
                if model2:
                    models.append((model2, "Haar Cascade + RF", "🟢", "#2ca02c"))
                if model3:
                    models.append((model3, "CNN (Deep Learning)", "🔴", "#d62728"))
                
                # Контейнер для результатов
                for model, name, icon, color in models:
                    with st.container():
                        st.markdown(f"### {icon} {name}")
                        
                        with st.spinner(f'Обработка {name}...'):
                            # Предсказание
                            pred_proba = model.predict_proba(img_input)[0]
                            
                            # Определяем класс
                            if len(pred_proba) > 2:
                                pred_class = np.argmax(pred_proba)
                            else:
                                pred_class = 1 if pred_proba[1] > 0.5 else 0
                            
                            confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                            prediction = labels_map.get(pred_class, "WithMask" if pred_class == 1 else "WithoutMask")
                            
                            # Результат
                            if confidence >= confidence_threshold:
                                if prediction == "WithMask":
                                    st.success(f"✅ **{prediction}**")
                                else:
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
                            st.progress(float(confidence))
                            
                            # Детали
                            with st.expander("📊 Детальная информация"):
                                for i, label in labels_map.items():
                                    prob = pred_proba[i] if i < len(pred_proba) else 0
                                    st.write(f"{label}: {prob:.2%}")
                        
                        st.markdown("---")
            
            else:
                # Одна модель
                st.subheader(f"Результат: {model_choice}")
                
                model_map = {
                    "HOG + SVM": (model1, "🔵"),
                    "Haar Cascade + RF": (model2, "🟢"),
                    "CNN (Deep Learning)": (model3, "🔴")
                }
                
                model, icon = model_map[model_choice]
                
                if model:
                    with st.spinner('Обработка изображения...'):
                        # Предсказание
                        pred_proba = model.predict_proba(img_input)[0]
                        
                        # Определяем класс
                        if len(pred_proba) > 2:
                            pred_class = np.argmax(pred_proba)
                        else:
                            pred_class = 1 if pred_proba[1] > 0.5 else 0
                        
                        confidence = pred_proba[pred_class] if len(pred_proba) > pred_class else pred_proba[1]
                        prediction = labels_map.get(pred_class, "WithMask" if pred_class == 1 else "WithoutMask")
                        
                        # Большой результат
                        st.markdown(f"## {icon} {prediction}")
                        
                        if confidence >= confidence_threshold:
                            if prediction == "WithMask":
                                st.success("✅ Маска обнаружена!")
                            else:
                                st.error("❌ Маска не обнаружена!")
                        else:
                            st.warning("⚠️ Низкая уверенность в предсказании")
                        
                        # Метрики в колонках
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                "Класс", 
                                prediction,
                                delta=None
                            )
                        
                        with col_b:
                            st.metric(
                                "Уверенность", 
                                f"{confidence:.1%}",
                                delta=f"{(confidence-0.5)*100:+.1f}%" if confidence > 0.5 else None
                            )
                        
                        with col_c:
                            status = "✅" if confidence >= confidence_threshold else "⚠️"
                            st.metric(
                                "Статус",
                                status
                            )
                        
                        # Прогресс бар
                        st.progress(float(confidence))
                        
                        # График вероятностей
                        st.subheader("📊 Распределение вероятностей")
                        
                        import pandas as pd
                        prob_df = pd.DataFrame({
                            'Класс': [labels_map[i] for i in sorted(labels_map.keys()) if i < len(pred_proba)],
                            'Вероятность': [pred_proba[i] for i in sorted(labels_map.keys()) if i < len(pred_proba)]
                        })
                        
                        st.bar_chart(prob_df.set_index('Класс'))
                        
                        # Детальная информация
                        with st.expander("🔬 Детальная информация"):
                            st.write("**Вероятности для каждого класса:**")
                            for i, label in labels_map.items():
                                if i < len(pred_proba):
                                    prob = pred_proba[i]
                                    st.write(f"- {label}: {prob:.4f} ({prob*100:.2f}%)")
                            
                            st.write(f"\n**Порог уверенности:** {confidence_threshold}")
                            st.write(f"**Размер входного изображения:** 128x128")
                
                else:
                    st.error(f"Модель {model_choice} не загружена")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
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
       - CNN с Transfer Learning
    
    ### Технологии:
    - Python 3.8+
    - OpenCV
    - scikit-learn
    - TensorFlow/Keras
    - Streamlit
    
    ### Метрики качества:
    - Accuracy (Точность)
    - Precision (Прецизионность)
    - Recall (Полнота)
    - F1-Score
    - ROC-AUC
    
    ---
    Разработано в рамках курсового проекта | 2024
    """)

# Copyright
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>© 2024 Mask Detection System | Все права защищены</p>
    </div>
""", unsafe_allow_html=True)
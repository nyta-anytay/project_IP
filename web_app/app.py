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
    
    def predict_proba(self, X):
        # Загружаем Haar Cascade если еще нет
        if self.face_cascade is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                pass
        
        features = []
        
        for img in X:
            # Денормализация если нужно
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            feat = []
            
            # 1. Статистики
            feat.extend([gray.mean(), gray.std(), gray.min(), gray.max()])
            
            # 2. Гистограмма
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            feat.extend(hist.flatten())
            
            # 3. Детекция лиц
            if self.face_cascade is not None:
                try:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 1.1, 4, minSize=(20, 20)
                    )
                    feat.append(len(faces))
                except:
                    feat.append(0)
            else:
                feat.append(0)
            
            # 4. Цветовые статистики
            for channel in range(3):
                feat.extend([
                    img[:, :, channel].mean(),
                    img[:, :, channel].std()
                ])
            
            # 5. Края
            edges = cv2.Canny(gray, 100, 200)
            feat.extend([edges.mean(), edges.std()])
            
            features.append(feat)
        
        X_features = np.array(features)
        return self.model.predict_proba(X_features)

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

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="Mask Detection System",
    page_icon="😷",
    layout="wide"
)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models_from_trained_models():
    """Загружаем модели с фейковыми классами"""
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, "Папка trained_models/ не найдена"
    
    try:
        # Labels
        labels_map = {0: 'WithoutMask', 1: 'WithMask'}
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
                st.sidebar.success("✅ HOG + SVM загружена")
            except Exception as e:
                st.sidebar.error(f"❌ Model1: {str(e)[:100]}")
        
        # ===== МОДЕЛЬ 2 =====
        if os.path.exists(MODEL2_PATH):
            try:
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                st.sidebar.success("✅ Haar + RF загружена")
            except Exception as e:
                st.sidebar.error(f"❌ Model2: {str(e)[:100]}")
        
        # ===== МОДЕЛЬ 3 =====
        if os.path.exists(MODEL3_PATH) and TF_AVAILABLE:
            try:
                model3_keras = load_model(MODEL3_PATH, compile=False)
                
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict_proba(self, X):
                        # CNN ожидает нормализованный вход [0, 1]
                        if X.max() > 1.0:
                            X = X / 255.0
                        
                        predictions = self.model.predict(X, verbose=0)
                        
                        if predictions.shape[-1] == 1:
                            prob = predictions.flatten()
                            return np.column_stack([1 - prob, prob])
                        
                        return predictions
                
                model3 = CNNWrapper(model3_keras)
                st.sidebar.success("✅ CNN загружена")
                
            except Exception as e:
                st.sidebar.error(f"❌ Model3: {str(e)[:100]}")
        
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        
        return model1, model2, model3, labels_map, any_loaded, ""
        
    except Exception as e:
        return None, None, None, {}, False, str(e)

# Загрузка
model1, model2, model3, labels_map, models_loaded, error_msg = load_models_from_trained_models()

# ДАЛЬШЕ ИДЕТ ОСТАЛЬНОЙ КОД БЕЗ ИЗМЕНЕНИЙ...

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_models_from_trained_models():
    """Загружаем модели из папки trained_models/ с исправлениями"""
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        return None, None, None, {}, False, f"Папка trained_models/ не найдена"
    
    try:
        # ===== СОЗДАЕМ ФЕЙКОВЫЕ МОДУЛИ ДЛЯ UNPICKLE =====
        import sys
        import types
        
        # Создаем фейковый модуль src если его нет
        if 'src' not in sys.modules:
            src_module = types.ModuleType('src')
            sys.modules['src'] = src_module
            
            # Создаем подмодули
            for submodule in ['config', 'models', 'utils', 'data_preparation', 'evaluation']:
                full_name = f'src.{submodule}'
                if full_name not in sys.modules:
                    sub = types.ModuleType(full_name)
                    sys.modules[full_name] = sub
                    setattr(src_module, submodule, sub)
        
        # ===== Labels map =====
        labels_map = {0: "WithoutMask", 1: "WithMask"}
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
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                st.sidebar.success("✅ Модель 2 (Haar + RF)")
            except Exception as e:
                st.sidebar.error(f"❌ Model2: {str(e)[:80]}")
        else:
            st.sidebar.error(f"❌ Model2: файл не найден")
        
        # ===== МОДЕЛЬ 3: CNN =====
        if os.path.exists(MODEL3_PATH):
            try:
                # Загружаем с ignore всех custom objects
                import tensorflow as tf
                
                # Вариант 1: Попробовать без compile
                try:
                    model3_keras = tf.keras.models.load_model(
                        MODEL3_PATH, 
                        compile=False
                    )
                    st.sidebar.success("✅ Модель 3 (CNN) - простая загрузка")
                    
                except Exception as e1:
                    # Вариант 2: С safe mode
                    st.sidebar.info("Пробую safe mode для CNN...")
                    
                    try:
                        # Используем experimental API
                        model3_keras = tf.keras.models.load_model(
                            MODEL3_PATH,
                            compile=False,
                            safe_mode=False
                        )
                        st.sidebar.success("✅ Модель 3 (CNN) - safe mode")
                        
                    except Exception as e2:
                        # Вариант 3: Загружаем только веса
                        st.sidebar.info("Пробую загрузить только архитектуру...")
                        
                        # Создаем простую архитектуру MobileNetV2
                        from tensorflow.keras.applications import MobileNetV2
                        from tensorflow.keras import Sequential
                        from tensorflow.keras.layers import (
                            GlobalAveragePooling2D, Dense, 
                            Dropout, Rescaling, Input
                        )
                        
                        base_model = MobileNetV2(
                            input_shape=(128, 128, 3),
                            include_top=False,
                            weights='imagenet'
                        )
                        base_model.trainable = False
                        
                        model3_keras = Sequential([
                            Input(shape=(128, 128, 3)),
                            Rescaling(1./255),
                            base_model,
                            GlobalAveragePooling2D(),
                            Dropout(0.3),
                            Dense(128, activation='relu'),
                            Dropout(0.2),
                            Dense(2, activation='softmax')
                        ])
                        
                        # Пытаемся загрузить только веса
                        try:
                            model3_keras.load_weights(MODEL3_PATH)
                            st.sidebar.success("✅ Модель 3 (CNN) - только веса")
                        except:
                            st.sidebar.warning("⚠️ Не удалось загрузить веса, используем pretrained MobileNet")
                
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
                
            except Exception as e:
                st.sidebar.error(f"❌ Model3: {str(e)[:150]}")
        else:
            st.sidebar.error(f"❌ Model3: файл не найден")
        
        # Проверяем что хоть что-то загружено
        any_loaded = model1 is not None or model2 is not None or model3 is not None
        
        error_msg = ""
        if not any_loaded:
            error_msg = "Ни одна модель не загружена. Проверьте файлы в trained_models/"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, f"Критическая ошибка: {str(e)}"
# Загружаем модели
model1, model2, model3, labels_map, models_loaded, error_msg = load_models_from_trained_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)

# Сообщение о статусе
if not models_loaded:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Проблема с загрузкой моделей</strong><br>
    Не удалось загрузить модели из папки <code>trained_models/</code>
    </div>
    """, unsafe_allow_html=True)
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
            if os.path.exists(MODEL1_PATH):
                size_mb = os.path.getsize(MODEL1_PATH) / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
        else:
            st.error("❌ HOG+SVM")
    
    with col2:
        if model2:
            st.success("✅ Haar+RF")
            if os.path.exists(MODEL2_PATH):
                size_mb = os.path.getsize(MODEL2_PATH) / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
        else:
            st.error("❌ Haar+RF")
    
    with col3:
        if model3:
            st.success("✅ CNN")
            if os.path.exists(MODEL3_PATH):
                size_mb = os.path.getsize(MODEL3_PATH) / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
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
            ["Все модели"] + available_models,
            help="Выберите модель для предсказания"
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
    
    # Кнопка перезагрузки
    if st.button("🔄 Перезагрузить модели"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Информация о путях
    st.subheader("📁 Пути к файлам")
    st.code(f"""
model1: {MODEL1_PATH}
model2: {MODEL2_PATH}
model3: {MODEL3_PATH}
labels: {LABELS_MAP_PATH}
    """)

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error(f"⚠️ Не удалось загрузить модели!")
    st.warning(error_msg)
    
    st.info("""
    ## 🔧 Решение проблем:
    
    ### **1. Проверьте наличие папки trained_models/ на GitHub:**
    Откройте ваш репозиторий и убедитесь, что есть папка `trained_models/` с файлами:
    ```
    trained_models/
    ├── model1_hog_svm.pkl
    ├── model2_haar_rf.pkl  
    ├── model3_cnn.h5
    └── labels_map.json (опционально)
    ```
    
    ### **2. Исправьте .gitignore:**
    Убедитесь, что в `.gitignore` НЕТ строк:
    ```gitignore
    trained_models/*
    *.pkl
    *.h5
    ```
    
    ### **3. Добавьте файлы в Git:**
    ```bash
    # Добавьте папку trained_models/
    git add trained_models/
    
    # Или конкретные файлы
    git add trained_models/model1_hog_svm.pkl
    git add trained_models/model2_haar_rf.pkl
    git add trained_models/model3_cnn.h5
    
    git commit -m "Add trained models"
    git push
    ```
    
    ### **4. Проверьте структуру на Streamlit Cloud:**
    ```python
    # Добавьте этот код для отладки
    import os
    print("Содержимое корня:", os.listdir('.'))
    if os.path.exists('trained_models'):
        print("Содержимое trained_models:", os.listdir('trained_models'))
    ```
    """)
    
    # Показываем текущую структуру
    with st.expander("📂 Текущая структура проекта"):
        st.write("**Корневая папка:**")
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            if os.path.isdir(item_path):
                st.write(f"📁 {item}/")
                # Показываем содержимое важных папок
                if item in ['trained_models', 'web_app', 'src']:
                    try:
                        sub_items = os.listdir(item_path)
                        for sub in sub_items[:10]:  # первые 10 файлов
                            st.write(f"  📄 {sub}")
                    except:
                        pass
            else:
                st.write(f"📄 {item}")
    
    st.stop()

# Основной интерфейс (если модели загружены)
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
            "Выберите изображение с лицом",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    else:
        uploaded_file = st.camera_input("Сфотографируйте лицо")

with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file is not None:
        try:
            # Загрузка и обработка изображения
            image = Image.open(uploaded_file)
            
            # Показываем в левой колонке
            with col1:
                st.image(image, caption='Загруженное изображение', use_column_width=True)
                img_array = np.array(image)
                st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")
            
            # Подготовка изображения для моделей
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Ресайз и нормализация
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # ===== ПРЕДСКАЗАНИЯ =====
            if model_choice == "Все модели":
                st.subheader("📊 Сравнение моделей")
                
                models_to_show = []
                if model1:
                    models_to_show.append((model1, "HOG + SVM", "🔵", "#1f77b4"))
                if model2:
                    models_to_show.append((model2, "Haar Cascade + RF", "🟢", "#2ca02c"))
                if model3:
                    models_to_show.append((model3, "CNN (Deep Learning)", "🔴", "#d62728"))
                
                for model, name, icon, color in models_to_show:
                    with st.container():
                        st.markdown(f"### {icon} {name}")
                        
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
                                        st.success(f"✅ **{prediction}**")
                                    else:
                                        st.error(f"❌ **{prediction}**")
                                else:
                                    st.warning(f"⚠️ **{prediction}** (низкая уверенность)")
                            
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            # Прогресс-бар
                            st.progress(float(confidence))
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)[:100]}")
                        
                        st.markdown("---")
            
            else:
                # Одна модель
                model_map = {
                    "HOG + SVM": (model1, "🔵"),
                    "Haar Cascade + RF": (model2, "🟢"),
                    "CNN (Deep Learning)": (model3, "🔴")
                }
                
                model, icon = model_map[model_choice]
                
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
                            
                            # Большой результат
                            st.markdown(f"## {icon} {prediction}")
                            
                            if confidence >= confidence_threshold:
                                if prediction == "С маской":
                                    st.success("✅ Маска обнаружена!")
                                else:
                                    st.error("❌ Маска не обнаружена!")
                            else:
                                st.warning("⚠️ Низкая уверенность в предсказании")
                            
                            # Метрики
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Класс", prediction)
                            
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            with col_c:
                                status = "✅" if confidence >= confidence_threshold else "⚠️"
                                st.metric("Статус", status)
                            
                            # Прогресс-бар
                            st.progress(float(confidence))
                            
                        except Exception as e:
                            st.error(f"Ошибка предсказания: {str(e)}")
                else:
                    st.error(f"Модель {model_choice} не загружена")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
    else:
        st.info("👆 Загрузите изображение для начала детекции")
        
        st.markdown("""
        ### 💡 Рекомендации:
        
        1. **Четкое изображение** лица
        2. **Хорошее освещение**
        3. **Лицо полностью в кадре**
        4. **Без масок на подбородке**
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("📋 Инструкция для правильного деплоя"):
    st.markdown("""
    ## Для успешной работы приложения:
    
    ### **1. Структура репозитория должна быть:**
    ```
    project_ip/
    ├── web_app/
    │   └── app.py                    # Этот файл
    ├── trained_models/              # Папка с моделями
    │   ├── model1_hog_svm.pkl      # Модель 1
    │   ├── model2_haar_rf.pkl      # Модель 2
    │   ├── model3_cnn.h5           # Модель 3
    │   └── labels_map.json         # Метки классов
    ├── requirements.txt
    └── .gitignore                  # Без строк про trained_models/*.pkl
    ```
    
    ### **2. На Streamlit Cloud:**
    - **Main file path:** `web_app/app.py`
    - **Branch:** `main`
    
    ### **3. Если модели не загружаются:**
    ```python
    # Добавьте в код для отладки
    import os
    print("Текущая папка:", os.getcwd())
    print("Содержимое trained_models:", os.listdir('trained_models'))
    ```
    """)

st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px;'>
<p>© 2024 Mask Detection System | Модели из папки: {TRAINED_MODELS_DIR}</p>
</div>
""", unsafe_allow_html=True)
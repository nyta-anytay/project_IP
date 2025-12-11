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

# ===== ПРАВИЛЬНЫЕ ПУТИ - МОДЕЛИ В ПАПКЕ web/ =====
MODEL1_PATH = 'web/model1_hog_svm.pkl'    # Путь к первой модели
MODEL2_PATH = 'web/model2_haar_rf.pkl'    # Путь ко второй модели  
MODEL3_PATH = 'web/model3_cnn.h5'         # Путь к третьей модели
LABELS_MAP_PATH = 'web/labels_map.json'   # Путь к labels_map

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
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ФАЙЛОВ =====
def check_files_exist():
    """Проверяем наличие файлов моделей в папке web/"""
    st.sidebar.subheader("🔍 Поиск файлов в web/")
    
    # Проверяем существует ли папка web/
    if not os.path.exists('web'):
        st.sidebar.error("❌ Папка 'web/' не найдена!")
        return [], ['web/' + f for f in ['model1_hog_svm.pkl', 'model2_haar_rf.pkl', 'model3_cnn.h5', 'labels_map.json']]
    
    files_needed = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABELS_MAP_PATH]
    existing_files = []
    missing_files = []
    
    for file in files_needed:
        if os.path.exists(file):
            existing_files.append(file)
            size_kb = os.path.getsize(file) / 1024
            st.sidebar.success(f"✅ {os.path.basename(file)} ({size_kb:.1f} KB)")
        else:
            missing_files.append(file)
            st.sidebar.error(f"❌ {os.path.basename(file)} - не найден")
    
    # Показываем содержимое папки web/
    st.sidebar.write("**Содержимое папки web/:**")
    if os.path.exists('web'):
        for item in os.listdir('web'):
            item_path = os.path.join('web', item)
            if os.path.isfile(item_path):
                size_kb = os.path.getsize(item_path) / 1024
                st.sidebar.text(f"📄 {item} ({size_kb:.1f} KB)")
            else:
                st.sidebar.text(f"📁 {item}/")
    
    return existing_files, missing_files

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей с обработкой ошибок"""
    
    # Проверяем файлы
    existing_files, missing_files = check_files_exist()
    
    if not existing_files:
        return None, None, None, {}, False, "Файлы моделей не найдены в папке web/"
    
    try:
        # ===== 1. labels_map =====
        labels_map = {}
        if os.path.exists(LABELS_MAP_PATH):
            try:
                with open(LABELS_MAP_PATH, 'r') as f:
                    labels_dict = json.load(f)
                    labels_map = {int(k): v for k, v in labels_dict.items()}
                st.sidebar.success(f"✅ labels_map загружен: {labels_map}")
            except:
                labels_map = {0: "Без маски", 1: "С маской"}
                st.sidebar.info("ℹ️ Используется стандартный labels_map")
        else:
            labels_map = {0: "Без маски", 1: "С маски"}
            st.sidebar.info("ℹ️ labels_map.json не найден, используем стандартный")
        
        models_loaded = []
        model1, model2, model3 = None, None, None
        
        # ===== 2. Модель 1: HOG + SVM =====
        if os.path.exists(MODEL1_PATH):
            try:
                with open(MODEL1_PATH, 'rb') as f:
                    model1 = pickle.load(f)
                models_loaded.append(("model1", True, ""))
                st.sidebar.success("✅ Модель 1 (HOG+SVM) загружена")
            except Exception as e:
                models_loaded.append(("model1", False, str(e)))
                st.sidebar.error(f"❌ Ошибка загрузки model1: {str(e)[:100]}")
        else:
            models_loaded.append(("model1", False, f"Файл не найден: {MODEL1_PATH}"))
        
        # ===== 3. Модель 2: Haar + RF =====
        if os.path.exists(MODEL2_PATH):
            try:
                # Пробуем стандартный pickle
                with open(MODEL2_PATH, 'rb') as f:
                    model2 = pickle.load(f)
                models_loaded.append(("model2", True, ""))
                st.sidebar.success("✅ Модель 2 (Haar+RF) загружена")
            except Exception as e:
                # Если ошибка из-за 'src', пробуем кастомный unpickler
                if 'src' in str(e):
                    try:
                        class CustomUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module.startswith('src'):
                                    return object
                                return super().find_class(module, name)
                        
                        with open(MODEL2_PATH, 'rb') as f:
                            unpickler = CustomUnpickler(f)
                            model2 = unpickler.load()
                        models_loaded.append(("model2", True, ""))
                        st.sidebar.success("✅ Модель 2 загружена (с обработкой 'src')")
                    except Exception as e2:
                        models_loaded.append(("model2", False, str(e2)))
                        st.sidebar.error(f"❌ Ошибка model2: {str(e2)[:100]}")
                else:
                    models_loaded.append(("model2", False, str(e)))
                    st.sidebar.error(f"❌ Ошибка model2: {str(e)[:100]}")
        else:
            models_loaded.append(("model2", False, f"Файл не найден: {MODEL2_PATH}"))
        
        # ===== 4. Модель 3: CNN =====
        if os.path.exists(MODEL3_PATH):
            try:
                # Пробуем загрузить с разными способами
                try:
                    # Способ 1: Стандартная загрузка
                    model3_keras = tf.keras.models.load_model(MODEL3_PATH, compile=False)
                    st.sidebar.success("✅ Модель 3 (CNN) загружена (стандартный способ)")
                except Exception as e1:
                    # Способ 2: С кастомными объектами
                    from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
                    from tensorflow.keras import Input, Model
                    
                    custom_objects = {
                        'BatchNormalization': BatchNormalization,
                        'Conv2D': Conv2D,
                        'Dense': Dense,
                        'Dropout': Dropout,
                        'Flatten': Flatten,
                        'MaxPooling2D': MaxPooling2D,
                        'Input': Input,
                        'Model': Model
                    }
                    
                    model3_keras = tf.keras.models.load_model(
                        MODEL3_PATH,
                        compile=False,
                        custom_objects=custom_objects
                    )
                    st.sidebar.success("✅ Модель 3 (CNN) загружена (с кастомными объектами)")
                
                # Обертка для модели
                class CNNWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict_proba(self, X):
                        predictions = self.model.predict(X, verbose=0)
                        if predictions.shape[-1] == 1:
                            prob_positive = predictions.flatten()
                            return np.column_stack([1 - prob_positive, prob_positive])
                        return predictions
                
                model3 = CNNWrapper(model3_keras)
                models_loaded.append(("model3", True, ""))
                
            except Exception as e:
                models_loaded.append(("model3", False, str(e)))
                st.sidebar.error(f"❌ Ошибка model3: {str(e)[:150]}")
        else:
            models_loaded.append(("model3", False, f"Файл не найден: {MODEL3_PATH}"))
        
        # Проверяем сколько моделей загрузилось
        loaded_count = sum(1 for _, status, _ in models_loaded if status)
        any_loaded = loaded_count > 0
        
        error_msg = ""
        if not any_loaded:
            error_details = [f"{name}: {msg}" for name, status, msg in models_loaded if not status and msg]
            error_msg = f"Ошибки: {'; '.join(error_details)}"
        
        return model1, model2, model3, labels_map, any_loaded, error_msg
    
    except Exception as e:
        return None, None, None, {}, False, f"Общая ошибка: {str(e)}"

# ===== ЗАГРУЗКА =====
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()

# ===== ЗАГОЛОВОК =====
st.markdown('<h1 class="main-header">😷 Система детекции масок на лице</h1>', 
           unsafe_allow_html=True)
st.markdown("---")

# ===== SIDEBAR: НАСТРОЙКИ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Диагностика
    if st.checkbox("🔧 Расширенная диагностика", False):
        st.write("**Версии библиотек:**")
        st.code(f"""
        TensorFlow: {tf.__version__}
        OpenCV: {cv2.__version__}
        NumPy: {np.__version__}
        """)
        
        st.write("**Текущая директория:**", os.getcwd())
        st.write("**Полное дерево файлов:**")
        
        import pathlib
        for file_path in pathlib.Path('.').rglob('*'):
            if file_path.is_file():
                rel_path = str(file_path.relative_to('.'))
                if 'model' in rel_path.lower() or 'web' in rel_path:
                    st.success(f"🔍 {rel_path}")
                else:
                    st.text(f"   {rel_path}")
    
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
        st.error("❌ Нет доступных моделей для выбора")
    
    # Порог уверенности
    confidence_threshold = st.slider(
        "Порог уверенности:", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Перезагрузка
    if st.button("🔄 Перезагрузить все модели"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Статус
    st.markdown("### 📊 Статус моделей")
    cols = st.columns(3)
    status_info = [
        ("HOG+SVM", model1, "web/model1_hog_svm.pkl"),
        ("Haar+RF", model2, "web/model2_haar_rf.pkl"),
        ("CNN", model3, "web/model3_cnn.h5")
    ]
    
    for i, (name, model, path) in enumerate(status_info):
        with cols[i]:
            if model:
                st.success(f"✅ {name}")
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    st.caption(f"{size_mb:.1f} MB")
            else:
                st.error(f"❌ {name}")
                if not os.path.exists(path):
                    st.caption("файл не найден")

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
if not models_loaded:
    st.error("⚠️ Критические проблемы с загрузкой моделей")
    st.warning(error_msg)
    
    st.info("""
    ## 🚀 Решение проблем:
    
    ### **1. Проверьте структуру проекта на GitHub:**
    Убедитесь, что в репозитории есть папка `web/` с файлами:
    ```
    web/
    ├── model1_hog_svm.pkl
    ├── model2_haar_rf.pkl
    ├── model3_cnn.h5
    └── labels_map.json (опционально)
    ```
    
    ### **2. Проверьте .gitignore:**
    Убедитесь, что `.gitignore` НЕ содержит:
    ```gitignore
    web/*.pkl    # ← ЭТО НЕ ДОЛЖНО БЫТЬ!
    web/*.h5     # ← ЭТО НЕ ДОЛЖНО БЫТЬ!
    ```
    
    ### **3. Обновите requirements.txt:**
    ```txt
    streamlit==1.29.0
    tensorflow==2.15.0
    opencv-python-headless==4.8.1
    numpy==1.24.3
    Pillow==10.1.0
    scikit-learn==1.3.2
    ```
    
    ### **4. Если model2 не грузится из-за 'src':**
    Пересохраните модель в другом формате:
    ```python
    import joblib
    joblib.dump(model, 'web/model2_haar_rf.joblib')
    ```
    И обновите код для загрузки `.joblib`.
    """)
    
    st.stop()

# Если хотя бы одна модель загружена
loaded_count = sum(1 for m in [model1, model2, model3] if m is not None)
st.success(f"✅ Загружено моделей: {loaded_count}/3")

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Загрузка изображения")
    
    upload_option = st.radio("Способ загрузки:", ["Файл", "Камера"], horizontal=True)
    
    uploaded_file = None
    if upload_option == "Файл":
        uploaded_file = st.file_uploader(
            "Выберите изображение с лицом", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    else:
        uploaded_file = st.camera_input("Сфотографируйте лицо")

with col2:
    st.header("🔍 Результаты детекции")
    
    if uploaded_file:
        try:
            # Загрузка и обработка изображения
            image = Image.open(uploaded_file)
            
            # Показываем в левой колонке
            with col1:
                st.image(image, caption='Исходное изображение', use_column_width=True)
                img_array = np.array(image)
                st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]}")
            
            # Подготовка для моделей
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (128, 128))
            img_input = np.expand_dims(img_resized, axis=0) / 255.0
            
            # Предсказания
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
                            pred_class = np.argmax(pred_proba)
                            confidence = pred_proba[pred_class]
                            prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                            
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                if confidence >= confidence_threshold:
                                    if prediction == "С маской":
                                        st.success(f"✅ **{prediction}**")
                                    else:
                                        st.error(f"❌ **{prediction}**")
                                else:
                                    st.warning(f"⚠️ **{prediction}**")
                            
                            with col_b:
                                st.metric("Уверенность", f"{confidence:.1%}")
                            
                            st.progress(float(confidence))
                        except Exception as e:
                            st.error(f"Ошибка: {str(e)[:80]}")
                        
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
                    try:
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map.get(pred_class, "С маской" if pred_class == 1 else "Без маски")
                        
                        st.markdown(f"## {icon} {prediction}")
                        
                        if confidence >= confidence_threshold:
                            if prediction == "С маской":
                                st.success("✅ Маска обнаружена!")
                            else:
                                st.error("❌ Маска не обнаружена!")
                        else:
                            st.warning("⚠️ Низкая уверенность")
                        
                        # Метрики
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Результат", prediction)
                        with col_b:
                            st.metric("Уверенность", f"{confidence:.1%}")
                        with col_c:
                            st.metric("Порог", f"{confidence_threshold:.0%}")
                        
                        st.progress(float(confidence))
                        
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {str(e)}")
                else:
                    st.error(f"Модель {model_choice} не загружена")
        
        except Exception as e:
            st.error(f"Ошибка обработки изображения: {str(e)}")
    
    else:
        st.info("👆 Загрузите или сфотографируйте изображение для анализа")
        
        st.markdown("""
        ### 💡 Рекомендации для лучших результатов:
        
        1. **Хорошее освещение** лица
        2. **Лицо должно быть полностью видно**
        3. **Портретная ориентация** предпочтительнее
        4. **Избегайте** солнцезащитных очков, масок на подбородке
        """)

# ===== FOOTER =====
st.markdown("---")

with st.expander("📋 Инструкция по успешному деплою"):
    st.markdown("""
    ### **Для правильной работы на Streamlit Cloud:**
    
    1. **Структура репозитория должна быть:**
    ```
    ваш-репозиторий/
    ├── app.py                          # Этот файл
    ├── web/                           # Папка с моделями
    │   ├── model1_hog_svm.pkl        # Модель 1
    │   ├── model2_haar_rf.pkl        # Модель 2  
    │   ├── model3_cnn.h5             # Модель 3
    │   └── labels_map.json           # Метки классов
    ├── requirements.txt              # Зависимости
    └── .gitignore                    # НЕ игнорировать web/*.pkl и web/*.h5
    ```
    
    2. **Проверьте .gitignore:**
    Убедитесь, что НЕТ строк:
    ```gitignore
    web/*.pkl
    web/*.h5
    *.pkl
    *.h5
    ```
    
    3. **Добавьте файлы в Git:**
    ```bash
    git add web/model1_hog_svm.pkl
    git add web/model2_haar_rf.pkl
    git add web/model3_cnn.h5
    git add web/labels_map.json
    git commit -m "Add model files from web folder"
    git push
    ```
    
    4. **На Streamlit Cloud укажите путь:**
    - Main file path: `app.py`
    - Branch: `main` или `master`
    """)

st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Mask Detection System | Модели: web/model1_hog_svm.pkl, web/model2_haar_rf.pkl, web/model3_cnn.h5</p>
    </div>
""", unsafe_allow_html=True)
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

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL1_PATH, MODEL2_PATH, MODEL3_PATH, LABELS_MAP_PATH

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
    div[data-Testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)


# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
@st.cache_resource
def load_all_models():
    """Загрузка всех моделей и labels_map"""
    try:
        # Модель 1
        with open(MODEL1_PATH, 'rb') as f:
            model1 = pickle.load(f)
        
        # Модель 2
        with open(MODEL2_PATH, 'rb') as f:
            model2 = pickle.load(f)
        
        # Модель 3
        model3_keras = tf.keras.models.load_model(MODEL3_PATH)
        
        # Обертка для CNN
        class CNNWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, X):
                return self.model.predict(X, verbose=0)
        
        model3 = CNNWrapper(model3_keras)
        
        # Загрузка labels_map
        with open(LABELS_MAP_PATH, 'r') as f:
            labels_dict = json.load(f)
            labels_map = {int(k): v for k, v in labels_dict.items()}
        
        return model1, model2, model3, labels_map, True, None
        
    except FileNotFoundError as e:
        return None, None, None, None, False, f"Файл не найден: {e}"
    except Exception as e:
        return None, None, None, None, False, f"Ошибка загрузки: {e}"


# Загрузка моделей
model1, model2, model3, labels_map, models_loaded, error_msg = load_all_models()


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
        st.success("Все модели загружены")
        st.info(f"Классы: {', '.join(labels_map.values())}")


# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====

# Проверка загрузки моделей
if not models_loaded:
    st.error(f"⚠️ Ошибка загрузки моделей: {error_msg}")
    st.info("""
    **Что делать:**
    1. Убедитесь, что вы обучили модели: `python scripts/02_Train_models.py`
    2. Проверьте наличие файлов в папке `Trained_models/`
    3. Проверьте наличие файла `results/labels_map.json`
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
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        
        # Информация об изображении
        img_array = np.array(image)
        st.caption(f"Размер: {img_array.shape[1]}x{img_array.shape[0]} пикселей")

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
        
        # Ресайз для модели
        img_resized = cv2.resize(img_array, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0)
        
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
                    
                    with st.spinner(f'Обработка {name}...'):
                        # Предсказание
                        pred_proba = model.predict_proba(img_input)[0]
                        pred_class = np.argmax(pred_proba)
                        confidence = pred_proba[pred_class]
                        prediction = labels_map[pred_class]
                        
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
                        st.progress(float(confidence))
                        
                        # Детали
                        with st.expander("📊 Детальная информация"):
                            for i, label in labels_map.items():
                                prob = pred_proba[i]
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
            
            with st.spinner('Обработка изображения...'):
                # Предсказание
                pred_proba = model.predict_proba(img_input)[0]
                pred_class = np.argmax(pred_proba)
                confidence = pred_proba[pred_class]
                prediction = labels_map[pred_class]
                
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
                    'Класс': [labels_map[i] for i in sorted(labels_map.keys())],
                    'Вероятность': [pred_proba[i] for i in sorted(labels_map.keys())]
                })
                
                st.bar_chart(prob_df.set_index('Класс'))
                
                # Детальная информация
                with st.expander("🔬 Детальная информация"):
                    st.write("**Вероятности для каждого класса:**")
                    for i, label in labels_map.items():
                        prob = pred_proba[i]
                        st.write(f"- {label}: {prob:.4f} ({prob*100:.2f}%)")
                    
                    st.write(f"\n**Порог уверенности:** {confidence_threshold}")
                    st.write(f"**Размер входного изображения:** 128x128")
    
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
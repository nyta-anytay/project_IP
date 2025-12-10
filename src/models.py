"""
Модели для классификации изображений
"""
import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from .config import HAAR_CASCADE_PATH, CNN_LEARNING_RATE


class HOG_SVM_Model:
    """
    Модель на основе HOG (Histogram of Oriented Gradients) признаков 
    и SVM (Support Vector Machine) классификатора
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel='rbf', 
            probability=True, 
            random_state=42,
            verbose=False
        )
        self.name = "HOG + SVM"
        
    def extract_hog_features(self, images):
        """
        Извлечение HOG признаков из изображений
        
        Args:
            images: numpy array изображений
            
        Returns:
            numpy array с HOG признаками
        """
        features = []
        
        for img in images:
            # Конвертация в оттенки серого
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Извлечение HOG признаков
            fd = hog(
                gray, 
                orientations=9,           # Количество ориентаций
                pixels_per_cell=(8, 8),   # Размер ячейки
                cells_per_block=(2, 2),   # Размер блока
                visualize=False,
                channel_axis=None
            )
            features.append(fd)
        
        return np.array(features)
    
    def Train(self, X_Train, y_Train):
        """Обучение модели"""
        print(f"\n{'='*70}")
        print(f"ОБУЧЕНИЕ: {self.name}")
        print(f"{'='*70}")
        
        print("Этап 1/3: Извлечение HOG признаков...")
        X_Train_hog = self.extract_hog_features(X_Train)
        print(f"  Форма признаков: {X_Train_hog.shape}")
        
        print("Этап 2/3: Нормализация признаков...")
        X_Train_scaled = self.scaler.fit_transform(X_Train_hog)
        
        print("Этап 3/3: Обучение SVM классификатора...")
        self.model.fit(X_Train_scaled, y_Train)
        
        print(f"✓ {self.name} успешно обучена!")
        print("="*70)
        
    def predict(self, X):
        """Предсказание классов"""
        X_hog = self.extract_hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Предсказание вероятностей классов"""
        X_hog = self.extract_hog_features(X)
        X_scaled = self.scaler.transform(X_hog)
        return self.model.predict_proba(X_scaled)


class HaarCascade_RF_Model:
    """
    Модель на основе Haar Cascade детектора лиц и Random Forest
    """
    
    def __init__(self):
        # Загрузка Haar Cascade
        self.face_cascade = self._load_haar_cascade()
        self.cascade_path = None  # Сохраним путь для повторной загрузки
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            verbose=0,
            n_jobs=-1
        )
        self.name = "Haar Cascade + Random Forest"
    
    def __getstate__(self):
        """
        Кастомная сериализация для pickle
        Исключаем face_cascade из сохранения
        """
        state = self.__dict__.copy()
        # Удаляем объект cascade, который нельзя pickle'ить
        state['face_cascade'] = None
        return state
    
    def __setstate__(self, state):
        """
        Кастомная десериализация для pickle
        Восстанавливаем face_cascade при загрузке
        """
        self.__dict__.update(state)
        # Загружаем cascade заново
        self.face_cascade = self._load_haar_cascade()
    
    def _load_haar_cascade(self):
        """Загрузка Haar Cascade классификатора"""
        # Попробуем несколько путей
        cascade_paths = [
            HAAR_CASCADE_PATH,  # Из config
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',  # OpenCV
            'haarcascade_frontalface_default.xml'  # Текущая директория
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self.cascade_path = path  # Сохраняем рабочий путь
                    return cascade
        
        # Если не нашли, пытаемся скачать
        print("⚠️  Haar Cascade не найден, пытаюсь скачать...")
        self._download_haar_cascade()
        
        cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if cascade.empty():
            raise IOError(
                "Не удалось загрузить Haar Cascade! "
                "Запустите: python scripts/download_resources.py"
            )
        
        self.cascade_path = HAAR_CASCADE_PATH
        return cascade
    
    def _download_haar_cascade(self):
        """Загрузка Haar Cascade файла"""
        import urllib.request
        
        url = ("https://raw.githubusercontent.com/opencv/opencv/master/"
               "data/haarcascades/haarcascade_frontalface_default.xml")
        
        try:
            urllib.request.urlretrieve(url, HAAR_CASCADE_PATH)
            print(f"✓ Haar Cascade загружен: {HAAR_CASCADE_PATH}")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
    
    def extract_features(self, images):
        """
        Извлечение признаков из изображений
        
        Признаки:
        - Статистики в оттенках серого
        - Гистограмма
        - Количество обнаруженных лиц
        - Цветовые статистики
        - Края (Canny)
        """
        features = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            feat = []
            
            # 1. Статистики в оттенках серого (4 признака)
            feat.extend([
                gray.mean(),
                gray.std(),
                gray.min(),
                gray.max()
            ])
            
            # 2. Гистограмма (32 признака)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            feat.extend(hist.flatten())
            
            # 3. Обнаружение лиц (1 признак)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=4,
                minSize=(20, 20)
            )
            feat.append(len(faces))
            
            # 4. Цветовые статистики RGB (6 признаков)
            for channel in range(3):
                feat.extend([
                    img[:, :, channel].mean(),
                    img[:, :, channel].std()
                ])
            
            # 5. Края Canny (2 признака)
            edges = cv2.Canny(gray, 100, 200)
            feat.extend([edges.mean(), edges.std()])
            
            features.append(feat)
        
        return np.array(features)
    
    def train(self, X_train, y_train):
        """Обучение модели"""
        print(f"\n{'='*70}")
        print(f"ОБУЧЕНИЕ: {self.name}")
        print(f"{'='*70}")
        
        print("Этап 1/2: Извлечение признаков...")
        X_features = self.extract_features(X_train)
        print(f"  Форма признаков: {X_features.shape}")
        
        print("Этап 2/2: Обучение Random Forest...")
        self.model.fit(X_features, y_train)
        
        print(f"✓ {self.name} успешно обучена!")
        print("="*70)
        
    def predict(self, X):
        """Предсказание классов"""
        X_features = self.extract_features(X)
        return self.model.predict(X_features)
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        X_features = self.extract_features(X)
        return self.model.predict_proba(X_features)

class CNN_Model:
    """
    Сверточная нейронная сеть с Transfer Learning (MobileNetV2)
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = "CNN (MobileNetV2)"
        self.model = self.build_model()
        
    def build_model(self):
    
        print(f"\nПостроение {self.name}...")
        
        # Базовая модель MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Размораживаем последние слои для fine-tuning
        base_model.trainable = True
        # Замораживаем все слои кроме последних 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Улучшенная архитектура
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Rescaling(1./255),
            
            # ✅ УЛУЧШЕННАЯ Data Augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),  # Увеличено
            layers.RandomZoom(0.15),      # Увеличено
            layers.RandomContrast(0.2),   # Увеличено
            layers.RandomBrightness(0.2), # Добавлено
            layers.RandomTranslation(0.1, 0.1),  # Добавлено
            
            base_model,
            
            # ✅ Улучшенная классификационная голова
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),  # Увеличен dropout
            
            layers.Dense(256, activation='relu'),  # Увеличено количество нейронов
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ], name='MaskDetectionCNN_Enhanced')
        
        print("✓ Улучшенная архитектура построена")
        
        return model
    
    def compile_model(self):
        """Компиляция модели"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def Train(self, X_Train, y_Train, X_val, y_val, epochs=20, batch_size=32):
        """
        Обучение модели
        
        Args:
            X_Train, y_Train: тренировочные данные
            X_val, y_val: валидационные данные
            epochs: количество эпох
            batch_size: размер батча
            
        Returns:
            history: история обучения
        """
        print(f"\n{'='*70}")
        print(f"ОБУЧЕНИЕ: {self.name}")
        print(f"{'='*70}")
        
        # Компиляция
        self.compile_model()
        
        # Вывод архитектуры
        print("\nАрхитектура модели:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Обучение
        print(f"\nНачало обучения...")
        print(f"Эпох: {epochs}, Batch size: {batch_size}")
        print("="*70)
        
        history = self.model.fit(
            X_Train, y_Train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("="*70)
        print(f"✓ {self.name} успешно обучена!")
        print("="*70)
        
        return history
    
    def predict(self, X):
        """Предсказание классов"""
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Предсказание вероятностей"""
        return self.model.predict(X, verbose=0)


# Тестирование модуля
if __name__ == "__main__":
    print("Тестирование создания моделей...\n")
    
    # Модель 1
    model1 = HOG_SVM_Model()
    print(f"✓ {model1.name} создана")
    
    # Модель 2
    model2 = HaarCascade_RF_Model()
    print(f"✓ {model2.name} создана")
    
    # Модель 3
    model3 = CNN_Model()
    print(f"✓ {model3.name} создана")
    
    print("\n" + "="*70)
    print("Архитектура CNN:")
    print("="*70)
    model3.model.summary()
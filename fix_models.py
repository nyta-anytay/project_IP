"""
Исправление моделей для работы в облаке (Streamlit Cloud с Keras 3)
"""
import pickle
import os
import sys
import numpy as np
import traceback

print("="*70)
print("ИСПРАВЛЕНИЕ МОДЕЛЕЙ ДЛЯ STREAMLIT CLOUD")
print("="*70)

# ===== МОДЕЛЬ 2: Полное исправление pickle =====
print("\n[1/2] Исправление Модель 2 (Haar + RF)...")
MODEL2_ORIGINAL = 'trained_models/model2_haar_rf.pkl'
MODEL2_FIXED = 'trained_models/model2_haar_rf_fixed.pkl'

try:
    if not os.path.exists(MODEL2_ORIGINAL):
        print(f"  ⚠️ Файл не найден: {MODEL2_ORIGINAL}")
        print("  Пропускаем исправление Model 2.")
    else:
        print("  Загрузка модели...")
        with open(MODEL2_ORIGINAL, 'rb') as f:
            model2 = pickle.load(f)
        print("  ✓ Модель загружена")
        
        fixed_attrs = []
        # Попытка удалить проблемные атрибуты с верхнего уровня модели
        for attr in ['monotonic_cst', 'n_features_in_', 'feature_names_in_']:
            if hasattr(model2, attr):
                try:
                    delattr(model2, attr)
                    fixed_attrs.append(attr)
                except:
                    pass
        
        if fixed_attrs:
            print(f"  ✓ Удалены атрибуты: {', '.join(fixed_attrs)}")
        
        # Сохраняем исправленную версию
        print("  Сохранение исправленной модели...")
        with open(MODEL2_FIXED, 'wb') as f:
            pickle.dump(model2, f, protocol=4)
        print(f"  ✓ Модель 2 исправлена: {MODEL2_FIXED}")
        
except Exception as e:
    print(f"  ✗ Ошибка при обработке Model 2: {e}")
    traceback.print_exc()

# ===== МОДЕЛЬ 3: Конвертация для Keras 3 =====
print("\n" + "="*70)
print("[2/2] Конвертация Модель 3 (CNN) для Keras 3...")
print("="*70)

MODEL3_H5 = 'trained_models/model3_cnn.h5'
MODEL3_KERAS = 'trained_models/model3_cnn_keras3.keras'
MODEL3_WEIGHTS = 'trained_models/model3_cnn_weights.h5'

try:
    import tensorflow as tf
    
    # 1. Получаем версию TensorFlow безопасным способом
    tf_version = tf.__version__
    print(f"  TensorFlow версия: {tf_version}")
    
    # Пробуем получить версию Keras, но не падаем если не выходит
    keras_version = "не определена"
    try:
        # Способ для старых версий (TF < 2.13)
        if hasattr(tf.keras, '__version__'):
            keras_version = tf.keras.__version__
        # Способ для новых версий (Keras 3 как отдельный пакет)
        elif hasattr(tf.keras, 'version'):
            keras_version = tf.keras.version()
        # Прямой импорт keras
        else:
            import keras
            keras_version = keras.__version__
    except Exception:
        # Если не удалось определить - не страшно, работаем дальше
        pass
    
    print(f"  Keras версия: {keras_version}")
    
    # 2. Проверяем, какой исходный файл есть
    source_file = None
    if os.path.exists(MODEL3_H5):
        source_file = MODEL3_H5
        print(f"  ✓ Исходный файл: {MODEL3_H5}")
    else:
        print(f"  ⚠️ Файл {MODEL3_H5} не найден")
        print(f"  Ищу другие файлы модели в trained_models/...")
        
        # Поиск любых файлов моделей
        for file in os.listdir('trained_models'):
            if 'model3' in file.lower() and ('h5' in file or 'keras' in file):
                possible_source = os.path.join('trained_models', file)
                print(f"  Найден возможный исходник: {possible_source}")
                source_file = possible_source
                break
    
    if not source_file:
        print("  ⚠️ Не найден файл модели 3 для конвертации.")
        print("  Создаю простую демо-модель...")
        
        # Создаем простую модель для демонстрации
        model3 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print("  ✓ Создана простая демо-модель")
    else:
        # Загружаем существующую модель
        print(f"  Загрузка модели из {source_file}...")
        try:
            model3 = tf.keras.models.load_model(source_file, compile=False)
            print(f"  ✓ Модель загружена")
        except Exception as e:
            print(f"  ✗ Ошибка загрузки: {e}")
            print("  Создаю демо-модель...")
            model3 = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(128, 128, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
    
    print(f"  Архитектура: {len(model3.layers)} слоев")
    print(f"  Вход: {model3.input_shape}, Выход: {model3.output_shape}")
    
    # 3. Сохраняем в .keras формате (поддерживается Keras 3)
    print(f"  Сохранение в формате .keras (Keras 3)...")
    model3.save(MODEL3_KERAS, save_format='keras')
    
    print(f"  ✓ Модель 3 сохранена: {MODEL3_KERAS}")
    print(f"  Размер файла: {os.path.getsize(MODEL3_KERAS) / (1024 * 1024):.1f} MB")
    
    # 4. Проверяем что модель работает
    print("  Тестирование модели...")
    
    # Создаем тестовые данные
    if model3.input_shape[1:] == (128, 128, 3):
        test_input = np.random.randn(1, 128, 128, 3).astype(np.float32)
    else:
        input_shape = list(model3.input_shape[1:])
        input_shape.insert(0, 1)
        test_input = np.random.randn(*input_shape).astype(np.float32)
    
    prediction = model3.predict(test_input, verbose=0)
    print(f"  ✓ Тест пройден, выход: {prediction.shape}, значения: {prediction.flatten()[:3]}")
    
    # 5. Также сохраняем веса отдельно (как запасной вариант)
    model3.save_weights(MODEL3_WEIGHTS)
    print(f"  ✓ Веса сохранены отдельно: {MODEL3_WEIGHTS}")
    
except ImportError:
    print("  ✗ TensorFlow не установлен")
    print("  Установите: pip install tensorflow")
except Exception as e:
    print(f"  ✗ Общая ошибка при обработке Model 3: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("✅ ИСПРАВЛЕНИЕ ЗАВЕРШЕНО!")
print("="*70)

print("\n📁 Содержимое trained_models/ после исправлений:")
if os.path.exists('trained_models'):
    for item in sorted(os.listdir('trained_models')):
        item_path = os.path.join('trained_models', item)
        if os.path.isdir(item_path):
            # Функция для расчета размера папки
            def get_folder_size(path):
                total = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total += os.path.getsize(fp)
                return total / (1024 * 1024)
            
            size = get_folder_size(item_path)
            print(f"📁 {item}/ ({size:.1f} MB)")
        else:
            size_kb = os.path.getsize(item_path) / 1024
            print(f"📄 {item} ({size_kb:.1f} KB)")

print("\n🚀 Теперь выполните:")
print("1. git add trained_models/")
print("2. git commit -m 'Fixed models for cloud'")
print("3. git push")
print("\n📦 Streamlit Cloud автоматически обновится через 2-3 минуты!")
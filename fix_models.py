"""
Исправление моделей для работы в облаке
"""
import pickle
import os
import sys
import numpy as np  # ← ДОБАВИЛИ!

def get_folder_size(folder_path):
    """Размер папки в MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

print("="*70)
print("ИСПРАВЛЕНИЕ МОДЕЛЕЙ ДЛЯ STREAMLIT CLOUD")
print("="*70)

# Добавляем src в path
sys.path.append('src')

# ===== МОДЕЛЬ 2: Удаление проблемного атрибута =====
print("\n[1/2] Исправление Модель 2 (Haar + RF)...")

try:
    # Загружаем
    with open('trained_models/model2_haar_rf.pkl', 'rb') as f:
        model2 = pickle.load(f)
    
    print("  ✓ Модель загружена")
    
    # Удаляем проблемный атрибут из каждого дерева в Random Forest
    if hasattr(model2, 'model') and hasattr(model2.model, 'estimators_'):
        rf = model2.model
        
        fixed_count = 0
        for tree in rf.estimators_:
            # Удаляем все потенциально проблемные атрибуты sklearn 1.4+
            attrs_to_remove = [
                'monotonic_cst', 
                'n_features_in_',
                'feature_names_in_'
            ]
            
            for attr in attrs_to_remove:
                if hasattr(tree, attr):
                    try:
                        delattr(tree, attr)
                        fixed_count += 1
                    except:
                        pass
        
        print(f"  ✓ Удалено проблемных атрибутов: {fixed_count}")
    
    # Сохраняем обратно
    with open('trained_models/model2_haar_rf.pkl', 'wb') as f:
        pickle.dump(model2, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("  ✓ Модель 2 исправлена и сохранена!")
    
except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    print("  Модель 2 не будет работать в облаке")

# ===== МОДЕЛЬ 3: Конвертация в SavedModel формат =====
print("\n[2/2] Конвертация Модель 3 (CNN)...")

try:
    import tensorflow as tf
    
    # Подавляем warnings
    tf.get_logger().setLevel('ERROR')
    
    # Загружаем .h5
    print("  Загрузка .h5 файла...")
    model3 = tf.keras.models.load_model(
        'trained_models/model3_cnn.h5',
        compile=False
    )
    
    print("  ✓ Модель загружена")
    print(f"  Архитектура: {len(model3.layers)} слоев")
    print(f"  Вход: {model3.input_shape}, Выход: {model3.output_shape}")
    
    # Сохраняем в SavedModel формате (лучше совместимость)
    save_path = 'trained_models/model3_cnn_savedmodel'
    
    print(f"  Сохранение в формате SavedModel...")
    model3.save(save_path, save_format='tf')
    
    print(f"  ✓ Модель 3 сохранена: {save_path}/")
    print(f"  Размер папки: {get_folder_size(save_path):.1f} MB")
    
    # Также создаем .keras формат (альтернатива)
    keras_path = 'trained_models/model3_cnn.keras'
    model3.save(keras_path, save_format='keras')
    print(f"  ✓ Также сохранена в .keras формате")
    
    # Проверяем что модель работает
    print("  Тестирование модели...")
    
    # Создаем тестовые данные в правильном формате
    if model3.input_shape[1:] == (128, 128, 3):  # RGB
        test_input = np.random.randn(1, 128, 128, 3).astype(np.float32)
    elif model3.input_shape[1:] == (128, 128, 1):  # Grayscale
        test_input = np.random.randn(1, 128, 128, 1).astype(np.float32)
    else:
        # Общий случай
        input_shape = list(model3.input_shape[1:])
        input_shape.insert(0, 1)  # Добавляем batch dimension
        test_input = np.random.randn(*input_shape).astype(np.float32)
    
    prediction = model3.predict(test_input, verbose=0)
    print(f"  ✓ Тест пройден, выход: {prediction.shape}")
    
except ImportError:
    print("  ✗ TensorFlow не установлен, пропускаем модель 3")
except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    print("  Проблема с моделью 3")

print("\n" + "="*70)
print("✅ ГОТОВО!")
print("="*70)

print("\nСтруктура trained_models/ после исправлений:")
if os.path.exists('trained_models'):
    for item in os.listdir('trained_models'):
        item_path = os.path.join('trained_models', item)
        if os.path.isdir(item_path):
            size = get_folder_size(item_path)
            print(f"📁 {item}/ ({size:.1f} MB)")
        else:
            size_kb = os.path.getsize(item_path) / 1024
            print(f"📄 {item} ({size_kb:.1f} KB)")

print("\nТеперь выполните:")
print("1. git add trained_models/")
print("2. git commit -m 'fix models for cloud'")
print("3. git push")
print("\nStreamlit Cloud автоматически обновится через 2-3 минуты!")
"""
Скрипт для конвертации модели в новый формат Keras 3
Запускать в окружении, где модель изначально обучалась
"""
import tensorflow as tf
import os

# Путь к оригинальной модели
ORIGINAL_MODEL = 'trained_models/model3_cnn.h5'

# Пути для новых файлов (оригинал НЕ удаляется)
NEW_MODEL_KERAS = 'trained_models/model3_cnn_new.keras'
NEW_MODEL_H5 = 'trained_models/model3_cnn_fixed.h5'
NEW_WEIGHTS = 'trained_models/model3_cnn.weights.h5'

print(f"TensorFlow версия: {tf.__version__}")

# Пробуем получить версию Keras разными способами
try:
    import keras
    print(f"Keras версия: {keras.__version__}")
except:
    print("Keras версия: встроенная в TensorFlow")

# Проверяем существование файла
if not os.path.exists(ORIGINAL_MODEL):
    print(f"\n❌ Файл не найден: {ORIGINAL_MODEL}")
    print("   Проверьте путь к модели!")
    exit(1)

# Загружаем оригинальную модель
print(f"\n📂 Загружаю модель из: {ORIGINAL_MODEL}")
try:
    model = tf.keras.models.load_model(ORIGINAL_MODEL, compile=False)
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit(1)

print("\n📊 Информация о модели:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Количество слоёв: {len(model.layers)}")

# Показываем краткую информацию
print("\n📋 Слои модели:")
for i, layer in enumerate(model.layers[:10]):  # Первые 10 слоёв
    print(f"   {i}: {layer.name} ({layer.__class__.__name__})")
if len(model.layers) > 10:
    print(f"   ... и ещё {len(model.layers) - 10} слоёв")

# Сохраняем в формате .keras (если поддерживается)
print(f"\n💾 Сохраняю в формате .keras: {NEW_MODEL_KERAS}")
try:
    model.save(NEW_MODEL_KERAS)
    print("   ✅ Готово!")
except Exception as e:
    print(f"   ⚠️ Не удалось сохранить в .keras: {e}")
    print("   Пробуем альтернативный формат...")

# Сохраняем в формате .h5 (более совместимый)
print(f"\n💾 Сохраняю в формате .h5: {NEW_MODEL_H5}")
try:
    model.save(NEW_MODEL_H5, save_format='h5')
    print("   ✅ Готово!")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

# Сохраняем только веса
print(f"\n💾 Сохраняю веса: {NEW_WEIGHTS}")
try:
    model.save_weights(NEW_WEIGHTS)
    print("   ✅ Готово!")
except Exception as e:
    print(f"   ⚠️ Ошибка: {e}")

# Проверяем размеры файлов
print("\n📁 Размеры файлов:")
for path in [ORIGINAL_MODEL, NEW_MODEL_KERAS, NEW_MODEL_H5, NEW_WEIGHTS]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {path}: {size_mb:.2f} MB")
    else:
        print(f"   ❌ {path}: не создан")

# Тестируем загрузку новых файлов
print("\n🧪 Тестирую загрузку новых файлов...")

for path in [NEW_MODEL_KERAS, NEW_MODEL_H5]:
    if os.path.exists(path):
        try:
            test_model = tf.keras.models.load_model(path, compile=False)
            print(f"   ✅ {path}: загружается успешно!")
        except Exception as e:
            print(f"   ❌ {path}: ошибка загрузки - {e}")

print("\n" + "="*50)
print("✅ Конвертация завершена!")
print("   Оригинальный файл сохранён")
print("\n📌 Следующие шаги:")
print("   1. Закоммитьте новые файлы в git")
print("   2. Обновите app.py для использования новых файлов")
print("="*50)
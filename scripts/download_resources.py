"""
Загрузка дополнительных ресурсов (Haar Cascade, веса MobileNetV2)
"""
import os
import sys
import urllib.request

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import HAAR_CASCADE_PATH, ASSETS_DIR


def download_haar_cascade():
    """Загрузка Haar Cascade классификатора"""
    print("\n" + "="*70)
    print("ЗАГРУЗКА HAAR CASCADE")
    print("="*70)
    
    if os.path.exists(HAAR_CASCADE_PATH):
        print(f"✓ Haar Cascade уже существует: {HAAR_CASCADE_PATH}")
        return True
    
    url = ("https://raw.githubusercontent.com/opencv/opencv/master/"
           "data/haarcascades/haarcascade_frontalface_default.xml")
    
    print(f"Загрузка из: {url}")
    print(f"Сохранение в: {HAAR_CASCADE_PATH}")
    
    try:
        # Создаем директорию если не существует
        os.makedirs(ASSETS_DIR, exist_ok=True)
        
        # Загрузка с прогрессом
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\r  Прогресс: {percent:.1f}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, HAAR_CASCADE_PATH, reporthook)
        print("\n✓ Haar Cascade успешно загружен!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка загрузки: {e}")
        return False


def download_mobilenet_weights():
    """Предзагрузка весов MobileNetV2"""
    print("\n" + "="*70)
    print("ЗАГРУЗКА ВЕСОВ MOBILENETV2")
    print("="*70)
    
    print("Проверка наличия весов MobileNetV2...")
    print("При первом запуске веса будут скачаны автоматически (~14 MB)")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        
        print("\nЗагрузка MobileNetV2 с весами ImageNet...")
        print("Это может занять несколько минут...")
        
        model = MobileNetV2(
            input_shape=(128, 128, 3),
            include_top=False,
            weights='imagenet'
        )
        
        print(f"✓ Веса MobileNetV2 загружены!")
        print(f"  Количество слоев: {len(model.layers)}")
        print(f"  Параметров: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nВозможные причины:")
        print("  1. Нет подключения к интернету")
        print("  2. Проблемы с TensorFlow")
        print("\nРешение:")
        print("  pip install --upgrade tensorflow")
        return False


def main():
    print("\n" + "="*70)
    print("ЗАГРУЗКА НЕОБХОДИМЫХ РЕСУРСОВ")
    print("="*70)
    
    success = True
    
    # 1. Haar Cascade
    success = download_haar_cascade() and success
    
    # 2. MobileNetV2 weights
    success = download_mobilenet_weights() and success
    
    # Итог
    print("\n" + "="*70)
    if success:
        print("✅ ВСЕ РЕСУРСЫ ЗАГРУЖЕНЫ!")
        print("\nМожно приступать к обучению моделей:")
        print("  python scripts/01_analyze_data.py")
    else:
        print("⚠️  НЕКОТОРЫЕ РЕСУРСЫ НЕ ЗАГРУЖЕНЫ")
        print("Проверьте подключение к интернету и попробуйте снова")
    print("="*70 + "\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
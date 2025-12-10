"""
Проверка всех зависимостей проекта
"""
import sys
import os


def check_package(package_name, import_name=None):
    """Проверка наличия пакета"""
    if import_name is None:
        import_name = package_name
    
    try:
        import importlib
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name:25s} {version}")
        return True
    except ImportError:
        print(f"  ❌ {package_name:25s} НЕ УСТАНОВЛЕН")
        return False
    except Exception as e:
        print(f"  ⚠️  {package_name:25s} ОШИБКА: {e}")
        return False


def check_python_version():
    """Проверка версии Python"""
    print(f"\n🐍 Python версия:")
    version_info = sys.version_info
    print(f"  Версия: {sys.version}")
    print(f"  ✓ Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info < (3, 8):
        print("  ❌ Требуется Python 3.8 или выше!")
        return False
    
    return True


def check_opencv_data():
    """Проверка данных OpenCV"""
    try:
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            print(f"  ✓ Haar Cascade найден")
            return True
        else:
            print(f"  ❌ Haar Cascade НЕ НАЙДЕН")
            print(f"     Запустите: python scripts/download_resources.py")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка проверки OpenCV: {e}")
        return False


def check_tensorflow_gpu():
    """Проверка GPU для TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"  ✓ TensorFlow GPU: {len(gpus)} устройств(а)")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
            return True
        else:
            print("  ⚠️  TensorFlow GPU не найден (будет использоваться CPU)")
            print("     Обучение CNN будет медленнее")
            return True
    except Exception as e:
        print(f"  ❌ Ошибка проверки TensorFlow GPU: {e}")
        return False


def check_data_folder():
    """Проверка наличия папки с данными и правильной структуры"""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    if not os.path.exists(data_path):
        print(f"  ❌ Папка data НЕ НАЙДЕНА")
        print(f"     Создайте папку: {data_path}")
        return False
    
    print(f"  ✓ Папка data найдена")
    
    # Проверка обязательных подпапок
    required_folders = ['train', 'test', 'validation']
    all_found = True
    
    for folder in required_folders:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            print(f"  ❌ Папка {folder} НЕ НАЙДЕНА")
            all_found = False
        else:
            # Проверяем подпапки с классами
            class_folders = [f for f in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, f))]
            
            if len(class_folders) == 0:
                print(f"  ❌ В папке {folder} нет подпапок с классами")
                all_found = False
            else:
                total_images = 0
                for class_folder in class_folders:
                    class_path = os.path.join(folder_path, class_folder)
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    total_images += len(images)
                
                print(f"    ✓ {folder:12s}: {len(class_folders)} класса(ов), {total_images} изображений")
    
    if not all_found:
        print(f"\n  ⚠️  НЕПРАВИЛЬНАЯ СТРУКТУРА ДАННЫХ!")
        print(f"     Ожидаемая структура:")
        print(f"     data/")
        print(f"     ├── train/")
        print(f"     │   ├── with_mask/")
        print(f"     │   └── without_mask/")
        print(f"     ├── validation/")
        print(f"     │   ├── with_mask/")
        print(f"     │   └── without_mask/")
        print(f"     └── test/")
        print(f"         ├── with_mask/")
        print(f"         └── without_mask/")
        return False
    
    return True


def main():
    """Основная функция"""
    print("\n" + "="*70)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ И ОКРУЖЕНИЯ")
    print("="*70)
    
    all_ok = True
    
    # 1. Python версия
    python_ok = check_python_version()
    all_ok = all_ok and python_ok
    
    # 2. Основные библиотеки
    print(f"\n📦 Основные библиотеки:")
    packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        ('Pillow', 'PIL'),
        ('scikit-learn', 'sklearn'),
        ('opencv-python', 'cv2'),
        ('scikit-image', 'skimage'),
        'tensorflow',
        'streamlit',
        'tqdm',
        'joblib',
    ]
    
    for pkg in packages:
        if isinstance(pkg, tuple):
            ok = check_package(pkg[0], pkg[1])
        else:
            ok = check_package(pkg)
        all_ok = all_ok and ok
    
    # 3. Дополнительные проверки
    print(f"\n🔍 Дополнительные проверки:")
    opencv_ok = check_opencv_data()
    all_ok = all_ok and opencv_ok
    
    check_tensorflow_gpu()  # Не влияет на all_ok
    
    data_ok = check_data_folder()
    all_ok = all_ok and data_ok
    
    # 4. Итог
    print("\n" + "="*70)
    if all_ok:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("\nМожно приступать к работе:")
        print("  1. python scripts/check_data_structure.py")
        print("  2. python scripts/download_resources.py")
        print("  3. python scripts/01_analyze_data.py")
    else:
        print("⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        print("\nЧто делать:")
        print("  1. Установите недостающие пакеты:")
        print("     pip install -r requirements.txt")
        print("  2. Проверьте структуру папки data")
    print("="*70 + "\n")
    
    return all_ok


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
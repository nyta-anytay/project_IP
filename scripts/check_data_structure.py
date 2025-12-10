"""
Проверка структуры папки data
"""
import os
import sys

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR


def check_structure():
    """Проверка и вывод структуры папки data"""
    print("\n" + "="*70)
    print("ПРОВЕРКА СТРУКТУРЫ ДАННЫХ")
    print("="*70)
    
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Папка data не найдена: {DATA_DIR}")
        print("\nСоздайте папку и поместите туда данные в следующей структуре:")
        print_expected_structure()
        return False
    
    print(f"\n✓ Папка data найдена: {DATA_DIR}\n")
    
    # Ожидаемая структура
    expected = {
        'Train': ['WithMask', 'WithoutMask'],
        'Validation': ['WithMask', 'WithoutMask'],
        'Test': ['WithMask', 'WithoutMask']
    }
    
    all_ok = True
    total_images = 0
    
    print("Анализ структуры:\n")
    
    for split_name, class_names in expected.items():
        split_path = os.path.join(DATA_DIR, split_name)
        
        if not os.path.exists(split_path):
            print(f"❌ {split_name:12s} - ОТСУТСТВУЕТ")
            all_ok = False
            continue
        
        print(f"✓ {split_name:12s}")
        split_total = 0
        
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"  ❌ {class_name:15s} - ОТСУТСТВУЕТ")
                all_ok = False
                continue
            
            # Подсчет изображений
            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            count = len(images)
            split_total += count
            total_images += count
            
            status = "✓" if count > 0 else "⚠️"
            print(f"  {status} {class_name:15s}: {count:5d} изображений")
        
        print(f"  {'─'*30}")
        print(f"  Итого в {split_name:9s}: {split_total:5d} изображений\n")
    
    print("="*70)
    print(f"ВСЕГО ИЗОБРАЖЕНИЙ: {total_images}")
    print("="*70)
    
    if not all_ok:
        print("\n⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ СО СТРУКТУРОЙ!")
        print("\nОжидаемая структура:")
        print_expected_structure()
        return False
    
    if total_images == 0:
        print("\n⚠️  НЕТ ИЗОБРАЖЕНИЙ!")
        print("Поместите изображения в соответствующие папки")
        return False
    
    print("\n✅ СТРУКТУРА ДАННЫХ КОРРЕКТНА!")
    print("\nМожно приступать к обучению:")
    print("  python scripts/01_analyze_data.py")
    
    return True


def print_expected_structure():
    """Вывод ожидаемой структуры"""
    print("""
    data/
    ├── Train/
    │   ├── WithMask/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── WithoutMask/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    ├── Validation/
    │   ├── WithMask/
    │   │   └── ...
    │   └── WithoutMask/
    │       └── ...
    └── Test/
        ├── WithMask/
        │   └── ...
        └── WithoutMask/
            └── ...
    """)


def main():
    success = check_structure()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
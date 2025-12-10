"""
Анализ и визуализация датасета
"""
import os
import sys

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, RESULTS_DIR
from src.data_preparation import DataPreparation
from src.utils import (plot_class_distribution, plot_sample_images, 
                       check_data_quality)
from collections import Counter


def analyze_split_distribution(y_Train, y_val, y_Test, labels_map):
    """Анализ распределения по выборкам"""
    print("\n" + "="*70)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПО ВЫБОРКАМ")
    print("="*70)
    
    total = len(y_Train) + len(y_val) + len(y_Test)
    
    print(f"\nОбщее количество образцов: {total}")
    print(f"\nРаспределение по выборкам:")
    print(f"  Train:      {len(y_Train):5d} ({len(y_Train)/total*100:5.1f}%)")
    print(f"  Validation: {len(y_val):5d} ({len(y_val)/total*100:5.1f}%)")
    print(f"  Test:       {len(y_Test):5d} ({len(y_Test)/total*100:5.1f}%)")
    
    # Распределение классов в каждой выборке
    print(f"\nРаспределение классов:")
    
    for subset_name, y_subset in [('Train', y_Train), 
                                   ('Validation', y_val), 
                                   ('Test', y_Test)]:
        counter = Counter(y_subset)
        print(f"\n  {subset_name}:")
        for class_idx in sorted(counter.keys()):
            count = counter[class_idx]
            percentage = count / len(y_subset) * 100
            print(f"    {labels_map[class_idx]:15s}: {count:4d} ({percentage:5.1f}%)")
    
    print("="*70)


def main():
    print("\n" + "="*70)
    print("АНАЛИЗ ДАТАСЕТА")
    print("="*70)
    
    try:
        # Загрузка данных с готовым разделением
        prep = DataPreparation(DATA_DIR)
        X_Train, X_val, X_Test, y_Train, y_val, y_Test, labels_map = prep.load_split_data()
        
        # Объединяем для общего анализа
        import numpy as np
        X_all = np.concatenate([X_Train, X_val, X_Test], axis=0)
        y_all = np.concatenate([y_Train, y_val, y_Test], axis=0)
        
        # Анализ распределения по выборкам
        analyze_split_distribution(y_Train, y_val, y_Test, labels_map)
        
        # Проверка качества данных
        check_data_quality(X_all, y_all, labels_map)
        
        # Визуализации
        print("\nСоздание визуализаций...")
        
        # 1. Распределение классов (общее)
        plot_class_distribution(
            y_all, labels_map,
            save_path=os.path.join(RESULTS_DIR, 'class_distribution.png')
        )
        
        # 2. Примеры изображений из Train
        plot_sample_images(
            X_Train, y_Train, labels_map,
            n_samples=5,
            save_path=os.path.join(RESULTS_DIR, 'sample_images.png')
        )
        
        print("\n" + "="*70)
        print("✅ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"   Результаты сохранены в: {RESULTS_DIR}")
        print("\nСледующий шаг:")
        print("  python scripts/02_Train_models.py")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nУбедитесь, что:")
        print("  1. Папка 'data' существует")
        print("  2. В ней есть папки: Train, Test, Validation")
        print("  3. В каждой есть подпапки с классами (WithMask, WithoutMask)")
        print("  4. В подпапках есть изображения (.jpg, .jpeg, .png)")
        print("\nПример структуры:")
        print("  data/")
        print("  ├── Train/")
        print("  │   ├── WithMask/")
        print("  │   └── WithoutMask/")
        print("  ├── Validation/")
        print("  │   ├── WithMask/")
        print("  │   └── WithoutMask/")
        print("  └── Test/")
        print("      ├── WithMask/")
        print("      └── WithoutMask/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
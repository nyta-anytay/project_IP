"""
Обучение всех трех моделей
"""
import os
import sys
import pickle
import json
import numpy as np

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (DATA_DIR, MODEL1_PATH, MODEL2_PATH, MODEL3_PATH,
                        LABELS_MAP_PATH, Test_DATA_PATH, RESULTS_DIR,
                        CNN_EPOCHS, CNN_BATCH_SIZE)
from src.data_preparation import DataPreparation
from src.models import HOG_SVM_Model, HaarCascade_RF_Model, CNN_Model
from src.utils import plot_training_history


def main():
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    try:
        # ===== 1. ЗАГРУЗКА ДАННЫХ =====
        print("\n[ЭТАП 1/4] Загрузка данных с готовым разделением...")
        
        prep = DataPreparation(DATA_DIR)
        X_Train, X_val, X_Test, y_Train, y_val, y_Test, labels_map = prep.load_split_data()
        
        # Сохранение тестовых данных
        np.savez(
            Test_DATA_PATH,
            X_Test=X_Test,
            y_Test=y_Test
        )
        print(f"✓ Тестовые данные сохранены: {Test_DATA_PATH}")
        
        # Сохранение labels_map
        with open(LABELS_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in labels_map.items()}, f, indent=2, ensure_ascii=False)
        print(f"✓ Labels map сохранен: {LABELS_MAP_PATH}")
        
        # ===== 2. ОБУЧЕНИЕ МОДЕЛИ 1: HOG + SVM =====
        print("\n[ЭТАП 2/4] Обучение Модели 1: HOG + SVM...")
        
        model1 = HOG_SVM_Model()
        model1.Train(X_Train, y_Train)
        
        # Сохранение
        with open(MODEL1_PATH, 'wb') as f:
            pickle.dump(model1, f)
        print(f"✓ Модель сохранена: {MODEL1_PATH}")
        
        # Быстрая оценка на Validation
        print("\nБыстрая оценка на Validation set...")
        y_pred_val = model1.predict(X_val)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_val, y_pred_val)
        print(f"  Validation Accuracy: {acc:.4f}")
        
        # ===== 3. ОБУЧЕНИЕ МОДЕЛИ 2: HAAR + RF =====
        print("\n[ЭТАП 3/4] Обучение Модели 2: Haar Cascade + RF...")
        
        model2 = HaarCascade_RF_Model()
        model2.train(X_Train, y_Train)
        
        # Сохранение
        with open(MODEL2_PATH, 'wb') as f:
            pickle.dump(model2, f)
        print(f"✓ Модель сохранена: {MODEL2_PATH}")
        
        # Быстрая оценка на Validation
        print("\nБыстрая оценка на Validation set...")
        y_pred_val = model2.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        print(f"  Validation Accuracy: {acc:.4f}")
        
        # ===== 4. ОБУЧЕНИЕ МОДЕЛИ 3: CNN =====
        print("\n[ЭТАП 4/4] Обучение Модели 3: CNN (MobileNetV2)...")
        
        model3 = CNN_Model(num_classes=len(labels_map))
        history = model3.Train(
            X_Train, y_Train,
            X_val, y_val,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE
        )
        
        # Сохранение
        model3.model.save(MODEL3_PATH)
        print(f"✓ Модель сохранена: {MODEL3_PATH}")
        
        
        
        # Лучшие метрики
        best_val_acc = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        print(f"\nЛучшая Validation accuracy: {best_val_acc:.4f} (эпоха {best_epoch})")
        
        # ===== ИТОГИ =====
        print("\n" + "="*70)
        print("✅ ВСЕ МОДЕЛИ УСПЕШНО ОБУЧЕНЫ И СОХРАНЕНЫ!")
        print("="*70)
        print("\nСохраненные файлы:")
        print(f"  1. {MODEL1_PATH}")
        print(f"  2. {MODEL2_PATH}")
        print(f"  3. {MODEL3_PATH}")
        print(f"  4. {Test_DATA_PATH}")
        print(f"  5. {LABELS_MAP_PATH}")
        print(f"\nРезультаты Validation:")
        print(f"  HOG + SVM:              {acc:.4f}")  # последняя сохраненная
        print(f"  CNN (best):             {best_val_acc:.4f}")
        print("\nСледующий шаг:")
        print("  python scripts/03_evaluate_models.py")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
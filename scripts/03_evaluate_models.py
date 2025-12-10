"""
Оценка и сравнение обученных моделей
"""
import os
import sys
import pickle
import json
import numpy as np
import tensorflow as tf

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (MODEL1_PATH, MODEL2_PATH, MODEL3_PATH,
                        LABELS_MAP_PATH, Test_DATA_PATH)
from src.evaluation import ModelEvaluator


class CNNWrapper:
    """Обертка для CNN модели"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)


def main():
    print("\n" + "="*70)
    print("ОЦЕНКА МОДЕЛЕЙ")
    print("="*70)
    
    try:
        # ===== 1. ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ =====
        print("\n[ЭТАП 1/3] Загрузка тестовых данных...")
        
        if not os.path.exists(Test_DATA_PATH):
            raise FileNotFoundError(
                f"Тестовые данные не найдены: {Test_DATA_PATH}\n"
                "Сначала запустите: python scripts/02_Train_models.py"
            )
        
        data = np.load(Test_DATA_PATH)
        X_Test = data['X_Test']
        y_Test = data['y_Test']
        
        print(f"✓ Загружено {len(X_Test)} тестовых образцов")
        
        # Загрузка labels_map
        with open(LABELS_MAP_PATH, 'r') as f:
            labels_dict = json.load(f)
            labels_map = {int(k): v for k, v in labels_dict.items()}
        
        print(f"✓ Классы: {labels_map}")
        
        # ===== 2. ЗАГРУЗКА МОДЕЛЕЙ =====
        print("\n[ЭТАП 2/3] Загрузка обученных моделей...")
        
        # Модель 1: HOG + SVM
        print("  Загрузка HOG + SVM...")
        with open(MODEL1_PATH, 'rb') as f:
            model1 = pickle.load(f)
        print("  ✓ HOG + SVM загружена")
        
        # Модель 2: Haar + RF
        print("  Загрузка Haar Cascade + RF...")
        with open(MODEL2_PATH, 'rb') as f:
            model2 = pickle.load(f)
        print("  ✓ Haar Cascade + RF загружена")
        
        # Модель 3: CNN
        print("  Загрузка CNN...")
        model3_keras = tf.keras.models.load_model(MODEL3_PATH)
        model3 = CNNWrapper(model3_keras)
        print("  ✓ CNN загружена")
        
        # ===== 3. ОЦЕНКА МОДЕЛЕЙ =====
        print("\n[ЭТАП 3/3] Оценка моделей на тестовой выборке...")
        
        evaluator = ModelEvaluator(
            models=[model1, model2, model3],
            model_names=['HOG + SVM', 'Haar Cascade + RF', 'CNN (MobileNetV2)'],
            X_Test=X_Test,
            y_Test=y_Test,
            labels_map=labels_map
        )
        
        results_df = evaluator.evaluate_all()
        
        # ===== ИТОГИ =====
        print("\n" + "="*70)
        print("✅ ОЦЕНКА ЗАВЕРШЕНА!")
        print("="*70)
        print("\nРезультаты сохранены в папке 'results/':")
        print("  - model_comparison.csv - сравнение моделей")
        print("  - confusion_matrix_*.png - матрицы ошибок")
        print("  - models_comparison.png - график сравнения")
        print("  - roc_curve_*.png - ROC кривые (если применимо)")
        print("\nСледующий шаг:")
        print("  cd web_app")
        print("  streamlit run app.py")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
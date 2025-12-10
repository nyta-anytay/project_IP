"""
Конфигурация проекта
"""
import os

# ===== ПУТИ =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
MODELS_DIR = os.path.join(BASE_DIR, 'Trained_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Создание директорий если не существуют
for directory in [ASSETS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== ПАРАМЕТРЫ ДАННЫХ =====
IMG_SIZE = (128, 128)
RANDOM_STATE = 42
Test_SIZE = 0.2
VAL_SIZE = 0.1

# ===== ПУТИ К МОДЕЛЯМ =====
MODEL1_PATH = os.path.join(MODELS_DIR, 'model1_hog_svm.pkl')
MODEL2_PATH = os.path.join(MODELS_DIR, 'model2_haar_rf.pkl')
MODEL3_PATH = os.path.join(MODELS_DIR, 'model3_cnn.h5')

# ===== ПУТИ К РЕСУРСАМ =====
HAAR_CASCADE_PATH = os.path.join(ASSETS_DIR, 'haarcascade_frontalface_default.xml')
LABELS_MAP_PATH = os.path.join(RESULTS_DIR, 'labels_map.json')
Test_DATA_PATH = os.path.join(RESULTS_DIR, 'Test_data.npz')

# ===== ПАРАМЕТРЫ ОБУЧЕНИЯ =====
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001

# Вывод конфигурации
if __name__ == "__main__":
    print("="*70)
    print("КОНФИГУРАЦИЯ ПРОЕКТА")
    print("="*70)
    print(f"Базовая директория: {BASE_DIR}")
    print(f"Папка данных:       {DATA_DIR}")
    print(f"Папка ресурсов:     {ASSETS_DIR}")
    print(f"Папка моделей:      {MODELS_DIR}")
    print(f"Папка результатов:  {RESULTS_DIR}")
    print(f"\nРазмер изображения: {IMG_SIZE}")
    print(f"Random state:       {RANDOM_STATE}")
    print(f"Test size:          {Test_SIZE}")
    print(f"Val size:           {VAL_SIZE}")
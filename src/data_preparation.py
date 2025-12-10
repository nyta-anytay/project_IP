"""
Подготовка данных для обучения
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from .config import IMG_SIZE


class DataPreparation:
    """Класс для загрузки данных с готовым разделением на Train/val/Test"""
    
    def __init__(self, data_path, img_size=IMG_SIZE):
        self.data_path = data_path
        self.img_size = img_size
        
        # Проверка существования папки
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Папка с данными не найдена: {data_path}")
        
        # Проверка структуры
        required_folders = ['Train', 'Test', 'Validation']
        for folder in required_folders:
            folder_path = os.path.join(data_path, folder)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(
                    f"Ожидаемая папка не найдена: {folder_path}\n"
                    f"Структура должна быть: data/Train/, data/Test/, data/Validation/"
                )
    
    def load_split_data(self):
        """
        Загрузка данных с готовым разделением
        
        Returns:
            X_Train, X_val, X_Test: массивы изображений
            y_Train, y_val, y_Test: массивы меток
            labels_map: словарь {индекс: название_класса}
        """
        print("\n" + "="*70)
        print("ЗАГРУЗКА ДАННЫХ С ГОТОВЫМ РАЗДЕЛЕНИЕМ")
        print("="*70)
        
        # Определяем классы из первой доступной папки
        labels_map = self._get_labels_map()
        print(f"Найденные классы: {labels_map}\n")
        
        # Загружаем каждую выборку
        X_Train, y_Train = self._load_subset('Train', labels_map)
        X_val, y_val = self._load_subset('Validation', labels_map)
        X_Test, y_Test = self._load_subset('Test', labels_map)
        
        # Итоговая статистика
        print("\n" + "="*70)
        print("✓ ЗАГРУЗКА ЗАВЕРШЕНА")
        print("="*70)
        print(f"Train set:      {X_Train.shape} - {len(y_Train)} образцов")
        print(f"Validation set: {X_val.shape} - {len(y_val)} образцов")
        print(f"Test set:       {X_Test.shape} - {len(y_Test)} образцов")
        print(f"Всего:          {len(y_Train) + len(y_val) + len(y_Test)} образцов")
        print(f"Классы:         {labels_map}")
        print("="*70)
        
        return X_Train, X_val, X_Test, y_Train, y_val, y_Test, labels_map
    
    def _get_labels_map(self):
        """Определение классов из структуры папок"""
        # Берем первую доступную папку (Train, Test или Validation)
        for subset in ['Train', 'Test', 'Validation']:
            subset_path = os.path.join(self.data_path, subset)
            if os.path.exists(subset_path):
                class_folders = sorted([
                    f for f in os.listdir(subset_path)
                    if os.path.isdir(os.path.join(subset_path, f))
                ])
                if class_folders:
                    return {idx: name for idx, name in enumerate(class_folders)}
        
        raise ValueError("Не найдено подпапок с классами!")
    
    def _load_subset(self, subset_name, labels_map):
        """
        Загрузка одной выборки (Train/Validation/Test)
        
        Args:
            subset_name: название выборки ('Train', 'Validation', 'Test')
            labels_map: словарь классов
            
        Returns:
            X: массив изображений
            y: массив меток
        """
        subset_path = os.path.join(self.data_path, subset_name)
        
        print(f"\n{'─'*70}")
        print(f"Загрузка выборки: {subset_name.upper()}")
        print(f"{'─'*70}")
        
        X, y = [], []
        
        for class_idx, class_name in labels_map.items():
            class_path = os.path.join(subset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"⚠️  Папка {class_name} не найдена в {subset_name}, пропускаем...")
                continue
            
            # Получаем список изображений
            images = [
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            print(f"Класс '{class_name}': найдено {len(images)} изображений")
            
            if len(images) == 0:
                print(f"  ⚠️  Нет изображений в {class_path}")
                continue
            
            # Загрузка с прогресс-баром
            loaded = 0
            for img_name in tqdm(images, desc=f"  Загрузка", ncols=70):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Чтение изображения
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    # Конвертация BGR -> RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Изменение размера
                    img = cv2.resize(img, self.img_size)
                    
                    X.append(img)
                    y.append(class_idx)
                    loaded += 1
                    
                except Exception as e:
                    # print(f"  ⚠️  Ошибка при загрузке {img_name}: {e}")
                    continue
            
            print(f"  ✓ Успешно загружено: {loaded}/{len(images)}")
        
        # Конвертация в numpy arrays
        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.int32)
        
        print(f"✓ Выборка {subset_name}: {X.shape}")
        
        return X, y
    
    def load_data(self):
        """
        Загрузка всех данных без разделения (для совместимости)
        Объединяет Train + Validation + Test
        
        Returns:
            X: все изображения
            y: все метки
            labels_map: словарь классов
        """
        print("\n" + "="*70)
        print("ЗАГРУЗКА ВСЕХ ДАННЫХ (БЕЗ РАЗДЕЛЕНИЯ)")
        print("="*70)
        
        labels_map = self._get_labels_map()
        print(f"Найденные классы: {labels_map}\n")
        
        X_all, y_all = [], []
        
        for subset in ['Train', 'Validation', 'Test']:
            X, y = self._load_subset(subset, labels_map)
            X_all.append(X)
            y_all.append(y)
        
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        
        print("\n" + "="*70)
        print(f"✓ Всего загружено: {len(X)} изображений")
        print(f"  Форма данных: {X.shape}")
        print(f"  Классы: {labels_map}")
        print("="*70)
        
        return X, y, labels_map


# Тестирование модуля
if __name__ == "__main__":
    from config import DATA_DIR
    
    prep = DataPreparation(DATA_DIR)
    
    # Тест 1: Загрузка с разделением
    print("\n" + "="*70)
    print("ТЕСТ: Загрузка с готовым разделением")
    print("="*70)
    X_Train, X_val, X_Test, y_Train, y_val, y_Test, labels = prep.load_split_data()
    
    # Тест 2: Загрузка всех данных
    print("\n" + "="*70)
    print("ТЕСТ: Загрузка всех данных")
    print("="*70)
    X, y, labels = prep.load_data()
    
    print("\n✓ Модуль data_preparation работает корректно!")
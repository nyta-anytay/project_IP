"""
Вспомогательные функции
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def plot_class_distribution(y, labels_map, save_path=None):
    """
    Визуализация распределения классов
    
    Args:
        y: массив меток
        labels_map: словарь {индекс: название}
        save_path: путь для сохранения
    """
    counter = Counter(y)
    
    # Данные для графика
    classes = [labels_map[k] for k in sorted(counter.keys())]
    counts = [counter[k] for k in sorted(counter.keys())]
    
    # График
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    
    plt.title('Распределение классов в датасете', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Класс', fontsize=13)
    plt.ylabel('Количество образцов', fontsize=13)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Значения на столбцах
    for i, (bar, count) in enumerate(zip(bars, counts)):
        percentage = count / sum(counts) * 100
        plt.text(bar.get_x() + bar.get_width()/2., count + max(counts)*0.01,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Линия баланса
    perfect_balance = sum(counts) / len(counts)
    plt.axhline(y=perfect_balance, color='green', linestyle='--', 
               alpha=0.5, label=f'Идеальный баланс ({perfect_balance:.0f})')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ График распределения сохранен: {save_path}")
    
    plt.show()
    plt.close()


def plot_sample_images(X, y, labels_map, n_samples=5, save_path=None):
    """
    Визуализация примеров изображений из каждого класса
    
    Args:
        X: массив изображений
        y: массив меток
        labels_map: словарь {индекс: название}
        n_samples: количество примеров на класс
        save_path: путь для сохранения
    """
    classes = sorted(labels_map.keys())
    
    fig, axes = plt.subplots(len(classes), n_samples, 
                            figsize=(15, 5*len(classes)))
    
    # Если только один класс
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_id in enumerate(classes):
        # Получаем индексы изображений этого класса
        class_indices = np.where(y == class_id)[0]
        
        # Случайный выбор
        n_to_sample = min(n_samples, len(class_indices))
        sample_indices = np.random.choice(class_indices, n_to_sample, replace=False)
        
        for j in range(n_samples):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            
            if j < n_to_sample:
                idx = sample_indices[j]
                ax.imshow(X[idx])
                ax.set_title(f'Sample {j+1}', fontsize=10)
            
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(labels_map[class_id], 
                            fontsize=14, fontweight='bold', rotation=0, 
                            ha='right', va='center')
    
    plt.suptitle('Примеры изображений из датасета', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Примеры изображений сохранены: {save_path}")
    
    plt.show()
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Визуализация истории обучения нейронной сети
    
    Args:
        history: объект History из Keras
        save_path: путь для сохранения
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], 
                label='Train Accuracy', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], 
                label='Validation Accuracy', linewidth=2, marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], 
                label='Train Loss', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], 
                label='Validation Loss', linewidth=2, marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ История обучения сохранена: {save_path}")
    
    plt.show()
    plt.close()


def check_data_quality(X, y, labels_map):
    """
    Проверка качества данных
    
    Args:
        X: массив изображений
        y: массив меток
        labels_map: словарь {индекс: название}
    """
    from collections import Counter as CounterClass
    
    print("\n" + "="*70)
    print("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    print("="*70)
    
    # Базовая информация
    print(f"\nОбщая информация:")
    print(f"  Всего образцов:     {len(X)}")
    print(f"  Форма изображения:  {X[0].shape}")
    print(f"  Тип данных:         {X.dtype}")
    print(f"  Диапазон значений:  [{X.min()}, {X.max()}]")
    
    # Распределение по классам
    class_counter = CounterClass(y)
    print(f"\nРаспределение по классам:")
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = count / len(y) * 100
        print(f"  {labels_map[class_id]:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Баланс классов
    counts = list(class_counter.values())
    balance_ratio = min(counts) / max(counts) if counts else 0
    print(f"\nБаланс классов: {balance_ratio:.2%}")
    
    if balance_ratio < 0.7:
        print("  ⚠️  ВНИМАНИЕ: Сильный дисбаланс классов!")
        print("     Рекомендуется использовать взвешивание или аугментацию")
    elif balance_ratio < 0.9:
        print("  ⚠️  Небольшой дисбаланс классов")
    else:
        print("  ✓ Классы хорошо сбалансированы")
    
    # Проверка на дубликаты (по хешу) - только первые 1000 образцов
    print(f"\nПроверка на дубликаты:")
    try:
        sample_size = min(1000, len(X))
        print(f"  Проверяю {sample_size} образцов...")
        
        image_hashes = [hash(img.tobytes()) for img in X[:sample_size]]
        hash_counter = CounterClass(image_hashes)
        duplicates = sum(1 for count in hash_counter.values() if count > 1)
        
        print(f"  Уникальных изображений: {len(hash_counter)}")
        print(f"  Возможных дубликатов:   {duplicates}")
        
        if duplicates > 0:
            print("  ⚠️  Обнаружены возможные дубликаты!")
        else:
            print("  ✓ Дубликаты не обнаружены")
    except Exception as e:
        print(f"  ⚠️  Не удалось проверить дубликаты: {e}")
    
    print("="*70)


# Тестирование модуля
if __name__ == "__main__":
    print("Модуль utils.py готов к использованию")
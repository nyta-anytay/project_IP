"""
Оценка и сравнение моделей
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
from .config import RESULTS_DIR


class ModelEvaluator:
    """Класс для оценки и сравнения моделей"""
    
    def __init__(self, models, model_names, X_Test, y_Test, labels_map):
        """
        Args:
            models: список моделей
            model_names: список названий моделей
            X_Test: тестовые данные
            y_Test: тестовые метки
            labels_map: словарь {индекс: название_класса}
        """
        self.models = models
        self.model_names = model_names
        self.X_Test = X_Test
        self.y_Test = y_Test
        self.labels_map = labels_map
        
    def evaluate_all(self):
        """Оценка всех моделей"""
        results = []
        
        print("\n" + "="*70)
        print("ОЦЕНКА МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("="*70)
        
        for model, name in zip(self.models, self.model_names):
            print(f"\n{'─'*70}")
            print(f"📊 Модель: {name}")
            print(f"{'─'*70}")
            
            # Предсказания
            print("Выполнение предсказаний...")
            y_pred = model.predict(self.X_Test)
            y_proba = model.predict_proba(self.X_Test)
            
            # Вычисление метрик
            metrics = self._calculate_metrics(y_pred, y_proba, name)
            results.append(metrics)
            
            # Вывод метрик
            self._print_metrics(metrics)
            
            # Classification Report
            print("\n📋 Детальный отчет:")
            print(classification_report(
                self.y_Test, y_pred, 
                target_names=list(self.labels_map.values()),
                zero_division=0
            ))
            
            # Confusion Matrix
            self._plot_confusion_matrix(self.y_Test, y_pred, name)
            
            # ROC Curve для бинарной классификации
            if len(np.unique(self.y_Test)) == 2:
                self._plot_roc_curve(y_proba, name)
        
        # Сравнение моделей
        results_df = self._compare_models(results)
        
        return results_df
    
    def _calculate_metrics(self, y_pred, y_proba, model_name):
        """Вычисление метрик"""
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_Test, y_pred),
            'Precision': precision_score(self.y_Test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(self.y_Test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(self.y_Test, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC для бинарной классификации
        if len(np.unique(self.y_Test)) == 2:
            try:
                metrics['ROC-AUC'] = roc_auc_score(self.y_Test, y_proba[:, 1])
            except:
                metrics['ROC-AUC'] = 0.0
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Вывод метрик"""
        print("\n📈 Метрики качества:")
        for metric, value in metrics.items():
            if metric != 'Model':
                # Цветовое выделение
                if value >= 0.9:
                    symbol = "🟢"
                elif value >= 0.7:
                    symbol = "🟡"
                else:
                    symbol = "🔴"
                print(f"  {symbol} {metric:15s}: {value:.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Визуализация матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(self.labels_map.values()),
            yticklabels=list(self.labels_map.values()),
            cbar_kws={'label': 'Количество'},
            annot_kws={'size': 14}
        )
        
        plt.title(f'Confusion Matrix - {model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Истинный класс', fontsize=13)
        plt.xlabel('Предсказанный класс', fontsize=13)
        
        # Добавляем точность для каждого класса
        accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(accuracy_per_class):
            plt.text(len(cm) + 0.5, i + 0.5, f'{acc:.1%}', 
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        # Сохранение
        filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Confusion Matrix сохранена: {save_path}")
        plt.close()
    
    def _plot_roc_curve(self, y_proba, model_name):
        """Построение ROC кривой"""
        fpr, tpr, _ = roc_curve(self.y_Test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Сохранение
        filename = f'roc_curve_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        save_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ ROC Curve сохранена: {save_path}")
        plt.close()
    
    def _compare_models(self, results):
        """Сравнение всех моделей"""
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Лучшая модель по F1-Score
        best_idx = results_df['F1-Score'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_f1 = results_df.loc[best_idx, 'F1-Score']
        
        print(f"\n🏆 Лучшая модель: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")
        
        # Сохранение результатов
        csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Результаты сохранены: {csv_path}")
        
        # График сравнения
        self._plot_comparison(results_df)
        
        return results_df
    
    def _plot_comparison(self, results_df):
        """График сравнения моделей"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        if 'ROC-AUC' in results_df.columns:
            metrics.append('ROC-AUC')
        
        # Подготовка данных
        df_plot = results_df.set_index('Model')[metrics]
        
        # График
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(df_plot))
        width = 0.15
        multiplier = 0
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            offset = width * multiplier
            bars = ax.bar(x + offset, df_plot[metric], width, 
                         label=metric, color=colors[i], alpha=0.8)
            
            # Значения на столбцах
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            multiplier += 1
        
        ax.set_xlabel('Модель', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Сравнение моделей по всем метрикам', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(df_plot.index, rotation=15, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Сохранение
        save_path = os.path.join(RESULTS_DIR, 'models_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ График сравнения сохранен: {save_path}")
        plt.close()


# Тестирование модуля
if __name__ == "__main__":
    print("Модуль evaluation.py готов к использованию")
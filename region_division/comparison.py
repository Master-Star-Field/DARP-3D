import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from geodesic_kmeans import constrained_geodesic_kmeans
from potential_fields import divide_regions_potential_fields
from visualization import visualize_regions

def compare_methods(map_3d, num_agents, agent_positions, visualize=True, save_path=None):
    """
    Сравнивает методы разделения регионов по времени выполнения и балансу размеров
    
    Параметры:
        map_3d: 3D карта (0 - свободно, 1 - препятствие)
        num_agents: количество агентов
        agent_positions: позиции агентов [(x, y, z), ...]
        visualize: флаг для отображения результатов
        save_path: директория для сохранения результатов
    """
    # Методы для сравнения
    methods = [
        ("Геодезический K-means", constrained_geodesic_kmeans),
        ("Потенциальные поля", divide_regions_potential_fields)
    ]
    
    # Результаты для таблицы
    results = {
        "Метод": [],
        "Время выполнения (сек)": [],
        "Дисбаланс (%)": [],
        "Минимальный размер": [],
        "Максимальный размер": [],
        "Средний размер": []
    }
    
    # Выполняем каждый метод и собираем данные
    for method_name, method_func in methods:
        print(f"\nЗапуск метода: {method_name}")
        
        # Замеряем время выполнения
        start_time = time.time()
        region_assignment = method_func(map_3d, num_agents, agent_positions)
        elapsed_time = time.time() - start_time
        
        print(f"Время выполнения: {elapsed_time:.2f} сек")
        
        # Вычисляем размеры регионов и дисбаланс
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        min_size = min(region_sizes)
        max_size = max(region_sizes)
        avg_size = sum(region_sizes) / num_agents
        
        # Дисбаланс в процентах от среднего размера
        imbalance = (max_size - min_size) / avg_size * 100 if avg_size > 0 else 0
        
        # Сохраняем результаты
        results["Метод"].append(method_name)
        results["Время выполнения (сек)"].append(f"{elapsed_time:.2f}")
        results["Дисбаланс (%)"].append(f"{imbalance:.2f}")
        results["Минимальный размер"].append(min_size)
        results["Максимальный размер"].append(max_size)
        results["Средний размер"].append(f"{avg_size:.2f}")
        
        # Визуализируем результаты
        if visualize:
            save_file = f"{save_path}/{method_name.lower().replace(' ', '_')}.png" if save_path else None
            visualize_regions(map_3d, region_assignment, agent_positions, method_name, save_file)
    
    # Создаем таблицу результатов
    results_df = pd.DataFrame(results)
    
    # Выводим таблицу
    print("\nРезультаты сравнения методов:")
    print(results_df.to_string(index=False))
    
    # Сохраняем таблицу
    if save_path:
        results_df.to_csv(f"{save_path}/comparison_results.csv", index=False)
    
    # Создаем графики сравнения
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # График времени выполнения
        method_names = results["Метод"]
        execution_times = [float(t) for t in results["Время выполнения (сек)"]]
        
        ax1.bar(method_names, execution_times, color=['#3498db', '#e74c3c'])
        ax1.set_title("Время выполнения методов", fontsize=14)
        ax1.set_ylabel("Время (сек)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        for i, v in enumerate(execution_times):
            ax1.text(i, v + 0.1, f"{v:.2f}с", ha='center', fontsize=11)
        
        # График дисбаланса
        imbalances = [float(d) for d in results["Дисбаланс (%)"]]
        
        ax2.bar(method_names, imbalances, color=['#2ecc71', '#9b59b6'])
        ax2.set_title("Дисбаланс размеров регионов", fontsize=14)
        ax2.set_ylabel("Дисбаланс (%)", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        for i, v in enumerate(imbalances):
            ax2.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=11)
        
        plt.tight_layout()
        
        # Сохраняем график
        if save_path:
            plt.savefig(f"{save_path}/performance_comparison.png", dpi=300)
        
        plt.show()
    
    return results_df
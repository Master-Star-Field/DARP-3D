import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

def constrained_geodesic_kmeans(map_3d, num_agents, agent_positions, max_iterations=5):
    """
    Оптимизированное разделение 3D пространства на регионы с равномерным распределением
    
    Параметры:
        map_3d: 3D массив (0 - свободно, 1 - препятствие)
        num_agents: количество агентов
        agent_positions: позиции агентов [(x, y, z), ...]
        max_iterations: максимальное число итераций балансировки
        
    Возвращает:
        region_assignment: 3D массив (z, y, x) → agent_id или -1 (препятствие)
    """
    z_levels, height, width = map_3d.shape
    
    # Проверка и коррекция позиций агентов
    agent_positions = np.array(agent_positions, dtype=np.int32)
    for i, (x, y, z) in enumerate(agent_positions):
        if z < 0 or z >= z_levels or y < 0 or y >= height or x < 0 or x >= width:
            print(f"Предупреждение: Агент {i} вне карты! Корректировка положения.")
            agent_positions[i] = [width//2, height//2, z_levels//2]
        
        if map_3d[z, y, x] == 1:  # Агент в препятствии
            free_pos = find_nearest_free_cell(map_3d, x, y, z)
            if free_pos is not None:
                agent_positions[i] = [free_pos[2], free_pos[1], free_pos[0]]  # x,y,z
                print(f"Агент {i} перемещён в свободную ячейку {agent_positions[i]}")
    
    # Инициализация карты регионов
    region_assignment = np.full((z_levels, height, width), -1)
    region_assignment[map_3d == 1] = -1  # Препятствия
    
    # Находим свободные ячейки и их координаты
    free_cells = np.argwhere(map_3d == 0)
    if len(free_cells) == 0:
        print("Нет свободных ячеек!")
        return region_assignment
    
    # Координаты свободных ячеек в формате (x, y, z)
    free_cells_coords = free_cells[:, [2, 1, 0]]
    
    # K-means с начальными центроидами в позициях агентов
    kmeans = KMeans(n_clusters=num_agents, 
                   init=np.array(agent_positions), 
                   random_state=42, n_init=1)
    cluster_labels = kmeans.fit_predict(free_cells_coords)
    
    # Назначаем метки кластеров свободным ячейкам
    for i, (cell_z, cell_y, cell_x) in enumerate(free_cells):
        region_assignment[cell_z, cell_y, cell_x] = cluster_labels[i]
    
    # Цикл балансировки размеров регионов
    for iteration in range(max_iterations):
        print(f"Итерация балансировки {iteration+1}/{max_iterations}")
        
        # Текущие размеры регионов
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        total_free_cells = sum(region_sizes)
        target_size = total_free_cells // num_agents
        print(f"Целевой размер региона: {target_size} ячеек")
        
        # 1. Обеспечиваем связность регионов
        for agent_id in range(num_agents):
            region_mask = (region_assignment == agent_id)
            if not np.any(region_mask):
                continue
                
            # Находим компоненты связности в регионе
            labeled_array, num_components = ndimage.label(region_mask, structure=np.ones((3, 3, 3)))
            
            if num_components > 1:
                # Находим компонент с агентом или самый большой
                ax, ay, az = agent_positions[agent_id]
                if 0 <= az < z_levels and 0 <= ay < height and 0 <= ax < width:
                    agent_component = labeled_array[az, ay, ax]
                    
                    if agent_component > 0:
                        # Отключаем компоненты без агента
                        region_assignment[(labeled_array != agent_component) & region_mask] = -1
                    else:
                        # Находим самый большой компонент
                        component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_components + 1)]
                        largest_component = max(component_sizes, key=lambda x: x[1])[0]
                        region_assignment[(labeled_array != largest_component) & region_mask] = -1
        
        # 2. Переназначаем отключенные ячейки с учетом размеров регионов
        disconnected_mask = (map_3d == 0) & (region_assignment == -1)
        disconnected_cells = np.argwhere(disconnected_mask)
        
        if len(disconnected_cells) > 0:
            print(f"Переназначение {len(disconnected_cells)} отключенных ячеек...")
            
            for z, y, x in disconnected_cells:
                # Находим ближайший регион с учетом его текущего размера
                min_dist = float('inf')
                nearest_agent = -1
                
                for agent_id in range(num_agents):
                    # Координаты клеток текущего региона
                    region_cells = np.argwhere(region_assignment == agent_id)
                    if len(region_cells) == 0:
                        continue
                    
                    # Расстояние до региона (Manhattan)
                    dists = np.abs(region_cells[:, 0] - z) + np.abs(region_cells[:, 1] - y) + np.abs(region_cells[:, 2] - x)
                    min_region_dist = np.min(dists)
                    
                    # Увеличиваем расстояние для больших регионов
                    size_factor = region_sizes[agent_id] / target_size if target_size > 0 else 1
                    adjusted_dist = min_region_dist * size_factor
                    
                    if adjusted_dist < min_dist:
                        min_dist = adjusted_dist
                        nearest_agent = agent_id
                
                # Назначаем ячейку ближайшему региону
                if nearest_agent != -1:
                    region_assignment[z, y, x] = nearest_agent
                    region_sizes[nearest_agent] += 1
        
        # 3. Проверяем баланс размеров регионов
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        max_size = max(region_sizes)
        min_size = min(region_sizes)
        
        imbalance = (max_size - min_size) / target_size if target_size > 0 else 0
        print(f"Размеры регионов: {region_sizes}")
        print(f"Дисбаланс: {imbalance:.2f}")
        
        # Если дисбаланс невелик, завершаем
        if imbalance < 0.2:  # 20% дисбаланс допустим
            print("Достигнуто равномерное распределение!")
            break
        
        # 4. Иначе, перераспределяем граничные ячейки от больших регионов к малым
        if iteration < max_iterations - 1:
            for agent_id in range(num_agents):
                if region_sizes[agent_id] > target_size * 1.1:  # Регион на 10% больше цели
                    # Находим границу региона (клетки примыкающие к другим регионам)
                    region_mask = (region_assignment == agent_id)
                    eroded_mask = ndimage.binary_erosion(region_mask, structure=np.ones((3, 3, 3)))
                    boundary_mask = region_mask & ~eroded_mask
                    boundary_cells = np.argwhere(boundary_mask)
                    
                    # Находим самый маленький регион
                    smallest_agent = min(range(num_agents), key=lambda i: region_sizes[i])
                    
                    # Перемещаем до 10% ячеек
                    cells_to_move = min(int(region_sizes[agent_id] * 0.1), 
                                     region_sizes[agent_id] - target_size)
                    cells_to_move = min(cells_to_move, len(boundary_cells))
                    
                    if cells_to_move > 0:
                        import random
                        random.shuffle(boundary_cells)
                        move_cells = boundary_cells[:cells_to_move]
                        
                        for z, y, x in move_cells:
                            region_assignment[z, y, x] = smallest_agent
                            region_sizes[agent_id] -= 1
                            region_sizes[smallest_agent] += 1
    
    # Финальная проверка связности и назначение отделенных ячеек
    final_region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
    print(f"Финальное распределение: {final_region_sizes}")
    
    return region_assignment

def find_nearest_free_cell(map_3d, x, y, z):
    """Находит ближайшую свободную ячейку к указанной позиции"""
    z_levels, height, width = map_3d.shape
    
    # Корректировка координат к границам карты
    x = max(0, min(x, width-1))
    y = max(0, min(y, height-1))
    z = max(0, min(z, z_levels-1))
    
    # Если указанная позиция уже свободна
    if map_3d[z, y, x] == 0:
        return (z, y, x)
    
    # Вычисляем расстояние от указанной точки до всех ячеек
    point_mask = np.zeros_like(map_3d, dtype=bool)
    point_mask[z, y, x] = True
    
    dist = ndimage.distance_transform_edt(~point_mask)
    dist[map_3d == 1] = np.inf  # Бесконечное расстояние для препятствий
    
    # Находим ближайшую свободную ячейку
    if np.any(dist < np.inf):
        nearest_free = np.unravel_index(np.argmin(dist), dist.shape)
        return nearest_free
    
    return None  # Нет свободных ячеек
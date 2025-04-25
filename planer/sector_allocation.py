import numpy as np
from numba import njit
from scipy.spatial import cKDTree
import time
from collections import defaultdict
from scipy.ndimage import label, find_objects

@njit
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

@njit
def is_valid_point(point, shape):
    return (0 <= point[0] < shape[0] and 
            0 <= point[1] < shape[1] and 
            0 <= point[2] < shape[2])

def constrained_geodesic_kmeans(map_3d, num_agents, agent_positions, max_iterations=5):
    """
    алгоритм геодезических k-средних с гарантией связности секторов.
    
    Параметры:
        map_3d (numpy.ndarray): 3D карта препятствий (1 - препятствие, 0 - свободное пространство)
        num_agents (int): Количество агентов/секторов
        agent_positions (numpy.ndarray): Начальные позиции агентов формы (num_agents, 3)
        max_iterations (int): Максимальное количество итераций алгоритма
        
    Возвращает:
        numpy.ndarray: 3D массив меток секторов той же формы, что и map_3d
    """
    print("Разделение пространства на сектора...")
    start_time = time.time()
    
    depth, height, width = map_3d.shape
    
    regions = np.full((depth, height, width), -1, dtype=np.int32)
    
    free_cells = np.where(map_3d == 0)
    free_cells_coords = np.column_stack([free_cells[0], free_cells[1], free_cells[2]])
    
    if len(free_cells_coords) == 0:
        print("Нет свободных ячеек для назначения секторов")
        return regions
    
    agent_indices = np.round(agent_positions).astype(np.int32)
    
    for i, pos in enumerate(agent_indices):
        if not is_valid_point(pos, map_3d.shape) or map_3d[tuple(pos)] != 0:
            tree = cKDTree(free_cells_coords)
            dist, idx = tree.query(pos)
            agent_indices[i] = free_cells_coords[idx]
    
    tree = cKDTree(free_cells_coords)
    
    sector_weights = np.ones(num_agents)
    target_sector_size = len(free_cells_coords) / num_agents
    
    for iteration in range(max_iterations):
        print(f"Итерация {iteration+1}/{max_iterations}")
        
        distances = np.zeros((len(free_cells_coords), num_agents))
        
        for agent_id in range(num_agents):
            # Инициализируем структуры для волнового алгоритма
            visited = np.zeros_like(map_3d, dtype=bool)
            distance_map = np.full_like(map_3d, np.inf, dtype=float)
            
            agent_pos = tuple(agent_indices[agent_id])
            visited[agent_pos] = True
            distance_map[agent_pos] = 0
            
            queue = [agent_pos]
            
            directions = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0),
                (0, -1, 0), (0, 0, 1), (0, 0, -1)
            ]
            
            while queue:
                current = queue.pop(0)
                current_dist = distance_map[current]
                
                for d in directions:
                    neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
                    
                    if (is_valid_point(neighbor, map_3d.shape) and
                        map_3d[neighbor] == 0 and 
                        not visited[neighbor]):
                        
                        visited[neighbor] = True
                        next_dist = current_dist + 1
                        distance_map[neighbor] = next_dist
                        queue.append(neighbor)
            
            for i, (z, y, x) in enumerate(free_cells_coords):
                if visited[z, y, x]:
                    # Применяем вес сектора для балансировки размера
                    distances[i, agent_id] = distance_map[z, y, x] * sector_weights[agent_id]
                else:
                    # Если недостижимо, ставим очень большое значение
                    distances[i, agent_id] = np.inf
        
        cell_assignments = np.argmin(distances, axis=1)
        
        # Заполняем массив регионов
        regions.fill(-1)  # Сначала очищаем все регионы
        
        for i, (z, y, x) in enumerate(free_cells_coords):
            if not np.all(np.isinf(distances[i])):
                agent_id = cell_assignments[i]
                regions[z, y, x] = agent_id
        
        for agent_id in range(num_agents):
            # Создаем маску для текущего региона
            region_mask = (regions == agent_id)
            
            structure = np.ones((3, 3, 3), dtype=bool)  # 26-связное соседство
            labeled_regions, num_features = label(region_mask, structure=structure)
            
            if num_features > 1:
                agent_pos = tuple(agent_indices[agent_id])
                
                if region_mask[agent_pos]:
                    agent_component = labeled_regions[agent_pos]
                else:
                    component_sizes = np.bincount(labeled_regions.ravel())[1:]  # Исключаем фон (0)
                    agent_component = np.argmax(component_sizes) + 1
                
                # Сохраняем только компоненту с агентом
                regions[(labeled_regions != agent_component) & (labeled_regions > 0)] = -1
        
        # Заполняем пустые области ближайшим регионом
        empty_cells = np.where((regions == -1) & (map_3d == 0))
        empty_coords = np.column_stack([empty_cells[0], empty_cells[1], empty_cells[2]])
        
        if len(empty_coords) > 0:
            for z, y, x in empty_coords:
                min_dist = float('inf')
                best_agent = -1
                
                for d in [
                    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)
                ]:
                    nz, ny, nx = z + d[0], y + d[1], x + d[2]
                    
                    if (is_valid_point((nz, ny, nx), map_3d.shape) and
                        regions[nz, ny, nx] >= 0):
                        
                        curr_dist = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
                        if curr_dist < min_dist:
                            min_dist = curr_dist
                            best_agent = regions[nz, ny, nx]
                
                if best_agent >= 0:
                    regions[z, y, x] = best_agent
        
        for agent_id in range(num_agents):
            sector_cells = np.where(regions == agent_id)
            
            if len(sector_cells[0]) > 0:
                centroid = np.array([
                    np.mean(sector_cells[0]),
                    np.mean(sector_cells[1]),
                    np.mean(sector_cells[2])
                ])
                
                nearest_cell_idx = np.argmin(np.sum((free_cells_coords - centroid)**2, axis=1))
                agent_indices[agent_id] = free_cells_coords[nearest_cell_idx]

        sector_sizes = np.array([np.sum(regions == i) for i in range(num_agents)])
        size_ratios = sector_sizes / target_sector_size
        
        sector_weights = np.clip(size_ratios, 0.5, 2.0)
        
        # Нормализуем веса
        sector_weights = sector_weights / np.mean(sector_weights)
        
        if np.all(sector_sizes > 0) and np.max(size_ratios) / np.min(size_ratios) < 1.1:
            print(f"Достигнут баланс секторов, завершение на итерации {iteration+1}")
            break
    
    empty_cells = np.where((regions == -1) & (map_3d == 0))
    if len(empty_cells[0]) > 0:
        print(f"Заполнение {len(empty_cells[0])} оставшихся пустых ячеек...")
        
        empty_coords = np.column_stack([empty_cells[0], empty_cells[1], empty_cells[2]])
        
        non_empty_mask = (regions >= 0)
        if np.any(non_empty_mask):
            indices = np.where(non_empty_mask)
            non_empty_coords = np.column_stack([indices[0], indices[1], indices[2]])
            non_empty_values = regions[non_empty_mask]
            
            if len(non_empty_coords) > 0:
                tree = cKDTree(non_empty_coords)
                
                distances, indices = tree.query(empty_coords)
                
                for i, (z, y, x) in enumerate(empty_coords):
                    regions[z, y, x] = non_empty_values[indices[i]]
    
    print(f"Секторизация завершена за {time.time() - start_time:.2f} сек.")
    
    # Устанавливаем метки препятствий в -1
    regions[map_3d == 1] = -1
    
    return regions
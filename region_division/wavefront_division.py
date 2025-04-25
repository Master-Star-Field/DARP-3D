import numpy as np
import heapq
from collections import deque, defaultdict
from scipy import ndimage

def divide_regions_wavefront(map_3d, num_agents, agent_positions, balance_iterations=3):
    """
    Разделение пространства с помощью алгоритма распространения волнового фронта
    с последующей балансировкой размеров регионов
    
    Параметры:
        map_3d: 3D карта (0 - свободно, 1 - препятствие)
        num_agents: количество агентов
        agent_positions: позиции агентов [(x, y, z), ...]
        balance_iterations: количество итераций балансировки
        
    Возвращает:
        region_assignment: 3D массив (z, y, x) → agent_id или -1 (препятствие)
    """
    z_levels, height, width = map_3d.shape
    
    # Проверка и коррекция позиций агентов
    agent_positions = np.array(agent_positions, dtype=np.int32)
    corrected_positions = []
    
    for i, (x, y, z) in enumerate(agent_positions):
        if z < 0 or z >= z_levels or y < 0 or y >= height or x < 0 or x >= width:
            print(f"Предупреждение: Агент {i} вне карты! Корректировка положения.")
            x, y, z = width//2, height//2, z_levels//2
        
        if map_3d[z, y, x] == 1:  # Агент в препятствии
            free_pos = find_nearest_free_cell(map_3d, x, y, z)
            if free_pos is not None:
                z, y, x = free_pos
                print(f"Агент {i} перемещён в свободную ячейку ({x},{y},{z})")
        
        corrected_positions.append((x, y, z))
    
    # Инициализация карты регионов
    region_assignment = np.full((z_levels, height, width), -1)
    region_assignment[map_3d == 1] = -1  # Препятствия
    
    # Соседние ячейки (6-связность: верх, низ, север, юг, восток, запад)
    neighbors = [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]
    
    # Начальное распределение с помощью волнового фронта
    print("Начальное распространение волнового фронта...")
    
    # Очередь для волнового алгоритма: (расстояние, (x,y,z), agent_id)
    queue = []
    visited = set()
    
    # Добавляем начальные позиции агентов в очередь
    for agent_id, (x, y, z) in enumerate(corrected_positions):
        heapq.heappush(queue, (0, (x, y, z), agent_id))
        visited.add((x, y, z))
    
    # Распространяем волны
    while queue:
        dist, (x, y, z), agent_id = heapq.heappop(queue)
        
        # Если ячейка уже назначена другому агенту, пропускаем
        if region_assignment[z, y, x] != -1 and region_assignment[z, y, x] != agent_id:
            continue
        
        # Назначаем ячейку текущему агенту
        region_assignment[z, y, x] = agent_id
        
        # Распространяем волну дальше
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            
            # Проверяем границы карты
            if 0 <= nz < z_levels and 0 <= ny < height and 0 <= nx < width:
                # Пропускаем препятствия и уже посещенные ячейки
                if map_3d[nz, ny, nx] == 1 or (nx, ny, nz) in visited:
                    continue
                
                # Добавляем соседа в очередь
                heapq.heappush(queue, (dist + 1, (nx, ny, nz), agent_id))
                visited.add((nx, ny, nz))
    
    # Анализируем начальное распределение
    region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
    total_free_cells = sum(region_sizes)
    target_size = total_free_cells // num_agents
    
    print(f"Начальное распределение: {region_sizes}")
    print(f"Целевой размер: {target_size}")
    
    # Если распределение уже равномерное, возвращаем результат
    max_size = max(region_sizes)
    min_size = min(region_sizes)
    imbalance = (max_size - min_size) / target_size if target_size > 0 else 0
    
    if imbalance < 0.1:  # 10% дисбаланс считаем приемлемым
        print("Начальное распределение уже достаточно равномерное.")
        return region_assignment
    
    # Балансировка размеров регионов
    print("Выполняется балансировка размеров регионов...")
    
    for iteration in range(balance_iterations):
        print(f"Итерация балансировки {iteration+1}/{balance_iterations}")
        
        # Обновляем размеры регионов
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        
        # Находим регионы с максимальным и минимальным размером
        max_region_id = np.argmax(region_sizes)
        min_region_id = np.argmin(region_sizes)
        
        # Если уже достигнут хороший баланс, прекращаем
        if (region_sizes[max_region_id] - region_sizes[min_region_id]) <= target_size * 0.1:
            print("Достигнуто равномерное распределение!")
            break
        
        # Переносим ячейки из большого региона в маленький
        cells_to_move = min(
            int((region_sizes[max_region_id] - region_sizes[min_region_id]) / 2),
            int(region_sizes[max_region_id] * 0.1)  # Не более 10% от большого региона
        )
        
        # Находим граничные ячейки большого региона
        boundary_cells = find_boundary_cells(region_assignment, max_region_id)
        
        # Если нет граничных ячеек, переходим к следующей итерации
        if not boundary_cells:
            print(f"Не найдены граничные ячейки для региона {max_region_id}")
            continue
        
        # Сортируем граничные ячейки по близости к маленькому региону
        boundary_cells_with_dist = []
        
        # Находим ячейки маленького региона
        min_region_cells = np.argwhere(region_assignment == min_region_id)
        if len(min_region_cells) == 0:
            print(f"Регион {min_region_id} пуст!")
            continue
        
        for z, y, x in boundary_cells:
            # Вычисляем минимальное расстояние до ячеек маленького региона
            distances = np.abs(min_region_cells[:, 0] - z) + \
                      np.abs(min_region_cells[:, 1] - y) + \
                      np.abs(min_region_cells[:, 2] - x)
            min_dist = np.min(distances)
            boundary_cells_with_dist.append((min_dist, (z, y, x)))
        
        # Сортируем ячейки по расстоянию
        boundary_cells_with_dist.sort()
        
        # Перемещаем ближайшие ячейки в маленький регион
        for i, (_, (z, y, x)) in enumerate(boundary_cells_with_dist):
            if i >= cells_to_move:
                break
                
            # Проверяем, не нарушится ли связность большого региона
            temp_assignment = region_assignment.copy()
            temp_assignment[z, y, x] = min_region_id
            
            if check_connectivity(temp_assignment, max_region_id, corrected_positions[max_region_id]):
                region_assignment[z, y, x] = min_region_id
            else:
                print(f"Пропуск ячейки ({x},{y},{z}) - нарушается связность")
        
        # Обновляем размеры регионов
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        print(f"Размеры после балансировки: {region_sizes}")
    
    # Проверяем связность финальных регионов
    for agent_id in range(num_agents):
        ensure_connectivity(region_assignment, agent_id, corrected_positions[agent_id])
    
    # Финальное распределение
    final_region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
    print(f"Финальное распределение: {final_region_sizes}")
    
    return region_assignment

def find_boundary_cells(region_assignment, region_id):
    """Находит граничные ячейки региона (примыкающие к другим регионам)"""
    region_mask = (region_assignment == region_id)
    
    # Применяем эрозию для получения внутренних ячеек
    eroded_mask = ndimage.binary_erosion(region_mask, structure=np.ones((3, 3, 3)))
    
    # Граничные ячейки = все ячейки региона - внутренние ячейки
    boundary_mask = region_mask & ~eroded_mask
    
    return list(zip(*np.where(boundary_mask)))

def check_connectivity(region_assignment, region_id, agent_position):
    """Проверяет связность региона с учетом позиции агента"""
    x, y, z = agent_position
    region_mask = (region_assignment == region_id)
    
    if not np.any(region_mask):
        return True  # Пустой регион считается связным
    
    # Проверяем, что агент находится в своем регионе
    if region_assignment[z, y, x] != region_id:
        # Находим любую ячейку региона для начала проверки
        seed_point = np.argwhere(region_mask)[0]
        z, y, x = seed_point
    
    # Выполняем заливку из позиции агента
    labeled_array, _ = ndimage.label(region_mask, structure=np.ones((3, 3, 3)))
    
    # Если есть только одна компонента связности, регион связен
    return np.max(labeled_array) == 1

def ensure_connectivity(region_assignment, region_id, agent_position):
    """Обеспечивает связность региона, удаляя отделенные компоненты"""
    x, y, z = agent_position
    region_mask = (region_assignment == region_id)
    
    if not np.any(region_mask):
        return  # Пустой регион
    
    # Находим компоненты связности
    labeled_array, num_components = ndimage.label(region_mask, structure=np.ones((3, 3, 3)))
    
    if num_components <= 1:
        return  # Регион уже связен
    
    # Проверяем, что агент находится в своем регионе
    agent_component = 0
    if 0 <= z < region_assignment.shape[0] and 0 <= y < region_assignment.shape[1] and 0 <= x < region_assignment.shape[2]:
        if region_assignment[z, y, x] == region_id:
            agent_component = labeled_array[z, y, x]
    
    # Если агент не в своем регионе, берем самую большую компоненту
    if agent_component == 0:
        component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_components + 1)]
        largest_component = max(component_sizes, key=lambda x: x[1])[0]
        agent_component = largest_component
    
    # Удаляем отделенные компоненты
    disconnected_mask = (labeled_array != agent_component) & (region_assignment == region_id)
    region_assignment[disconnected_mask] = -1

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
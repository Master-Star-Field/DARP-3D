import numpy as np
import heapq
from scipy import ndimage

def divide_regions_potential_fields(map_3d, num_agents, agent_positions, max_iterations=10):
    """
    Разделение пространства с помощью потенциальных полей и динамической балансировки
    
    Параметры:
        map_3d: 3D карта (0 - свободно, 1 - препятствие)
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
    
    # Находим все свободные ячейки
    free_cells = np.argwhere(map_3d == 0)
    if len(free_cells) == 0:
        print("Нет свободных ячеек!")
        return np.full((z_levels, height, width), -1)
    
    # Целевой размер региона для равномерного распределения
    num_free_cells = len(free_cells)
    target_region_size = num_free_cells // num_agents
    print(f"Всего свободных ячеек: {num_free_cells}")
    print(f"Целевой размер региона: {target_region_size}")
    
    # Инициализация весов для балансировки (влияет на силу потенциала)
    region_weights = np.ones(num_agents)
    
    # Вычисляем карты геодезических расстояний от каждого агента
    agent_distance_maps = []
    
    for agent_id, (ax, ay, az) in enumerate(agent_positions):
        print(f"Вычисление карты расстояний для агента {agent_id}...")
        
        # Начальная точка - позиция агента
        if map_3d[az, ay, ax] == 1:
            # Если агент в препятствии, находим ближайшую свободную ячейку
            free_pos = find_nearest_free_cell(map_3d, ax, ay, az)
            if free_pos is not None:
                az, ay, ax = free_pos
        
        # Инициализация карты расстояний
        dist_map = np.full((z_levels, height, width), np.inf, dtype=float)
        dist_map[az, ay, ax] = 0
        
        # Соседние ячейки (6-связность)
        neighbors = [(0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0)]
        
        # Алгоритм Дейкстры для вычисления геодезических расстояний
        queue = [(0, (az, ay, ax))]
        
        while queue:
            curr_dist, (cz, cy, cx) = heapq.heappop(queue)
            
            # Если нашли лучший путь к ячейке ранее
            if curr_dist > dist_map[cz, cy, cx]:
                continue
            
            # Проверяем все соседние ячейки
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                
                # Проверка границ карты
                if 0 <= nz < z_levels and 0 <= ny < height and 0 <= nx < width:
                    # Пропускаем препятствия
                    if map_3d[nz, ny, nx] == 1:
                        continue
                    
                    # Расстояние до соседней ячейки
                    new_dist = curr_dist + 1
                    
                    # Если нашли более короткий путь
                    if new_dist < dist_map[nz, ny, nx]:
                        dist_map[nz, ny, nx] = new_dist
                        heapq.heappush(queue, (new_dist, (nz, ny, nx)))
        
        # Сохраняем карту расстояний
        agent_distance_maps.append(dist_map)
    
    # Основной цикл балансировки регионов
    for iteration in range(max_iterations):
        print(f"Итерация балансировки {iteration+1}/{max_iterations}")
        
        # Вычисляем взвешенные потенциальные поля
        weighted_fields = []
        for agent_id in range(num_agents):
            # Вес увеличивает "силу" потенциала, что уменьшает регион
            weighted_field = agent_distance_maps[agent_id] * region_weights[agent_id]
            weighted_fields.append(weighted_field)
        
        # Назначаем каждую ячейку агенту с наименьшим взвешенным расстоянием
        region_assignment = np.full((z_levels, height, width), -1)
        region_assignment[map_3d == 1] = -1  # Препятствия
        
        # Стек потенциальных полей для быстрого сравнения
        stacked_fields = np.stack(weighted_fields, axis=0)
        
        # Находим агента с минимальным потенциалом для каждой свободной ячейки
        free_mask = (map_3d == 0)
        min_agent_idx = np.argmin(stacked_fields[:, free_mask], axis=0)
        
        # Заполняем регионы
        for i, (z, y, x) in enumerate(free_cells):
            region_assignment[z, y, x] = min_agent_idx[i]
        
        # Обеспечиваем связность регионов
        for agent_id in range(num_agents):
            region_mask = (region_assignment == agent_id)
            if not np.any(region_mask):
                continue
                
            # Находим компоненты связности
            labeled_array, num_components = ndimage.label(region_mask, structure=np.ones((3, 3, 3)))
            
            if num_components > 1:
                # Находим компонент с агентом или самый большой
                ax, ay, az = agent_positions[agent_id]
                agent_component = 0
                
                if 0 <= az < z_levels and 0 <= ay < height and 0 <= ax < width:
                    if region_assignment[az, ay, ax] == agent_id:
                        agent_component = labeled_array[az, ay, ax]
                
                # Если не нашли компонент с агентом, используем самый большой
                if agent_component == 0:
                    component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_components + 1)]
                    largest_component = max(component_sizes, key=lambda x: x[1])[0]
                    agent_component = largest_component
                
                # Отмечаем отсоединенные части
                disconnected_mask = (labeled_array != agent_component) & (region_assignment == agent_id)
                region_assignment[disconnected_mask] = -1
        
        # Переназначаем отключенные ячейки
        disconnected_mask = (map_3d == 0) & (region_assignment == -1)
        disconnected_cells = np.argwhere(disconnected_mask)
        
        if len(disconnected_cells) > 0:
            print(f"Переназначение {len(disconnected_cells)} отключенных ячеек...")
            
            for z, y, x in disconnected_cells:
                min_dist = float('inf')
                nearest_agent = -1
                
                for agent_id in range(num_agents):
                    dist = agent_distance_maps[agent_id][z, y, x]
                    if dist < min_dist:
                        min_dist = dist
                        nearest_agent = agent_id
                
                if nearest_agent != -1:
                    region_assignment[z, y, x] = nearest_agent
        
        # Анализируем распределение
        region_sizes = [np.sum(region_assignment == i) for i in range(num_agents)]
        print(f"Текущие размеры регионов: {region_sizes}")
        
        # Расчет дисбаланса
        max_size = max(region_sizes)
        min_size = min(region_sizes)
        imbalance = (max_size - min_size) / target_region_size if target_region_size > 0 else 0
        print(f"Дисбаланс: {imbalance:.2f}")
        
        # Если распределение достаточно равномерное, выходим
        if imbalance < 0.1:  # 10% дисбаланс считаем приемлемым
            print("Достигнуто равномерное распределение!")
            break
        
        # Корректировка весов для следующей итерации
        for agent_id in range(num_agents):
            # Увеличиваем вес для больших регионов, уменьшаем для маленьких
            size_ratio = region_sizes[agent_id] / target_region_size if target_region_size > 0 else 1
            region_weights[agent_id] *= (size_ratio ** 0.5)  # Плавная корректировка
        
        # Нормализация весов
        mean_weight = np.mean(region_weights)
        region_weights /= mean_weight
        print(f"Скорректированные веса: {region_weights}")
    
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
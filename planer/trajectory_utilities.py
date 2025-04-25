import numpy as np
from numba import njit
from scipy.spatial import cKDTree
import networkx as nx

@njit
def is_path_clear_numba(p1, p2, map_3d, steps=8):
    """
    Оптимизированная JIT-компилированная функция проверки пути на препятствия.
    
    Параметры:
        p1, p2: Начальная и конечная точки пути
        map_3d: 3D карта препятствий
        steps: Число точек проверки
    """
    direction = p2 - p1
    distance = np.sqrt(np.sum((direction)**2))
    
    if distance < 1e-6:
        return True
    
    direction = direction / distance
    
    for i in range(1, steps):
        t = i / steps
        check_point = p1 + direction * distance * t
        
        # Округляем до индексов карты
        x, y, z = np.round(check_point).astype(np.int32)
        
        # Проверяем границы и наличие препятствия
        if (0 <= z < map_3d.shape[0] and 
            0 <= y < map_3d.shape[1] and 
            0 <= x < map_3d.shape[2]):
            if map_3d[z, y, x] > 0:  # Препятствие
                return False
        else:
            return False  # За границами карты
    
    return True

def create_optimized_serpentine(nodes, max_distance):
    """
    Создание оптимизированного змеевидного пути с гарантированным змеевидным шаблоном.
    """
    if len(nodes) < 2:
        return nodes
    
    # Преобразуем список в массив для векторизованных операций
    nodes_array = np.array([(x, y, z, idx) for x, y, z, idx in nodes])
    
    # Определяем границы слоя
    x_min, x_max = np.min(nodes_array[:, 0]), np.max(nodes_array[:, 0])
    
    # Определяем оптимальное число полос для змеевидного пути
    strip_width = max_distance * 0.8  # Немного уменьшаем для лучшего перекрытия
    num_strips = max(1, int(np.ceil((x_max - x_min) / strip_width)))
    
    # Точное вычисление ширины полосы для равномерного распределения
    strip_width = (x_max - x_min) / num_strips if num_strips > 1 else x_max - x_min
    
    # Определяем полосу для каждого узла (векторизованно)
    strip_indices = np.minimum(
        num_strips - 1, 
        ((nodes_array[:, 0] - x_min) / strip_width).astype(np.int32)
    )
    
    # Создаем змеевидный путь с гарантированным порядком
    snake_path = []
    
    # Для каждой полосы определяем узлы и сортируем их
    for strip_idx in range(num_strips):
        strip_mask = (strip_indices == strip_idx)
        if not np.any(strip_mask):
            continue
            
        strip_nodes = nodes_array[strip_mask]
        
        # Сортировка вверх или вниз по Y в зависимости от четности полосы
        if strip_idx % 2 == 0:
            sorted_indices = np.argsort(strip_nodes[:, 1])  # Снизу вверх
        else:
            sorted_indices = np.argsort(-strip_nodes[:, 1])  # Сверху вниз
        
        sorted_strip = strip_nodes[sorted_indices]
        
        # Добавляем промежуточные точки между полосами если нужно
        if snake_path and len(sorted_strip) > 0:
            last_point = snake_path[-1]
            first_point = sorted_strip[0]
            
            dist = np.linalg.norm(np.array(last_point[:3]) - np.array(first_point[:3]))
            if dist > max_distance:
                # Добавляем промежуточные точки для соединения полос
                steps = max(1, int(np.ceil(dist / max_distance)) - 1)
                for step in range(1, steps+1):
                    ratio = step / (steps + 1)
                    intermediate = (
                        last_point[0] + ratio * (first_point[0] - last_point[0]),
                        last_point[1] + ratio * (first_point[1] - last_point[1]),
                        last_point[2],
                        -1  # Маркер промежуточной точки
                    )
                    snake_path.append(intermediate)
        
        # Добавляем отсортированные узлы текущей полосы
        snake_path.extend(sorted_strip)
    
    return snake_path

def find_nearest_point_vectorized(agent_pos, layer_paths):
    """
    Векторизованный поиск ближайшей точки к агенту.
    """
    if not layer_paths:
        return None
    
    # Собираем все точки и их индексы в единый массив
    all_points = []
    point_indices = []
    
    for layer, path in layer_paths.items():
        if not path:
            continue
        points = np.array([p[:3] for p in path])
        all_points.append(points)
        point_indices.extend([(layer, idx) for idx in range(len(path))])
    
    if not all_points:
        return None
    
    # Объединяем все точки для векторизованного расчета
    all_points_array = np.vstack(all_points)
    
    # Векторизованный расчет расстояний
    agent_pos_array = np.array(agent_pos)
    distances = np.sum((all_points_array - agent_pos_array)**2, axis=1)
    
    # Находим индекс минимального расстояния
    nearest_idx = np.argmin(distances)
    
    return point_indices[nearest_idx]

def build_fast_layer_connections(layer_paths, sorted_layers, map_3d, max_edge_length, obstacle_check_cache):
    """
    Оптимизированное построение минимальных связей между слоями.
    """
    connections = {}
    
    for i in range(len(sorted_layers) - 1):
        current_layer = sorted_layers[i]
        next_layer = sorted_layers[i+1]
        
        if current_layer not in layer_paths or next_layer not in layer_paths:
            continue
            
        current_path = layer_paths[current_layer]
        next_path = layer_paths[next_layer]
        
        if len(current_path) == 0 or len(next_path) == 0:
            continue
        
        # Быстрая инициализация массивов точек
        current_coords = np.array([p[:3] for p in current_path])
        next_coords = np.array([p[:3] for p in next_path])
        
        # Экономичное использование KDTree
        k = min(3, len(next_path))
        tree = cKDTree(next_coords)
        distances, indices = tree.query(current_coords, k=k)
        
        # Эффективное построение графа для MST
        num_current = len(current_path)
        num_next = len(next_path)
        
        # Используем тип float32 для экономии памяти
        edges_from = []
        edges_to = []
        weights = []
        
        # Проверяем и добавляем только валидные ребра
        for curr_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for dist, next_idx in zip(dist_row, idx_row):
                if dist <= max_edge_length:
                    p1 = current_coords[curr_idx]
                    p2 = next_coords[next_idx]
                    
                    # Используем кэш для проверки препятствий
                    cache_key = (tuple(p1), tuple(p2))
                    if cache_key in obstacle_check_cache:
                        is_clear = obstacle_check_cache[cache_key]
                    else:
                        is_clear = is_path_clear_numba(p1, p2, map_3d)
                        obstacle_check_cache[cache_key] = is_clear
                    
                    if is_clear:
                        # Добавляем ребро в обоих направлениях для MST
                        from_idx = (current_layer, curr_idx)
                        to_idx = (next_layer, next_idx)
                        
                        edges_from.append(from_idx)
                        edges_to.append(to_idx)
                        weights.append(dist)
        
        # Если нет валидных ребер, пропускаем
        if not edges_from:
            continue
            
        # Построение MST для соединения слоев
        G = nx.Graph()
        
        for j in range(len(edges_from)):
            G.add_edge(edges_from[j], edges_to[j], weight=weights[j])
        
        mst = nx.minimum_spanning_tree(G)
        
        # Сохраняем все ребра MST
        for u, v in mst.edges():
            if u not in connections:
                connections[u] = []
            if v not in connections:
                connections[v] = []
            
            connections[u].append(v)
            connections[v].append(u)
    
    return connections

def build_efficient_snake_trajectory(layer_paths, connections, start_point, map_3d, max_edge_length, obstacle_check_cache):
    """
    Построение эффективной змеевидной траектории с минимальными переходами между слоями.
    
    Параметры:
        layer_paths: Словарь путей по слоям
        connections: Словарь соединений между точками разных слоев
        start_point: Начальная точка (layer, index)
        map_3d: 3D карта препятствий
        max_edge_length: Максимальная длина ребра
        obstacle_check_cache: Кэш для проверки препятствий
        
    Возвращает:
        Список точек траектории
    """
    if not layer_paths or not start_point:
        return []
    
    # Оптимизированное посещение всех точек с учетом соединений между слоями
    visited = set()
    current = start_point
    trajectory = []
    
    # BFS для доступа ко всем слоям
    layers_to_visit = list(layer_paths.keys())
    current_layer, current_idx = current
    
    # Обход всех слоев, начиная с текущего
    while layers_to_visit:
        # Определяем текущий слой для обхода
        if current_layer in layers_to_visit:
            layers_to_visit.remove(current_layer)
        
        # Обходим текущий путь по змеевидной схеме
        path = layer_paths[current_layer]
        
        # Добавляем все точки текущего пути
        for i in range(len(path)):
            point = path[i]
            point_id = (current_layer, i)
            
            if point_id not in visited:
                visited.add(point_id)
                trajectory.append(point[:3])  # Добавляем только координаты
                current_idx = i
        
        # Если все слои посещены, выходим
        if not layers_to_visit:
            break
        
        # Находим ближайший непосещенный слой
        current_point = (current_layer, current_idx)
        min_distance = float('inf')
        next_layer = None
        next_idx = None
        
        # Проверяем связи с точками в других слоях
        if current_point in connections:
            for connected_point in connections[current_point]:
                conn_layer, conn_idx = connected_point
                
                if conn_layer in layers_to_visit:
                    # Находим расстояние между точками
                    p1 = path[current_idx][:3]
                    p2 = layer_paths[conn_layer][conn_idx][:3]
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    
                    if dist < min_distance:
                        min_distance = dist
                        next_layer = conn_layer
                        next_idx = conn_idx
        
        # Если не нашли соединения, ищем ближайшую точку во всех непосещенных слоях
        if next_layer is None:
            for layer in layers_to_visit:
                if len(layer_paths[layer]) == 0:
                    continue
                
                p1 = path[current_idx][:3]
                layer_points = np.array([p[:3] for p in layer_paths[layer]])
                
                tree = cKDTree(layer_points)
                dist, idx = tree.query(p1, k=1)
                
                # Проверяем возможность прямого перехода
                p2 = layer_points[idx]
                
                cache_key = (tuple(p1), tuple(p2))
                if cache_key in obstacle_check_cache:
                    is_clear = obstacle_check_cache[cache_key]
                else:
                    is_clear = is_path_clear_numba(p1, p2, map_3d)
                    obstacle_check_cache[cache_key] = is_clear
                
                if is_clear and dist < min_distance:
                    min_distance = dist
                    next_layer = layer
                    next_idx = idx
        
        # Если нашли следующий слой, выполняем переход
        if next_layer is not None:
            # Добавляем промежуточные точки, если расстояние слишком большое
            p1 = path[current_idx][:3]
            p2 = layer_paths[next_layer][next_idx][:3]
            
            if min_distance > max_edge_length:
                steps = max(1, int(np.ceil(min_distance / max_edge_length)) - 1)
                for step in range(1, steps+1):
                    ratio = step / (steps + 1)
                    intermediate = (
                        p1[0] + ratio * (p2[0] - p1[0]),
                        p1[1] + ratio * (p2[1] - p1[1]),
                        p1[2] + ratio * (p2[2] - p1[2])
                    )
                    trajectory.append(intermediate)
            
            # Переходим к следующему слою
            current_layer = next_layer
            current_idx = next_idx
        else:
            # Если нет доступного перехода, выбираем случайный непосещенный слой
            if layers_to_visit:
                next_layer = layers_to_visit[0]
                current_layer = next_layer
                current_idx = 0
            else:
                break
    
    return trajectory
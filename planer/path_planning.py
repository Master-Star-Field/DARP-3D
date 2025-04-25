import sys, os
project_root = os.getcwd()
sys.path.append(project_root[:-6])
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import time
import networkx as nx
from grids.quadtree_grid import build_quadtree_grid_optimized
from trajectory_utilities import (
    is_path_clear_numba, 
    create_optimized_serpentine, 
    find_nearest_point_vectorized, 
    build_fast_layer_connections, 
    build_efficient_snake_trajectory
)

def build_full_3d_grid_quadtree_optimized(map_3d):
    """
    Строит полную 3D сетку, используя оптимизированное квадродерево для каждого слоя.
    
    Параметры:
        map_3d: 3D карта препятствий
        
    Возвращает:
        nodes_3d, edges_3d: Вершины и ребра 3D сетки
    """
    depth, height, width = map_3d.shape
    
    nodes_edges_per_layer = []
    
    # Обрабатываем каждый слой карты
    for z in range(depth):
        # Получаем 2D срез карты
        grid_slice = map_3d[z]
        
        nodes_2d, edges_2d = build_quadtree_grid_optimized(grid_slice)
        
        if len(nodes_2d) > 0:
            nodes_3d = np.column_stack((nodes_2d, np.full(len(nodes_2d), z)))
            nodes_edges_per_layer.append((nodes_3d, edges_2d, z))

    all_nodes = []
    all_edges = []
    node_offset = 0
    
    for nodes_3d, edges_2d, z in nodes_edges_per_layer:
        all_nodes.extend(nodes_3d)
        
        if len(edges_2d) > 0:
            updated_edges = edges_2d + node_offset
            all_edges.extend(updated_edges)
        
        node_offset += len(nodes_3d)
    
    if len(nodes_edges_per_layer) > 1:
        for i in range(len(nodes_edges_per_layer) - 1):
            current_nodes, _, current_z = nodes_edges_per_layer[i]
            next_nodes, _, next_z = nodes_edges_per_layer[i + 1]
            
            current_tree = cKDTree(current_nodes[:, :2])  # Только координаты X, Y
            next_tree = cKDTree(next_nodes[:, :2])
            
            # Находим пары ближайших точек между слоями
            for j, node in enumerate(current_nodes):
                dist, idx = next_tree.query(node[:2], k=1)
                
                if is_path_clear_numba(node, next_nodes[idx], map_3d):
                    # Вычисляем глобальные индексы
                    global_current_idx = sum(len(nodes) for nodes, _, _ in nodes_edges_per_layer[:i]) + j
                    global_next_idx = sum(len(nodes) for nodes, _, _ in nodes_edges_per_layer[:i+1]) + idx
                    
                    all_edges.append([global_current_idx, global_next_idx])
    
    return np.array(all_nodes), np.array(all_edges)

def build_trajectories_by_sectors(map_3d, regions, agent_positions, max_distance_in_layer=5.0, max_edge_length=7.0):
    """
    построение змеевидных траекторий для агентов.
    
    Параметры:
        map_3d: 3D карта препятствий
        regions: 3D массив меток секторов
        agent_positions: Позиции агентов
        max_distance_in_layer: Максимальное расстояние между точками в слое
        max_edge_length: Максимальная длина ребра
        
    Возвращает:
        Список траекторий для каждого агента
    """
    print("Построение 3D сетки...")
    start_time = time.time()
    nodes_3d, _ = build_full_3d_grid_quadtree_optimized(map_3d)
    n_nodes = len(nodes_3d)
    print(f"Построено узлов: {n_nodes}, время: {time.time() - start_time:.2f} сек.")
    
    if n_nodes == 0:
        return []
    
    n_agents = len(agent_positions)
    
    print("Распределение вершин по секторам и слоям...")
    node_coords_int = np.round(nodes_3d).astype(np.int32)
    
    valid_mask = (
        (node_coords_int[:, 0] >= 0) & (node_coords_int[:, 0] < regions.shape[2]) &
        (node_coords_int[:, 1] >= 0) & (node_coords_int[:, 1] < regions.shape[1]) &
        (node_coords_int[:, 2] >= 0) & (node_coords_int[:, 2] < regions.shape[0])
    )
    
    valid_indices = np.where(valid_mask)[0]
    valid_nodes = node_coords_int[valid_mask]
    
    if len(valid_nodes) == 0:
        return [[] for _ in range(n_agents)]
    
    # Эффективное получение меток секторов для всех узлов сразу
    sector_ids = regions[valid_nodes[:, 2], valid_nodes[:, 1], valid_nodes[:, 0]]
    
    # Оптимизированное распределение по секторам (без создания ненужных структур)
    sector_layer_nodes = [defaultdict(list) for _ in range(n_agents)]
    node_lookup = {}  # Для быстрого поиска узлов
    
    for i, (node_idx, sector_id) in enumerate(zip(valid_indices, sector_ids)):
        if 0 <= sector_id < n_agents:
            node = nodes_3d[node_idx]
            layer = int(round(node[2]))  # Округляем до целого слоя
            sector_layer_nodes[sector_id][layer].append((node[0], node[1], node[2], node_idx))
            node_lookup[(sector_id, layer, len(sector_layer_nodes[sector_id][layer])-1)] = (node[0], node[1], node[2])

    obstacle_check_cache = {}
    
    trajectories = []
    
    # Обрабатываем каждого агента
    for agent_id in range(n_agents):
        print(f"\nОбработка агента {agent_id}...")
        agent_layers = sector_layer_nodes[agent_id]
        
        if not agent_layers:
            print(f"У агента {agent_id} нет вершин в секторе")
            trajectories.append([])
            continue
        
        total_nodes = sum(len(nodes) for nodes in agent_layers.values())
        print(f"В секторе {agent_id} найдено {total_nodes} вершин в {len(agent_layers)} слоях")
        
        # Быстрая сортировка слоев
        sorted_layers = sorted(agent_layers.keys())
        
        # Создание оптимизированных змеевидных путей для каждого слоя
        layer_paths = {}
        for z in sorted_layers:
            nodes = agent_layers[z]
            if len(nodes) >= 2:  # Требуется минимум 2 узла для пути
                layer_paths[z] = create_optimized_serpentine(nodes, max_distance_in_layer)
        
        # Если после обработки не осталось слоев с путями, пропускаем агента
        if not layer_paths:
            print(f"Для агента {agent_id} не удалось построить пути в слоях")
            trajectories.append([])
            continue
        
        inter_layer_connections = build_fast_layer_connections(
            layer_paths, sorted_layers, map_3d, max_edge_length, obstacle_check_cache)
        
        start_point = find_nearest_point_vectorized(agent_positions[agent_id], layer_paths)
        
        if start_point:
            trajectory = build_efficient_snake_trajectory(
                layer_paths, inter_layer_connections, start_point, map_3d, max_edge_length, obstacle_check_cache)
            
            print(f"Построена траектория длиной {len(trajectory)} точек")
            trajectories.append(trajectory)
        else:
            print(f"Не удалось найти начальную точку для агента {agent_id}")
            trajectories.append([])
    
    return trajectories
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from quadtree_grid import process_layer_nodes

def build_deformed_grid(grid_3d, base_resolution=10, adaptive=True):
    """
    Создает 3D сетку путем деформации 2D сетки на основе препятствий.
    
    :param grid_3d: 3D массив, где 1 - препятствия, 0 - свободное пространство
    :param base_resolution: Базовое разрешение сетки
    :param adaptive: Использовать ли адаптивную сетку
    :return: (nodes, edges) - узлы и ребра сетки
    """
    depth, height, width = grid_3d.shape
    
    # Создаем базовую 2D сетку
    x = np.linspace(0, width, base_resolution)
    y = np.linspace(0, height, base_resolution)
    xx, yy = np.meshgrid(x, y)
    
    nodes_2d = np.column_stack((xx.flatten(), yy.flatten()))
    
    # Создаем ребра 2D сетки
    edges_2d = []
    for i in range(base_resolution):
        for j in range(base_resolution):
            idx = i * base_resolution + j
            # Горизонтальные ребра
            if j < base_resolution - 1:
                edges_2d.append((idx, idx + 1))
            # Вертикальные ребра
            if i < base_resolution - 1:
                edges_2d.append((idx, idx + base_resolution))
    
    edges_2d_array = np.array(edges_2d)
    
    # Деформируем сетку для каждого слоя
    nodes_3d = []
    edges_3d = []
    
    for z in range(depth):
        grid_slice = grid_3d[z]
        
        # Фильтруем узлы внутри препятствий
        valid_mask = process_layer_nodes(grid_slice, nodes_2d)
        valid_nodes = nodes_2d[valid_mask]
        
        # Если нет валидных узлов в слое, пропускаем
        if len(valid_nodes) == 0:
            continue
        
        # Добавляем координату Z
        nodes_with_z = np.column_stack((valid_nodes, np.full(len(valid_nodes), z)))
        
        # Сохраняем индексы узлов для отображения ребер
        node_indices = {tuple(node): len(nodes_3d) + i for i, node in enumerate(nodes_with_z)}
        
        # Добавляем узлы слоя
        nodes_3d.extend(nodes_with_z)
        
        # Добавляем ребра слоя
        if len(valid_nodes) > 1:
            # Ребра с учетом фильтрации узлов
            kdtree = cKDTree(valid_nodes)
            pairs = kdtree.query_pairs(width / (base_resolution - 1) * 1.5)
            
            for i, j in pairs:
                node1 = tuple(nodes_with_z[i])
                node2 = tuple(nodes_with_z[j])
                edges_3d.append((node_indices[node1], node_indices[node2]))
        
        # Соединяем слои
        if z > 0 and len(nodes_3d) > len(nodes_with_z):
            prev_layer_nodes = [node for node in nodes_3d if node[2] == z-1]
            
            if prev_layer_nodes:
                current_layer_nodes = nodes_with_z
                tree_prev = cKDTree([(node[0], node[1]) for node in prev_layer_nodes])
                
                for i, node in enumerate(current_layer_nodes):
                    dist, idx = tree_prev.query([node[0], node[1]], k=1)
                    prev_node = tuple(prev_layer_nodes[idx])
                    current_node = tuple(node)
                    
                    # Ребро между слоями
                    prev_idx = next(i for i, n in enumerate(nodes_3d) if tuple(n) == prev_node)
                    curr_idx = node_indices[current_node]
                    edges_3d.append((prev_idx, curr_idx))
    
    # Преобразуем в массивы NumPy
    nodes_3d_array = np.array(nodes_3d)
    edges_3d_array = np.array(edges_3d)
    
    return nodes_3d_array, edges_3d_array
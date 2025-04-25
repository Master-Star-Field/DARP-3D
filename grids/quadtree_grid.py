import numpy as np
from numba import njit
from collections import deque
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import networkx as nx

@njit
def collect_leaf_corners(leaf_cells, width, height):
    """Собирает углы ячеек листьев квадродерева"""
    corners = np.empty((len(leaf_cells) * 4, 2), dtype=np.float64)
    corner_count = 0
    
    for i in range(len(leaf_cells)):
        x0, y0, w, h = leaf_cells[i]
        
        if x0 >= 0 and y0 >= 0 and x0 + w <= width and y0 + h <= height:
            corners[corner_count] = (x0, y0)
            corner_count += 1
            corners[corner_count] = (x0 + w, y0)
            corner_count += 1
            corners[corner_count] = (x0, y0 + h)
            corner_count += 1
            corners[corner_count] = (x0 + w, y0 + h)
            corner_count += 1
    
    return corners[:corner_count]

@njit(parallel=True)
def process_layer_nodes(grid_slice, nodes_2d):
    """Обработка узлов слоя"""
    valid_mask = np.ones(len(nodes_2d), dtype=np.bool_)
    height, width = grid_slice.shape
    
    for i in range(len(nodes_2d)):
        x, y = nodes_2d[i]
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < width and 0 <= y_int < height) or grid_slice[y_int, x_int] == 1:
            valid_mask[i] = False
    
    return valid_mask

class QuadTreeOptimized:
    def __init__(self, x0, y0, width, height, min_size=1, max_depth=10):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.min_size = min_size
        self.max_depth = max_depth
        self.children = []
        self.is_leaf = True
        self.near_obstacle = False
    
    def build_quad_tree(self, grid_slice, dist_map, thresholds):
        """Нерекурсивное построение квадродерева"""
        queue = deque([(self, 0)])  # (узел, глубина)
        
        while queue:
            node, depth = queue.popleft()
            
            # Условия для остановки деления
            if depth >= self.max_depth or node.width <= self.min_size or node.height <= self.min_size:
                continue
            
            # Проверка необходимости деления
            x_min = int(max(0, node.x0))
            x_max = int(min(grid_slice.shape[1]-1, node.x0 + node.width))
            y_min = int(max(0, node.y0))
            y_max = int(min(grid_slice.shape[0]-1, node.y0 + node.height))
            
            needs_subdivision = False
            
            # Проверка препятствий
            if np.any(grid_slice[y_min:y_max+1, x_min:x_max+1] == 1):
                needs_subdivision = True
                node.near_obstacle = True
            
            # Проверка близости к препятствиям
            if not needs_subdivision:
                min_dist = np.min(dist_map[y_min:y_max+1, x_min:x_max+1])
                threshold = thresholds[min(depth, len(thresholds)-1)]
                if min_dist < threshold:
                    needs_subdivision = True
                    node.near_obstacle = True
            
            # Случайное деление для более равномерной сетки
            if not needs_subdivision and depth < 3:
                needs_subdivision = np.random.random() < 0.7
            elif not needs_subdivision and depth < 5:
                needs_subdivision = np.random.random() < 0.3
            
            if needs_subdivision:
                node.is_leaf = False
                w_half = node.width / 2
                h_half = node.height / 2
                
                # Создаем 4 дочерних квадранта
                node.children = [
                    QuadTreeOptimized(node.x0, node.y0, w_half, h_half, self.min_size, self.max_depth),
                    QuadTreeOptimized(node.x0 + w_half, node.y0, w_half, h_half, self.min_size, self.max_depth),
                    QuadTreeOptimized(node.x0, node.y0 + h_half, w_half, h_half, self.min_size, self.max_depth),
                    QuadTreeOptimized(node.x0 + w_half, node.y0 + h_half, w_half, h_half, self.min_size, self.max_depth)
                ]
                
                # Добавляем дочерние узлы в очередь
                for child in node.children:
                    queue.append((child, depth + 1))

def build_quadtree_grid_optimized(grid_slice, min_cell_size=1, max_depth=8):
    """
    Создает адаптивную сетку на основе квадродерева.
    
    :param grid_slice: 2D массив, где 1 - препятствия, 0 - свободное пространство
    :param min_cell_size: Минимальный размер ячейки квадродерева
    :param max_depth: Максимальная глубина рекурсии
    :return: (nodes, edges) - узлы и ребра сетки
    """
    height, width = grid_slice.shape
    
    # Карта расстояний
    dist_map = distance_transform_edt(1 - grid_slice)
    
    # Пороги расстояний в зависимости от глубины
    thresholds = np.array([5, 4, 3, 2.5, 2, 1.5, 1, 0.5, 0.5, 0.5])
    
    # Создаем корневой узел и запускаем построение
    root = QuadTreeOptimized(0, 0, width, height, min_cell_size, max_depth)
    root.build_quad_tree(grid_slice, dist_map, thresholds)
    
    # Сбор листовых ячеек
    leaf_cells = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        if node.is_leaf:
            leaf_cells.append((node.x0, node.y0, node.width, node.height))
        else:
            queue.extend(node.children)
    
    # Если нет ячеек, возвращаем пустые массивы
    if not leaf_cells:
        return np.array([]), np.array([])
    
    leaf_cells_array = np.array(leaf_cells)
    
    # Сбор углов ячеек
    corners = collect_leaf_corners(leaf_cells_array, width, height)
    
    # Удаление дубликатов
    unique_corners = np.unique(corners, axis=0)
    
    # Фильтрация углов внутри препятствий
    valid_corners = []
    for x, y in unique_corners:
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < width and 0 <= y_int < height and grid_slice[y_int, x_int] == 0:
            valid_corners.append((x, y))
    
    if not valid_corners:
        return np.array([]), np.array([])
    
    nodes_array = np.array(valid_corners)
    
    # Построение ребер с KDTree
    kdtree = cKDTree(nodes_array)
    edge_set = set()
    
    # Адаптивный радиус поиска
    avg_cell_size = np.mean([cell[2] for cell in leaf_cells])
    search_radius = avg_cell_size * 1.5
    
    # Находим соседей
    pairs = kdtree.query_pairs(search_radius)
    edge_set.update(pairs)
    
    # Проверка связности
    if nodes_array.size > 0 and edge_set:
        edges_array = np.array(list(edge_set))
        
        # Проверка с NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_array)))
        G.add_edges_from(edges_array)
        
        # Соединение компонент
        components = list(nx.connected_components(G))
        
        if len(components) > 1:
            for i in range(len(components) - 1):
                comp1 = list(components[i])
                comp2 = list(components[i + 1])
                
                tree1 = cKDTree(nodes_array[comp1])
                tree2 = cKDTree(nodes_array[comp2])
                
                distances, indices = tree1.query(nodes_array[comp2], k=1)
                min_idx = np.argmin(distances)
                
                node1 = comp1[indices[min_idx]]
                node2 = comp2[min_idx]
                edge_set.add((min(node1, node2), max(node1, node2)))
            
            edges_array = np.array(list(edge_set))
        
        return nodes_array, edges_array
    
    # Случай единственной точки
    if len(nodes_array) == 1:
        return nodes_array, np.array([])
    
    # Когда нет ребер, но есть точки
    if not edge_set and len(nodes_array) > 1:
        tree = cKDTree(nodes_array)
        distances, indices = tree.query(nodes_array, k=2)
        
        edges = [(i, indices[i, 1]) for i in range(len(nodes_array))]
        edges_array = np.array([(min(i, j), max(i, j)) for i, j in edges])
        edges_array = np.unique(edges_array, axis=0)
        
        return nodes_array, edges_array
    
    return nodes_array, np.array([])
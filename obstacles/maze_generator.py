import random
import numpy as np
from disjoint_set import DisjointSet
def generate_maze(rows, cols, add_connections=0):
    """
    Генерирует лабиринт с заданным числом строк и столбцов.
    
    Параметры:
        rows (int): Количество строк в лабиринте
        cols (int): Количество столбцов в лабиринте
        add_connections (int): Количество дополнительных соединений
        
    Возвращает:
        list: Список удаленных стен
    """
    disjoint_set = DisjointSet()
    all_walls = []

    # Генерируем горизонтальные стены (между столбцами)
    for row in range(rows):
        for col in range(cols - 1):
            all_walls.append(('h', row, col)) 

    # Генерируем вертикальные стены (между строками)
    for row in range(rows - 1):
        for col in range(cols):
            all_walls.append(('v', row, col)) 

    random.shuffle(all_walls)
    removed_walls = []


    for wall in all_walls:
        wall_type, row, col = wall
        
        if wall_type == 'h':  # Горизонтальная стена
            cell1 = (row, col)
            cell2 = (row, col + 1)
        else:  # Вертикальная стена
            cell1 = (row, col)
            cell2 = (row + 1, col)

        # Если объединение было успешным, удаляем стену
        if disjoint_set.union(cell1, cell2):
            removed_walls.append(wall)

    remaining_walls = list(set(all_walls) - set(removed_walls))
    extra_connections = min(add_connections, len(remaining_walls))
    removed_walls += random.sample(remaining_walls, extra_connections)

    return removed_walls

def render_maze(removed_walls, rows, cols, passage_width, wall_width, map_height):
    """
    Создает 3D карту лабиринта на основе удаленных стен.
    
    Параметры:
        removed_walls (list): Список удаленных стен
        rows, cols (int): Размеры лабиринта
        passage_width, wall_width (int): Ширина проходов и стен
        map_height (float): Максимальная высота стен
        
    Возвращает:
        numpy.ndarray: 3D массив (0 - проход, 1 - препятствие)
    """
    wall_heights = [i * (map_height / 10) for i in range(1, 11)]

    total_rows = rows * (passage_width + wall_width) + wall_width
    total_cols = cols * (passage_width + wall_width) + wall_width
    height_map = np.zeros((total_rows, total_cols))

    for i in range(total_rows):
        for j in range(total_cols):
            if i % (passage_width + wall_width) < wall_width or j % (passage_width + wall_width) < wall_width:
                height_map[i, j] = random.choice(wall_heights)
            else:
                height_map[i, j] = 0  # Проход

    for row in range(rows):
        for col in range(cols):
            room_row = row * (passage_width + wall_width) + wall_width
            room_col = col * (passage_width + wall_width) + wall_width
            height_map[room_row:room_row + passage_width, room_col:room_col + passage_width] = 0

    for wall in removed_walls:
        wall_type, row, col = wall
        
        if wall_type == 'h':
            passage_row = row * (passage_width + wall_width) + wall_width
            passage_col = col * (passage_width + wall_width) + wall_width + passage_width
            height_map[passage_row:passage_row + passage_width,
                     passage_col:passage_col + wall_width] = 0
        else: 
            passage_row = row * (passage_width + wall_width) + wall_width + passage_width
            passage_col = col * (passage_width + wall_width) + wall_width
            height_map[passage_row:passage_row + wall_width,
                     passage_col:passage_col + passage_width] = 0

    depth = int(map_height) + 1
    voxel_map = np.zeros((total_rows, total_cols, depth), dtype=np.uint8)
    
    for i in range(total_rows):
        for j in range(total_cols):
            if height_map[i, j] > 0:
                voxel_map[i, j, :int(height_map[i, j])] = 1
    
    return voxel_map
import numpy as np

def generate_mountains(num_mountains, map_size):
    """
    Генерирует 3D карту горной местности.
    
    Параметры:
        num_mountains (int): Количество гор
        map_size (tuple): Размер карты (ширина, высота, глубина)
        
    Возвращает:
        3D массив (0 - воздух, 1 - препятствие)
    """
    width, height, depth = map_size
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    terrain_heights = np.zeros((width, height))
    

    center_x, center_y = width/2, height/2
    spread = min(width, height)/4 
    
    for _ in range(num_mountains):

        mountain_x = np.random.normal(center_x, spread)  # Центр горы по X
        mountain_y = np.random.normal(center_y, spread)  # Центр горы по Y
        mountain_height = np.abs(np.random.normal(depth/5, depth/10))  # Высота горы, зависит от глубины
        mountain_width_x = np.abs(np.random.normal(width/15, width/30))  # Ширина горы по X, зависит от ширины карты
        mountain_width_y = np.abs(np.random.normal(height/15, height/30))  # Ширина горы по Y, зависит от высоты карты
        
        dx = (X - mountain_x) / mountain_width_x
        dy = (Y - mountain_y) / mountain_width_y
        terrain_heights += mountain_height * np.exp(-(dx**2 + dy**2))
    
    # Нормализация всот
    if terrain_heights.max() > 0:
        normalized_heights = (terrain_heights / terrain_heights.max()) * (depth-1)
    else:
        normalized_heights = np.zeros_like(terrain_heights)
    
    normalized_heights = normalized_heights.astype(int)
    
    # Создание 3D воксельной карты
    voxel_map = np.zeros((width, height, depth), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            if normalized_heights[i, j] > 0:
                voxel_map[i, j, :normalized_heights[i, j]+1] = 1
    
    return voxel_map
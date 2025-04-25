import numpy as np

def generate_spherical_obstacles(width, height, depth, num_obstacles, radius_range=(2, 5), wall_count=0, wall_length=10):
    """
    Генерирует 3D карту с сферическими препятствиями и стенами из сфер.
    
    Параметры:
        width, height, depth (int): Размеры карты
        num_obstacles (int): Количество сферических препятствий
        radius_range (tuple): Диапазон радиусов препятствий (мин, макс)
        wall_count (int): Количество стен из сферических препятствий
        wall_length (int): Длина стены (количество сфер)
        
    Возвращает:
        3D массив (0 - свободное пространство, 1 - препятствие)
    """
    obstacle_map = np.zeros((width, height, depth), dtype=np.uint8)
    
    x, y, z = np.indices((width, height, depth))
    
    for _ in range(num_obstacles):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        center_z = np.random.randint(0, depth)
        radius = np.random.randint(radius_range[0], radius_range[1] + 1)
        
        sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
        obstacle_map = np.logical_or(obstacle_map, sphere).astype(np.uint8)
    
    for _ in range(wall_count):
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        start_z = np.random.randint(0, depth)
        
        direction = np.random.randint(0, 3) 
        
        radius = np.random.randint(radius_range[0], radius_range[1] + 1)
        
        for i in range(wall_length):
            if direction == 0:
                center_x = min(start_x + i, width - 1)
                center_y = start_y
                center_z = start_z
            elif direction == 1:
                center_x = start_x
                center_y = min(start_y + i, height - 1)
                center_z = start_z
            else:
                center_x = start_x
                center_y = start_y
                center_z = min(start_z + i, depth - 1)
            

            sphere = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
            
            obstacle_map = np.logical_or(obstacle_map, sphere).astype(np.uint8)
    
    return obstacle_map
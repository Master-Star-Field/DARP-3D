import numpy as np

def generate_current_field(map_size, obstacles, vortices):
    """
    Генерация поля течений на основе моделей вихрей с учетом препятствий.
    
    Параметры:
    map_size: Кортеж размеров трехмерной области (nx, ny, nz)
    obstacles: Трехмерный массив препятствий, где 1 - препятствие, 0 - свободная область
    vortices: Список словарей с параметрами вихрей (x0, y0, z0, r0, zeta, lambda)
    
    Возвращает:
    current_field: Трехмерный массив векторов скорости, каждая ячейка содержит [Vx, Vy, Vz]
    """
    x = np.arange(map_size[0])
    y = np.arange(map_size[1])
    z = np.arange(map_size[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    Vx = np.zeros(map_size)
    Vy = np.zeros(map_size)
    Vz = np.zeros(map_size)
    
    for vortex in vortices:
        x0, y0, z0 = vortex['x0'], vortex['y0'], vortex['z0']
        r0 = vortex['r0']
        zeta = vortex['zeta']
        lambda_val = vortex['lambda']
        
        r = np.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)
        
        r_safe = np.maximum(r, 1e-10)
        
        exp_term = 1 - np.exp(-((r_safe - r0) / zeta)**2)
        
        Vx += -lambda_val * (Y - y0) / (2 * np.pi * (r_safe - r0)) * exp_term
        Vy += lambda_val * (X - x0) / (2 * np.pi * (r_safe - r0)) * exp_term
        Vz += lambda_val / (np.pi * zeta**2) * np.exp(-((r_safe - r0) / zeta)**2)
    
    Vx[obstacles == 1] = 0
    Vy[obstacles == 1] = 0
    Vz[obstacles == 1] = 0
    
    current_field = np.zeros((*map_size, 3))
    current_field[..., 0] = Vx
    current_field[..., 1] = Vy
    current_field[..., 2] = Vz
    
    return current_field

def generate_vortices(map_size, num_vortices=3):
    """
    Генерация случайных вихрей для формирования поля течений.
    
    Параметры:
    map_size: Кортеж размеров трехмерной области (nx, ny, nz)
    num_vortices: Количество генерируемых вихрей
    
    Возвращает:
    vortices: Список словарей с параметрами вихрей
    """
    vortices = []
    for _ in range(num_vortices):
        vortex = {
            'x0': np.random.randint(0, map_size[0]),
            'y0': np.random.randint(0, map_size[1]),
            'z0': np.random.randint(0, map_size[2]),
            'r0': np.random.uniform(5, 15),
            'zeta': np.random.uniform(2, 8),
            'lambda': np.random.uniform(-1, 1) * 10
        }
        vortices.append(vortex)
    
    return vortices
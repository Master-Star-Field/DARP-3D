import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

def visualize_current_field(current_field, obstacles, slice_idx=None, save_path=None):
    """
    Профессиональная визуализация поля течений в 3D и в виде срезов.
    
    Параметры:
    current_field: Трехмерный массив векторов скорости размера [nx, ny, nz, 3]
    obstacles: Трехмерный массив препятствий, где 1 - препятствие, 0 - свободная область
    slice_idx: Индекс среза для 2D-визуализации (если None, выбирается средний срез)
    save_path: Путь для сохранения изображения (если None, изображение отображается на экране)
    """
    # Настройка шрифтов для корректного отображения кириллицы
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    
    # Извлекаем компоненты скорости
    Vx = current_field[..., 0]
    Vy = current_field[..., 1]
    Vz = current_field[..., 2]
    
    # Расчет модуля скорости для цветовой карты
    V_magnitude = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    nx, ny, nz = obstacles.shape
    
    # Создаем 3D визуализацию
    fig = plt.figure(figsize=(18, 10))
    
    # 3D график поля течений
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Определяем разреженную сетку для более понятной визуализации
    step = max(1, min(nx, ny, nz) // 10)
    X, Y, Z = np.meshgrid(
        np.arange(0, nx, step),
        np.arange(0, ny, step),
        np.arange(0, nz, step),
        indexing='ij'
    )
    
    # Получаем значения скоростей на разреженной сетке
    U = Vx[::step, ::step, ::step]
    V = Vy[::step, ::step, ::step]
    W = Vz[::step, ::step, ::step]
    
    # Нормализуем векторы для лучшего отображения
    norm = np.sqrt(U**2 + V**2 + W**2)
    norm_safe = np.maximum(norm, 1e-10)
    U = U / norm_safe * step / 2
    V = V / norm_safe * step / 2
    W = W / norm_safe * step / 2
    
    # Рисуем векторы поля течений
    print(plt.cm.viridis(norm / np.max(norm) if np.max(norm) > 0 else norm))
    ax1.quiver(X, Y, Z, U, V, W, length=1.0, normalize=False, 
              color=[0.204903, 0.375746, 0.553533, 1.])
    
    # Отображаем препятствия в виде точек
    obstacle_points = np.where(obstacles == 1)
    if len(obstacle_points[0]) > 0:
        ax1.scatter(obstacle_points[0], obstacle_points[1], obstacle_points[2], 
                   color='red', s=10, alpha=0.5, label='Препятствия')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D визуализация поля течений')
    ax1.legend()
    
    # Визуализация среза
    ax2 = fig.add_subplot(122)
    
    # Если срез не указан, берем средний срез по Z
    if slice_idx is None:
        slice_idx = nz // 2
    
    # Обеспечиваем, что срез находится в пределах размеров массива
    slice_idx = min(max(slice_idx, 0), nz - 1)
    
    # Срез поля течений
    Vx_slice = Vx[:, :, slice_idx]
    Vy_slice = Vy[:, :, slice_idx]
    V_mag_slice = V_magnitude[:, :, slice_idx]
    obstacles_slice = obstacles[:, :, slice_idx]
    
    # Создаем сетку для среза
    x_slice = np.arange(0, nx)
    y_slice = np.arange(0, ny)
    X_slice, Y_slice = np.meshgrid(x_slice, y_slice, indexing='ij')
    
    # Рисуем цветовую карту скорости
    im = ax2.pcolormesh(X_slice, Y_slice, V_mag_slice, cmap='viridis', shading='auto')
    cb = plt.colorbar(im, ax=ax2)
    cb.set_label('Модуль скорости')
    
    # Разреженность для векторов на срезе
    slice_step = max(1, min(nx, ny) // 20)
    
    # Рисуем векторы поля течений на срезе
    ax2.quiver(X_slice[::slice_step, ::slice_step], Y_slice[::slice_step, ::slice_step],
              Vx_slice[::slice_step, ::slice_step], Vy_slice[::slice_step, ::slice_step],
              color='white', scale=20)
    
    # Отображаем препятствия на срезе
    obstacle_mask = obstacles_slice == 1
    if np.any(obstacle_mask):
        ax2.scatter(X_slice[obstacle_mask], Y_slice[obstacle_mask], 
                   marker='s', s=30, color='red', label='Препятствия')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Срез поля течений (Z = {slice_idx})')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
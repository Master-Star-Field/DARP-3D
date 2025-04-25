import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
from maze_generator import generate_maze, render_maze
from mountain_generator import generate_mountains
from spherical_obstacles_generator import generate_spherical_obstacles

def setup_russian_font():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12

def visualize_maze(rows=10, cols=10, add_connections=5, passage_width=3, wall_width=1, map_height=10):
    removed_walls = generate_maze(rows, cols, add_connections)
    maze_map = render_maze(removed_walls, rows, cols, passage_width, wall_width, map_height)
    
    maze_slice = np.zeros((maze_map.shape[0], maze_map.shape[1]))
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            for k in range(maze_map.shape[2]-1, -1, -1):
                if maze_map[i, j, k] == 1:
                    maze_slice[i, j] = k
                    break
    
    plt.figure(figsize=(10, 8))
    plt.imshow(maze_slice.T, cmap='viridis', interpolation='none')
    plt.title('Лабиринт (вид сверху)')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Высота препятствия')
    plt.tight_layout()

    plt.savefig('maze_visualization.png', dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    voxels = np.zeros(maze_map.shape, dtype=bool)
    voxels[maze_map == 1] = True
    
    colors = np.empty(voxels.shape + (4,), dtype=np.float32)
    colors[..., 0] = 0.5  
    colors[..., 1] = 0.7
    colors[..., 2] = 0.9 
    colors[..., 3] = 0.7  
    
    ax.voxels(voxels, facecolors=colors, edgecolor='gray', alpha=0.7)
    
    ax.view_init(elev=30, azim=45)
    
    ax.set_title('3D визуализация лабиринта')
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    ax.set_zlabel('Z координата (высота)')
    
    plt.savefig('maze_3d_visualization.png', dpi=300)
    plt.close()

def visualize_mountains(num_mountains=5, map_size=(50, 50, 20)):
    """Визуализация подводных гор"""
    width, height, depth = map_size
    
    print(f"Генерация гор с размерами: {width}x{height}x{depth}")
    
    mountain_map = generate_mountains(num_mountains, map_size)
    
    height_map = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            for k in range(depth-1, -1, -1):
                if mountain_map[i, j, k] == 1:
                    height_map[i, j] = k
                    break
    
    plt.figure(figsize=(10, 8))
    
    #Освещение это круто!
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(height_map, plt.cm.terrain, vert_exag=0.3, blend_mode='soft')
    
    plt.imshow(rgb)
    plt.title('Горный ландшафт (карта высот)')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.colorbar(label='Высота')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    

    plt.savefig('mountain_visualization.png', dpi=300)
    plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, height_map.T, cmap='terrain', 
                           linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_title('3D визуализация горного ландшафта')
    ax.set_xlabel('X координата')
    ax.set_ylabel('Y координата')
    ax.set_zlabel('Высота')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Высота')
    
    plt.savefig('mountain_3d_visualization.png', dpi=300)
    plt.close()

def visualize_spherical_obstacles(width=30, height=30, depth=30, num_obstacles=10, wall_count=2):
    obstacles_map = generate_spherical_obstacles(
        width, height, depth, num_obstacles, 
        radius_range=(2, 5), wall_count=wall_count, wall_length=8
    )
    
    xy_projection = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            for k in range(depth-1, -1, -1):
                if obstacles_map[i, j, k] == 1:
                    xy_projection[i, j] = k + 1  # +1 чтобы избежать нулевых значений
                    break
    
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    
    ax1 = plt.subplot(gs[0])
    
    cmap = plt.cm.plasma
    im = ax1.imshow(xy_projection.T, cmap=cmap, interpolation='bilinear')
    
    contour = ax1.contour(xy_projection.T, levels=5, colors='white', alpha=0.4, linewidths=0.8)
    
    ax1.set_title('Сферические препятствия (вид сверху)', fontsize=14)
    ax1.set_xlabel('X координата', fontsize=12)
    ax1.set_ylabel('Y координата', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5, color='white')
    
    cbar = plt.colorbar(im, ax=ax1, label='Высота препятствия')
    cbar.set_label('Высота препятствия', fontsize=12)
    
    ax2 = plt.subplot(gs[1], projection='3d')
    x, y, z = np.where(obstacles_map == 1)
    
    if len(x) > 5000:
        indices = np.random.choice(len(x), 5000, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    
    scatter = ax2.scatter(x, y, z, c=z, cmap='plasma', marker='o', s=25, alpha=0.7, edgecolors='none')
    
    ax2.set_title('3D визуализация сферических препятствий', fontsize=14)
    ax2.set_xlabel('X координата')
    ax2.set_ylabel('Y координата')
    ax2.set_zlabel('Z координата')
    
    ax2.view_init(elev=30, azim=45)
    
    cbar2 = plt.colorbar(scatter, ax=ax2, label='Высота')
    cbar2.set_label('Высота', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('spherical_obstacles_visualization.png', dpi=300)
    plt.close()

def visualize_all():
    """Визуализация всех методов генерации препятствий"""
    setup_russian_font()
    
    visualize_maze(rows=8, cols=8, add_connections=3)
    
    visualize_mountains(num_mountains=4, map_size=(100, 100, 40))
    
    visualize_spherical_obstacles(width=30, height=30, depth=20, num_obstacles=15, wall_count=3)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    img1 = plt.imread('maze_visualization.png')
    plt.imshow(img1)
    plt.title('Лабиринт')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    img2 = plt.imread('mountain_visualization.png')
    plt.imshow(img2)
    plt.title('Горный ландшафт')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    img3 = plt.imread('spherical_obstacles_visualization.png')
    plt.imshow(img3)
    plt.title('Сферические препятствия')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('all_methods_comparison.png', dpi=300)
    plt.close()
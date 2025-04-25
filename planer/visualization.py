import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

def visualize_sectors_and_trajectories(map_3d, regions, trajectories, agent_positions):
    """
    Визуализирует сектора и траектории агентов в 3D пространстве.
    """
    fig = go.Figure()
    
    # Получаем размеры карты
    depth, height, width = map_3d.shape
    n_agents = len(agent_positions)
    
    # Создаем цветовую схему для секторов
    colors = [
        'rgba(255, 0, 0, 0.5)', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 0, 255, 0.5)',
        'rgba(255, 255, 0, 0.5)', 'rgba(255, 0, 255, 0.5)', 'rgba(0, 255, 255, 0.5)',
        'rgba(128, 0, 0, 0.5)', 'rgba(0, 128, 0, 0.5)', 'rgba(0, 0, 128, 0.5)',
        'rgba(128, 128, 0, 0.5)', 'rgba(128, 0, 128, 0.5)', 'rgba(0, 128, 128, 0.5)'
    ]
    
    # Убеждаемся, что у нас достаточно цветов для всех агентов
    while len(colors) < n_agents:
        colors.extend(colors)
    
    # Визуализируем препятствия - обратите внимание на порядок осей: z теперь вертикальная
    x_obs, y_obs,  z_obs = np.where(map_3d == 1)
    
    fig.add_trace(go.Scatter3d(
        x=x_obs, 
        y=y_obs, 
        z=z_obs,  # z теперь отображается вертикально
        mode='markers',
        marker=dict(
            size=2,
            color='grey',
            opacity=0.3
        ),
        name='Препятствия'
    ))
    
    # Визуализируем сектора
    for agent_id in range(n_agents):
        x_sector, y_sector, z_sector = np.where(regions == agent_id)
        
        if len(z_sector) > 0:
            # Выбираем подмножество точек для ускорения отрисовки
            if len(z_sector) > 1000:
                indices = np.random.choice(len(z_sector), 1000, replace=False)
                x_sector, y_sector, z_sector = z_sector[indices], y_sector[indices], x_sector[indices]
            
            fig.add_trace(go.Scatter3d(
                x=x_sector, 
                y=y_sector, 
                z=z_sector,  # z теперь отображается вертикально
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[agent_id % len(colors)],
                    opacity=0.2
                ),
                name=f'Сектор {agent_id}'
            ))
    
    # Визуализируем траектории
    for agent_id, trajectory in enumerate(trajectories):
        if trajectory:
            traj_array = np.array(trajectory)
            
            fig.add_trace(go.Scatter3d(
                x=traj_array[:, 0], 
                y=traj_array[:, 1], 
                z=traj_array[:, 2],  # z отображается вертикально
                mode='lines',
                line=dict(
                    color=colors[agent_id % len(colors)].replace('0.5', '1.0'),
                    width=5
                ),
                name=f'Траектория агента {agent_id}'
            ))
    
    # Визуализируем начальные позиции агентов
    agent_pos_array = np.array(agent_positions)
    
    fig.add_trace(go.Scatter3d(
        x=agent_pos_array[:, 0], 
        y=agent_pos_array[:, 1], 
        z=agent_pos_array[:, 2],  # z отображается вертикально
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol='diamond',
            line=dict(color='white', width=1)
        ),
        name='Начальные позиции агентов'
    ))
    
    # Настройка макета с правильной ориентацией осей
    fig.update_layout(
        title='Сектора и траектории агентов',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (Глубина)',  # Подпись для оси Z
            aspectmode='data'
        ),
        legend=dict(
            title="Легенда:",
            font=dict(
                family="Arial",
                size=12,
                color="black"
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def visualize_single_agent_trajectory(map_3d, regions, trajectory, agent_position, agent_id):
    """
    Визуализирует траекторию одного агента в его секторе.
    """
    fig = go.Figure()
    
    # Получаем размеры карты
    height, width, depth = map_3d.shape
    
    # Визуализируем препятствия в секторе
    x_obs, y_obs, z_obs = np.where((map_3d == 1) & (regions == agent_id))
    
    if len(z_obs) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_obs, y=y_obs, z=z_obs,
            mode='markers',
            marker=dict(
                size=2,
                color='grey',
                opacity=0.3
            ),
            name='Препятствия в секторе'
        ))
    
    # Визуализируем границы сектора
    x_sector, y_sector, z_sector = np.where(regions == agent_id)
    
    if len(z_sector) > 0:
        # Выбираем подмножество точек для ускорения отрисовки
        if len(z_sector) > 1000:
            indices = np.random.choice(len(z_sector), 1000, replace=False)
            x_sector, y_sector, z_sector = x_sector[indices], y_sector[indices], z_sector[indices]
        
        fig.add_trace(go.Scatter3d(
            x=x_sector, y=y_sector, z=z_sector,
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(0, 255, 0, 0.2)',
                opacity=0.2
            ),
            name=f'Сектор {agent_id}'
        ))
    
    # Визуализируем траекторию
    if trajectory:
        traj_array = np.array(trajectory)
        
        # Линия траектории
        fig.add_trace(go.Scatter3d(
            x=traj_array[:, 0], y=traj_array[:, 1], z=traj_array[:, 2],
            mode='lines',
            line=dict(
                color='rgba(0, 0, 255, 0.7)',
                width=4
            ),
            name='Траектория'
        ))
        
        # Точки траектории с цветовой градацией
        points_color = np.linspace(0, 1, len(trajectory))
        
        fig.add_trace(go.Scatter3d(
            x=traj_array[:, 0], y=traj_array[:, 1], z=traj_array[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=points_color,
                colorscale=[[0, 'blue'], [0.5, 'green'], [1, 'red']],
                colorbar=dict(
                    title="Порядок посещения",
                    x=1.1
                ),
                opacity=0.8
            ),
            name='Точки траектории'
        ))
    
    # Визуализируем начальную позицию агента
    fig.add_trace(go.Scatter3d(
        x=[agent_position[0]], y=[agent_position[1]], z=[agent_position[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        name='Начальная позиция'
    ))
    
    # Настройка макета
    fig.update_layout(
        title=f'Траектория агента {agent_id}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (Глубина)',
            aspectmode='data'
        ),
        legend=dict(
            title="Легенда:",
            font=dict(
                family="Arial",
                size=12,
                color="black"
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def visualize_layer_coverage(map_3d, regions, trajectories, layer_idx):
    """
    Визуализирует покрытие слоя траекториями агентов.
    """
    fig = go.Figure()
    
    # Получаем 2D срез карты и регионов
    map_slice = map_3d[layer_idx]
    regions_slice = regions[layer_idx]
    
    # Визуализируем препятствия
    x_obs, y_obs = np.where(map_slice == 1)
    
    fig.add_trace(go.Scatter(
        x=x_obs, y=y_obs,
        mode='markers',
        marker=dict(
            size=5,
            color='black',
            opacity=0.5
        ),
        name='Препятствия'
    ))
    
    # Визуализируем сектора с разными цветами
    colors = [
        'rgba(255, 0, 0, 0.3)', 'rgba(0, 255, 0, 0.3)', 'rgba(0, 0, 255, 0.3)',
        'rgba(255, 255, 0, 0.3)', 'rgba(255, 0, 255, 0.3)', 'rgba(0, 255, 255, 0.3)',
        'rgba(128, 0, 0, 0.3)', 'rgba(0, 128, 0, 0.3)', 'rgba(0, 0, 128, 0.3)'
    ]
    
    for agent_id in range(max(len(trajectories), len(colors))):
        x_sector, y_sector = np.where(regions_slice == agent_id)
        
        if len(y_sector) > 0:
            fig.add_trace(go.Scatter(
                x=x_sector, y=y_sector,
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[agent_id % len(colors)],
                    opacity=0.3
                ),
                name=f'Сектор {agent_id}'
            ))
    
    # Визуализируем траектории в слое
    line_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
    
    for agent_id, trajectory in enumerate(trajectories):
        if trajectory:
            traj_array = np.array(trajectory)
            
            # Находим точки траектории, которые находятся на заданном слое
            layer_mask = np.abs(traj_array[:, 2] - layer_idx) < 0.5  # Допустимое отклонение
            layer_points = traj_array[layer_mask]
            
            if len(layer_points) > 0:
                fig.add_trace(go.Scatter(
                    x=layer_points[:, 0], y=layer_points[:, 1],
                    mode='lines+markers',
                    line=dict(
                        color=line_colors[agent_id % len(line_colors)],
                        width=3
                    ),
                    marker=dict(
                        size=7,
                        symbol='circle',
                        opacity=0.8
                    ),
                    name=f'Траектория агента {agent_id}'
                ))
    
    # Настройка макета
    fig.update_layout(
        title=f'Покрытие слоя {layer_idx}',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y', scaleanchor="x", scaleratio=1),
        legend=dict(
            title="Легенда:",
            font=dict(
                family="Arial",
                size=12,
                color="black"
            )
        )
    )
    
    return fig

def save_figure_as_png(fig, filename, width=1600, height=900):
    """
    Сохраняет фигуру Plotly как PNG-изображение.
    """
    
    fig.write_image(f"{filename}.png", width=width, height=height)
    print(f"Изображение сохранено: {filename}.png")

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.colors as pcolors

def visualize_regions(map_3d, region_assignment, agent_positions, method_name="", save_path=None):
    """
    Визуализирует 3D карту с разделенными регионами используя Plotly
    
    Параметры:
        map_3d: 3D карта (0 - свободно, 1 - препятствие)
        region_assignment: 3D массив с назначением регионов
        agent_positions: позиции агентов [(x, y, z), ...]
        method_name: название метода для заголовка
        save_path: путь для сохранения визуализации
    """
    width, height, z_levels = map_3d.shape
    num_agents = len(agent_positions)
    
    # Создаем максимально контрастный набор цветов для секторов
    distinct_colors = [
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', 
        '#FF7F00', '#FFFF33', '#A65628', '#F781BF',
        '#1B9E77', '#D95F02', '#7570B3', '#E7298A', 
        '#66A61E', '#E6AB02', '#A6761D', '#666666'
    ]
    
    # Если агентов больше, чем цветов, сгенерируем дополнительные
    if num_agents > len(distinct_colors):
        more_colors = px.colors.sample_colorscale(
            px.colors.diverging.Spectral, 
            [i/(num_agents-len(distinct_colors)) for i in range(num_agents-len(distinct_colors))]
        )
        distinct_colors.extend(more_colors)
    
    # Обрезаем до нужного количества
    distinct_colors = distinct_colors[:num_agents]
    
    # Создаем проекцию вида сверху (XY)
    xy_projection = np.zeros((width, height), dtype=int)
    
    # Заполняем проекцию (максимальное значение по высоте)
    for z in range(z_levels):
        for y in range(height):
            for x in range(width):
                if map_3d[x, y, z] == 1:  # Препятствие
                    xy_projection[x, y] = -1
                elif region_assignment[x, y, z] != -1 and xy_projection[x, y] != -1:
                    xy_projection[x, y] = region_assignment[x, y, z]
    
    # Рассчитываем метрики для гистограмм
    total_free_cells = np.sum(map_3d == 0)
    fair_share = total_free_cells / num_agents
    region_metrics = []
    
    for agent_id in range(num_agents):
        region_cells = np.sum(region_assignment == agent_id)
        percentage_excess = ((region_cells / fair_share) - 1) * 100
        region_metrics.append({
            'agent_id': agent_id,
            'cells': region_cells,
            'percentage_excess': percentage_excess
        })
    
    # Создаем фигуру с подграфиками (2D вид, 3D вид, гистограмма)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "heatmap"}, {"type": "scene"}],
               [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=('Вид сверху (XY)', '3D визуализация областей', 
                        'Распределение ячеек по секторам (% превышения над равной долей)'),
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Создаем colorscale для регионов и препятствий
    colorscale = [
        [-1.5, 'rgb(0,0,0)'],     # Черный для препятствий (-1)
        [-0.5, 'rgb(0,0,0)'],
        [-0.5, distinct_colors[0]]
    ]
    
    for i in range(num_agents-1):
        colorscale.append([i+0.5, distinct_colors[i]])
        colorscale.append([i+0.5, distinct_colors[i+1]])
    
    colorscale.append([num_agents-0.5, distinct_colors[num_agents-1]])
    
    # 2D вид сверху (XY projection)
# Альтернативный вариант исправления
    fig.add_trace(
        go.Heatmap(
            z=xy_projection.T,
            colorscale='Viridis',  # используем стандартную цветовую шкалу
            showscale=False,
            zmin=-1, 
            zmax=num_agents-1,
            # Для отображения дискретных значений:
            colorbar=dict(
                tickvals=list(range(-1, num_agents)),
                ticktext=['Препятствие'] + [f'Регион {i+1}' for i in range(num_agents)]
            )
        ),
        row=1, col=1
    )
    
    # Добавляем маркеры начальных позиций агентов на 2D вид
    for i, (x, y, z) in enumerate(agent_positions):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='white',
                    line=dict(color='black', width=1)
                ),
                name=f'Агент {i+1} (старт)',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 3D визуализация
    # Отображаем препятствия
    obstacle_coords = np.argwhere(map_3d == 1)
    if len(obstacle_coords) > 0:
        # Ограничиваем количество отображаемых препятствий для производительности
        if len(obstacle_coords) > 1000:
            indices = np.random.choice(len(obstacle_coords), 1000, replace=False)
            obstacle_coords = obstacle_coords[indices]
        
        x_obs, y_obs, z_obs = obstacle_coords[:, 0], obstacle_coords[:, 1], obstacle_coords[:, 2]
        fig.add_trace(
            go.Scatter3d(
                x=x_obs, y=y_obs, z=z_obs,
                mode='markers',
                marker=dict(
                    size=4,
                    color='black',
                    opacity=0.5,
                    symbol='square'
                ),
                name='Препятствия'
            ),
            row=1, col=2
        )
    
    # Отображаем регионы
    for agent_id in range(num_agents):
        region_coords = np.argwhere(region_assignment == agent_id)
        if len(region_coords) > 0:
            # Ограничиваем количество точек для производительности
            if len(region_coords) > 1000:
                indices = np.random.choice(len(region_coords), 1000, replace=False)
                region_coords = region_coords[indices]
            
            x_reg, y_reg, z_reg = region_coords[:, 0], region_coords[:, 1], region_coords[:, 2]
            fig.add_trace(
                go.Scatter3d(
                    x=x_reg, y=y_reg, z=z_reg,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=distinct_colors[agent_id],
                        opacity=0.7
                    ),
                    name=f'Регион {agent_id+1}'
                ),
                row=1, col=2
            )
    
    # Отображаем позиции агентов в 3D
    for i, (x, y, z) in enumerate(agent_positions):
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=10,
                    color='white',
                    symbol='diamond',
                    line=dict(color=distinct_colors[i], width=2)
                ),
                name=f'Агент {i+1} (старт)',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Создаем гистограмму с процентным превышением
    fig.add_trace(
        go.Bar(
            x=[f'Регион {m["agent_id"]+1}' for m in region_metrics],
            y=[m["percentage_excess"] for m in region_metrics],
            marker_color=[distinct_colors[m["agent_id"]] for m in region_metrics],
            text=[f'{m["percentage_excess"]:.1f}%' for m in region_metrics],
            textposition='auto',
            hovertemplate='Регион %{x}<br>Превышение: %{y:.2f}%<br>Ячеек: %{customdata}',
            customdata=[[m["cells"]] for m in region_metrics]
        ),
        row=2, col=1
    )
    
    # Добавляем горизонтальную линию на уровне 0% (равное распределение)
    fig.add_shape(
        type="line",
        x0=-0.5, y0=0, x1=num_agents-0.5, y1=0,
        line=dict(color="red", width=2, dash="dash"),
        row=2, col=1
    )
    
    # Настройка 3D вида
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='X координата',
        yaxis_title='Y координата',
        zaxis_title='Z координата',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    # Настройка 2D вида
    fig.update_xaxes(title_text='X координата', row=1, col=1)
    fig.update_yaxes(title_text='Y координата', row=1, col=1)
    
    # Настройка гистограммы
    fig.update_xaxes(title_text='Регионы', row=2, col=1)
    fig.update_yaxes(title_text='Процент превышения над равной долей (%)', row=2, col=1)
    
    # Обновляем заголовок
    fig.update_layout(
        title_text=f"Разделение областей - {method_name}",
        height=900,
        width=1200,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Сохраняем изображение
    if save_path:
        fig.write_image(save_path, scale=2)
    
    return fig
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors

def visualize_3d_grid(nodes, edges, grid_3d=None, title="3D сетка"):
    """
    Визуализирует 3D сетку с использованием Plotly.
    
    Параметры:
        nodes (numpy.ndarray): Массив узлов сетки
        edges (numpy.ndarray): Массив ребер сетки
        grid_3d (numpy.ndarray): 3D массив препятствий (опционально)
        title (str): Заголовок графика
        
    Возвращает:
        plotly.graph_objects.Figure: Объект фигуры Plotly
    """
    fig = go.Figure()
    
    # Визуализация узлов
    if len(nodes) > 0:
        fig.add_trace(go.Scatter3d(
            x=nodes[:, 0],  # X остается X
            y=nodes[:, 1],  # Y остается Y
            z=nodes[:, 2],  # Z остается Z
            mode='markers',
            marker=dict(
                size=3,
                color='blue',
                opacity=0.8
            ),
            name='Узлы сетки'
        ))
    
    # Визуализация ребер
    if len(edges) > 0:
        x_edges, y_edges, z_edges = [], [], []
        
        for edge in edges:
            i, j = edge
            x_edges.extend([nodes[i, 0], nodes[j, 0], None])
            y_edges.extend([nodes[i, 1], nodes[j, 1], None])
            z_edges.extend([nodes[i, 2], nodes[j, 2], None])
        
        fig.add_trace(go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode='lines',
            line=dict(color='green', width=2),
            name='Рёбра сетки'
        ))
    
    # Визуализация препятствий - исправляем порядок координат
    if grid_3d is not None:
        depth, height, width = grid_3d.shape
        
        # Получаем индексы препятствий
        # ВАЖНО: z, y, x - это порядок осей в нашем grid_3d
        z_coords, y_coords, x_coords = np.where(grid_3d == 1)
        
        # Сопоставляем их правильно с осями отображения
        fig.add_trace(go.Scatter3d(
            # Важно: фактически меняем местами оси, чтобы совпадало с отображением вершин
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=2,
                color='orangered',
                opacity=0.3
            ),
            name='Препятствия'
        ))
    
    # Настройка макета
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # использовать вместо 'cube' для лучшего масштабирования данных
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Настройка начального угла обзора
            )
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

def visualize_grid_layer(nodes, edges, grid_slice, z_layer, title="Слой сетки"):
    """
    Визуализирует один слой 3D сетки.
    
    Параметры:
        nodes (numpy.ndarray): Массив узлов сетки
        edges (numpy.ndarray): Массив ребер сетки
        grid_slice (numpy.ndarray): 2D срез препятствий
        z_layer (int): Номер слоя
        title (str): Заголовок графика
        
    Возвращает:
        plotly.graph_objects.Figure: Объект фигуры Plotly или None
    """
    # Фильтруем узлы и ребра для текущего слоя
    nodes_in_layer = [i for i, node in enumerate(nodes) if node[2] == z_layer]
    
    if not nodes_in_layer:
        print(f"Нет узлов в слое {z_layer}")
        return None
    
    # Фильтруем ребра
    edges_in_layer = []
    for edge in edges:
        i, j = edge
        if i in nodes_in_layer and j in nodes_in_layer:
            edges_in_layer.append((i, j))
    
    fig = go.Figure()
    
    # Визуализация препятствий
    if grid_slice is not None:
        # grid_slice имеет порядок [height, width]
        y_indices, x_indices = np.where(grid_slice == 1)
        
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=y_indices,
            mode='markers',
            marker=dict(
                size=4,
                color='orangered',
                opacity=0.7
            ),
            name='Препятствия'
        ))
    
    # Визуализация узлов
    fig.add_trace(go.Scatter(
        x=[nodes[i, 0] for i in nodes_in_layer],
        y=[nodes[i, 1] for i in nodes_in_layer],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            opacity=0.8
        ),
        name='Узлы сетки'
    ))
    
    # Визуализация ребер
    x_edges, y_edges = [], []
    
    for i, j in edges_in_layer:
        x_edges.extend([nodes[i, 0], nodes[j, 0], None])
        y_edges.extend([nodes[i, 1], nodes[j, 1], None])
    
    if x_edges:
        fig.add_trace(go.Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            line=dict(color='green', width=2),
            name='Рёбра сетки'
        ))
    
    # Настройка макета
    fig.update_layout(
        title=f"{title} (Слой {z_layer})",
        xaxis=dict(
            title='X',
            scaleanchor="y",  # Фиксируем соотношение осей
            scaleratio=1      # Устанавливаем соотношение 1:1
        ),
        yaxis=dict(title='Y'),
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
    
    Параметры:
        fig (plotly.graph_objects.Figure): Объект фигуры Plotly
        filename (str): Имя файла для сохранения (без расширения)
        width (int): Ширина изображения в пикселях
        height (int): Высота изображения в пикселях
    """
    try:
        fig.write_image(f"{filename}.png", width=width, height=height)
        print(f"Изображение сохранено: {filename}.png")
    except Exception as e:
        print(f"Ошибка сохранения изображения: {e}")
        print("Убедитесь, что установлены пакеты kaleido или orca:")
        print("pip install kaleido")

def display_all_visualizations(figures_dict):
    """
    Отображает все визуализации в интерактивном режиме.
    
    Параметры:
        figures_dict (dict): Словарь с объектами фигур Plotly
    """
    for name, fig in figures_dict.items():
        if fig is not None:
            print(f"Отображение: {name}")
            fig.show()
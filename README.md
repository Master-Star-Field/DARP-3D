# DARP-3D

#  Алгоритм генерации трехмерных лабиринтов с переменной высотой стен 

![ Пример лабиринта ](/images/maze_example.png ) 

##  1. Алгоритмическая основа 

###  1.1 Общее описание 
Представленный алгоритм реализует комбинированный подход, сочетающий:
 1.  **Модифицированный алгоритм Крускала**  для построения минимального остовного дерева
 2.  **Рандомизированное добавление циклов**  для увеличения сложности лабиринта
 3.  **Процедурную генерацию высот**  для создания трехмерного эффекта стен

Основная идея заключается в моделировании лабиринта как графа, где:
 -  Ячейки → вершины графа
 -  Стены → потенциальные ребра графа
 -  Удаление стен → добавление ребер в граф

 ###  1.2 Псевдокод алгоритма 

    Инициализировать систему непересекающихся множеств (DSU)

    Сгенерировать все возможные стены между ячейками:
    а) Горизонтальные стены (между столбцами)
    б) Вертикальные стены (между строками)

    Перемешать список стен для рандомизации

    Построить минимальное остовное дерево:
    Для каждой стены в случайном порядке:
    а) Определить смежные ячейки
    б) Если ячейки принадлежат разным множествам → удалить стену
    в) Объединить множества ячеек

    Добавить дополнительные соединения для создания циклов

    Преобразовать логическую структуру в трехмерную карту высот:
    а) Инициализировать матрицу высот
    б) Заполнить базовые стены случайными высотами
    в) Удалить стены, соответствующие остовному дереву
    г) Создать проходы заданной ширины 

## 2. Ключевые структуры данных

### 2.1 Система непересекающихся множеств (DSU)
- **Реализация**: Хеш-таблицы для parent и rank
- **Выбор обоснован**:
  1. O(α(n)) время операций union/find (практически константа)
  2. Эффективное управление связями между ячейками
  3. Позволяет обрабатывать до 10^6 ячеек без потерь производительности

### 2.2 Матрица высот
- **Структура**: Двумерный массив numpy
- **Преимущества**:
  1. Компактное хранение (4 байта на элемент)
  2. Векторизованные операции для быстрого заполнения
  3. Поддержка многомерной индексации

### 2.3 Список стен
- **Организация**: 
  1. Тип стены ('h'/'v')
  2. Координаты ячейки (i,j)
- **Оптимизация**:
  1. Ленивая генерация стен
  2. In-place перемешивание
  3. Множественные операции для быстрого сравнения

## 3. Параметры генерации

### 3.1 Основные параметры
1. `rows, cols` - размерность сетки ячеек
2. `passage_width` - ширина прохода (в пикселях)
3. `wall_width` - толщина стен (в пикселях)
4. `map_height` - максимальная высота стен

### 3.2 Особенности реализации
- **Высоты стен**: Дискретные значения из 10 уровней
- **Распределение высот**: Равномерное случайное
- **Гарантии связности**: Минимальное остовное дерево + дополнительные циклы

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon


def create_hurwitz_matrix(n):
    """
    Создаёт случайную гурвицеву матрицу n x n.
    """
    # Создаём случайную матрицу
    random_matrix = np.random.rand(n, n)
    # Создаём кососимметричную матрицу (собственные значения на мнимой оси)
    skew_symmetric = random_matrix - random_matrix.T
    # Сдвигаем собственные значения влево, чтобы сделать их действительные части отрицательными
    A = skew_symmetric - np.eye(n) * 0.5
    print("Сгенерированная гурвицева матрица A:\n", A)
    print("Собственные значения A:", np.linalg.eigvals(A))
    return A


def plot_corporeal_cone(n_dim):
    """
    Строит и отображает телесный конус для системы размерности n_dim (2 или 3).
    """
    if n_dim not in [2, 3]:
        raise ValueError("Размерность должна быть 2 или 3.")

    # 1. Определение параметров системы
    A = create_hurwitz_matrix(n_dim)
    b = np.random.rand(n_dim) * 2 - 1  # Случайный вектор b
    print("\nВектор b:\n", b)

    # 2. Генерация образующей кривой
    t_max = 15
    num_points = 200
    t_space = np.linspace(0, t_max, num_points)

    # Вычисляем траекторию x(t) = exp(A*t) * b
    trajectory_points = np.array([expm(A * t) @ b for t in t_space])

    # 3. Формирование точек для выпуклой оболочки
    # Добавляем начало координат (вершину конуса)
    origin = np.zeros((1, n_dim))
    points_for_hull = np.vstack([origin, trajectory_points])

    # 4. Вычисление выпуклой оболочки
    hull = ConvexHull(points_for_hull)

    # 5. Визуализация
    if n_dim == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Телесный конус (2D)", fontsize=16)

        # Рисуем траекторию
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'b-', label='Траектория $e^{At}b$')

        # Рисуем вершину конуса
        ax.plot(0, 0, 'ko', label='Начало координат (вершина)')

        # Рисуем выпуклую оболочку (конус)
        for simplex in hull.simplices:
            ax.plot(points_for_hull[simplex, 0], points_for_hull[simplex, 1], 'r-', alpha=0.8)

        # Заливаем конус для наглядности
        hull_polygon = Polygon(points_for_hull[hull.vertices], color='red', alpha=0.15)
        ax.add_patch(hull_polygon)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True)
        ax.legend()
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.axis('equal')

    elif n_dim == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Телесный конус (3D)", fontsize=16)

        # Рисуем траекторию
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2], 'b-',
                label='Траектория $e^{At}b$')

        # Рисуем вершину
        # ax.scatter(0, 0, 0, c='k', marker='o', s=50, label='Начало координат (вершина)')

        # Рисуем поверхность конуса
        # hull.simplices содержит индексы треугольников, образующих поверхность
        ax.plot_trisurf(
            points_for_hull[:, 0], points_for_hull[:, 1], points_for_hull[:, 2],
            triangles=hull.simplices,
            color='red', alpha=0.2, shade=True
        )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()

    plt.show()
    
if __name__ == "__main__":
    plot_corporeal_cone(n_dim=3)
import numpy as np
from scipy.signal import argrelextrema
import warnings
import scipy

import numpy as np

import numpy as np


class ConstrainedTrajectoryAnalyzer:
    def __init__(self, t: np.ndarray, dist_list: list[np.ndarray],
                 x1_list: list[np.ndarray], x2_list: list[np.ndarray]):
        """
        t: одномерный массив времени (размерность K - N)
        dist_list: список массивов расстояний между траекториями
        x1_list, x2_list: списки массивов координат x1(t) и x2(t) для проверки ограничений
        """
        self.t = np.asarray(t)
        self.dist_list = [np.asarray(d) for d in dist_list]
        self.x1_list = [np.asarray(x) for x in x1_list]
        self.x2_list = [np.asarray(x) for x in x2_list]

    def _calculate_constrained_integral(self, idx: int, x_min: float, x_max: float, alpha: float = None) -> float:
        """
        Вычисляет интеграл для траектории под индексом idx только в тех точках,
        где обе координаты x1 и x2 лежат в диапазоне [x_min, x_max].
        """
        t = self.t
        dist = self.dist_list[idx]
        x1 = self.x1_list[idx]
        x2 = self.x2_list[idx]

        # Создаем логическую маску: обе траектории внутри допустимого интервала
        mask = (x1 >= x_min) & (x1 <= x_max) & (x2 >= x_min) & (x2 <= x_max)

        # Фильтруем временную сетку и значения расстояний
        t_masked = t[mask]
        dist_masked = dist[mask]

        # Если валидных точек нет или их недостаточно для интегрирования (меньше 2)
        if len(t_masked) < 2:
            return -float('inf')

        # Если задан alpha, применяем экспоненциальный вес
        if alpha is not None:
            dist_masked = dist_masked * np.exp(-alpha * t_masked)

        # Вычисление интеграла методом трапеций по базовым операциям (для произвольной сетки)
        dt = np.diff(t_masked)
        dist_mean = (dist_masked[:-1] + dist_masked[1:]) / 2.0

        return float(np.sum(dist_mean * dt))

    def find_fastest_growth_integral(self, x_min: float = -1.0, x_max: float = 1.0) -> int:
        best_idx = -1
        max_area = -float('inf')

        for idx in range(len(self.dist_list)):
            area = self._calculate_constrained_integral(idx, x_min, x_max, alpha=None)
            if area > max_area:
                max_area = area
                best_idx = idx

        return best_idx

    def find_fastest_growth_weighted(self, alpha: float = 0.5, x_min: float = -1.0, x_max: float = 1.0) -> int:
        best_idx = -1
        max_score = -float('inf')

        for idx in range(len(self.dist_list)):
            score = self._calculate_constrained_integral(idx, x_min, x_max, alpha=alpha)
            if score > max_score:
                max_score = score
                best_idx = idx

        return best_idx


class TrajectoryAnalyzer:
    def __init__(self, t: np.ndarray, trajectories: list[np.ndarray]):
        """
        t: одномерный массив временных меток (shared для всех траекторий)
        trajectories: список одномерных массивов numpy с координатами x1(t) - x2(t)
        """
        self.t = np.asarray(t)
        self.trajectories = [np.asarray(traj) for traj in trajectories]

    def find_fastest_growth_integral(self) -> int:
        """
        Находит индекс траектории с максимальной площадью под кривой.
        Идеально подходит для гладких функций, так как ранний рост 
        дает больший вклад в интеграл на фиксированном отрезке.
        """
        best_idx = -1
        max_area = -float('inf')

        for idx, traj in enumerate(self.trajectories):
            # Вычисляем интеграл методом трапеций
            area = np.trapezoid(traj, self.t)
            if area > max_area:
                max_area = area
                best_idx = idx

        return best_idx

    def find_fastest_growth_weighted(self, alpha: float = 0.1) -> int:
        """
        Находит индекс траектории с учетом экспоненциального штрафа за время.
        Чем позже происходит рост, тем меньше он влияет на итоговую метрику.
        alpha: коэффициент затухания (чем больше, тем важнее именно начальный рост)
        """
        best_idx = -1
        max_score = -float('inf')
        weights = np.exp(-alpha * self.t)

        for idx, traj in enumerate(self.trajectories):
            # Интегрируем взвешенную функцию
            weighted_traj = traj * weights
            score = np.trapezoid(weighted_traj, self.t)
            if score > max_score:
                max_score = score
                best_idx = idx

        return best_idx


def calc_fourier_transform(sig, N=None, dt=None):
    if N is None:
        N = sig.shape[0] - 1

    if dt is None:
        dt = 1

    if N % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        sig = sig[0:-1]

    sigFFT = scipy.fft.fft(sig)
    freq = scipy.fft.fftfreq(N, dt)[1:N // 2]

    return freq[1:N // 2], 2. / N * np.abs(sigFFT[1:N // 2 - 1])


def gen_steps_sequence(first, n, count):
    result = [first]
    for i in range(1, count):
        first /= n
        result.append(first)
    return result


def find_local_extrema(data):
    idx_minimas = argrelextrema(data, np.less)[0]
    idx_maximas = argrelextrema(data, np.greater)[0]
    idx = np.sort(np.concatenate((idx_minimas, idx_maximas)))
    return data[idx]


def find_local_extrema_idx(data):
    idx_minimas = argrelextrema(data, np.less)[0]
    idx_maximas = argrelextrema(data, np.greater)[0]
    return idx_minimas, idx_maximas


def find_growth_segment(ts):
    """
    Находит максимальный непрерывный отрезок роста в временном ряде.
    Если ряд начинается со спада, ищет первую точку, после которой начинается рост.

    Параметры:
        ts: numpy-массив значений ряда.

    Возвращает:
        Кортеж (start, end, slope), где:
          - start: индекс начала отрезка роста,
          - end: индекс последнего значения отрезка роста,
          - slope: средняя скорость роста, рассчитанная как (ts[end] - ts[start]) / (end - start).
        Если подходящего отрезка не найдено (например, ряд не имеет роста), возвращает None.
    """
    n = len(ts)
    if n < 2:
        return None

    # Находим первую точку, где начинается рост
    start = None
    for i in range(n - 1):
        if ts[i + 1] > ts[i]:
            start = i
            break
    if start is None:
        return None  # нет ни одного положительного прироста

    # Расширяем отрезок до тех пор, пока значения строго растут
    end = start + 1
    while end < n and ts[end] > ts[end - 1]:
        end += 1
    end -= 1  # так как while остановился, когда рост прекратился

    # Если отрезок состоит из одного элемента, его не рассматриваем
    if end == start:
        return None

    growth = ts[end] - ts[start]
    length = end - start
    slope = growth / length

    return start, end, slope, ts[start], ts[end], growth


def select_best_growth(time_series_list):
    """
    Из списка временных рядов выбирает тот, у которого на найденном отрезке роста максимальный средний прирост.

    Параметры:
        time_series_list: список numpy-массивов, каждый из которых представляет временной ряд.

    Возвращает:
        Кортеж (best_idx, best_segment), где:
          - best_idx: индекс временного ряда с наибольшим ростом,
          - best_segment: данные отрезка роста (start, end, slope) для этого ряда.
        Если ни в одном ряде не найден подходящий отрезок, возвращает (None, None).
    """
    best_idx = None
    best_slope = -np.inf
    best_segment = None

    best_start_val, best_end_val, best_growth = None, None, None

    for idx, ts in enumerate(time_series_list):
        segment = find_growth_segment(ts)
        if segment is None:
            continue
        start, end, slope, start_val, end_val, growth = segment
        if slope > best_slope:
            best_slope = slope
            best_idx = idx
            best_segment = (start, end, slope)
            best_start_val, best_end_val, best_growth = start_val, end_val, growth

    return best_idx, best_segment, best_start_val, best_end_val, best_growth

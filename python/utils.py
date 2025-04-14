import numpy as np
from scipy.signal import argrelextrema


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

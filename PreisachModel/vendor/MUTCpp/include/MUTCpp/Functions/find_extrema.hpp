// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace mc
{
    struct ExtremaIndices
    {
        std::vector<Eigen::Index> maxima;
        std::vector<Eigen::Index> minima;
    };

    /**
     * @brief Находит индексы локальных максимумов и минимумов в 1D массиве.
     * Сочетает скорость однопроходного цикла с точностью обработки плато (плоских экстремумов).
     * * @param signal Вектор данных (принимает любой тип, совместимый с Eigen::VectorXd).
     * @return Структура с массивами индексов максимумов и минимумов.
     */
    inline ExtremaIndices find_extrema_1d(const Eigen::Ref<const Eigen::VectorXd> &signal)
    {
        ExtremaIndices result;
        const Eigen::Index n = signal.size();
        if (n <= 2)
            return result;

        // Резервируем память, предполагая, что экстремумы составляют около 10-20% от сигнала.
        result.maxima.reserve(n / 10);
        result.minima.reserve(n / 10);

        Eigen::Index i = 1;
        while (i < n - 1)
        {
            if (signal[i - 1] < signal[i])
            {
                // Идем вверх: потенциальный максимум
                Eigen::Index i_ahead = i + 1;
                while (i_ahead < n - 1 && signal[i_ahead] == signal[i])
                {
                    ++i_ahead;
                }
                if (signal[i_ahead] < signal[i])
                {
                    result.maxima.push_back((i + i_ahead - 1) / 2);
                }
                i = i_ahead;
            }
            else if (signal[i - 1] > signal[i])
            {
                // Идем вниз: потенциальный минимум
                Eigen::Index i_ahead = i + 1;
                while (i_ahead < n - 1 && signal[i_ahead] == signal[i])
                {
                    ++i_ahead;
                }
                if (signal[i_ahead] > signal[i])
                {
                    result.minima.push_back((i + i_ahead - 1) / 2);
                }
                i = i_ahead;
            }
            else
            {
                // Плато, не являющееся экстремумом (ступенька)
                ++i;
            }
        }

        return result;
    }
}

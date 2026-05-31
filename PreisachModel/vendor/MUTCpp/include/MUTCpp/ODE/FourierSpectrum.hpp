// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Ref/Ref.hpp"
#include "DynamicSystem.hpp"
#include "Json/JsonDocument.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

namespace mc
{
    namespace ode
    {
        /**
         * @brief Вычисляет односторонний амплитудный спектр Фурье для выбранной координаты.
         * @param signal Входной сигнал.
         * @param dt Шаг по времени.
         * @param remove_dc Если true, нулевая частота (0 Гц) исключается из результатов (как в Python).
         * @param use_blackman Если true, применяется окно Блэкмана перед БПФ.
         * @return JsonDocument с массивами частот и амплитуд.
         */
        inline json::JsonDocument FourierSpectrum(Eigen::Ref<Eigen::VectorXd> signal, double dt,
                                                  bool remove_dc = true,
                                                  bool use_blackman = false)
        {
            AL_PROFILE_FUNC("chaos::FourierSpectrum");
            
            const int n = static_cast<int>(signal.size());
            
            // Применение окна Блэкмана (w = scipy.signal.windows.blackman(N))
            if (use_blackman)
            {
                for (int i = 0; i < n; ++i)
                {
                    double w = 0.42 - 0.5 * std::cos(2.0 * consts::pi * i / (n - 1)) + 
                               0.08 * std::cos(4.0 * consts::pi * i / (n - 1));
                    signal(i) *= w;
                }
            }

            // Выполняем прямое быстрое преобразование Фурье
            Eigen::FFT<double> fft;
            Eigen::VectorXcd freq_domain(n);
            fft.fwd(freq_domain, signal);
            
            // Вычисляем физические амплитуды и частоты
            // По теореме Котельникова (Найквиста) значимой является только первая половина спектра.
            // Отсечение 0 Гц (как `[1:N // 2]` в Python)
            const int start_idx = remove_dc ? 1 : 0;
            const int half_n = n / 2;
            const int result_size = half_n - start_idx;
            
            Eigen::VectorXd frequencies(result_size);
            Eigen::VectorXd amplitudes(result_size);

            const double sampling_rate = 1.0 / dt; // Частота дискретизации (Гц)

            for (int i = start_idx; i < half_n; ++i)
            {
                const int out_idx = i - start_idx;
                
                // Физическая частота бина: f = i * (f_s / N)
                // Эквивалент: scipy.fft.fftfreq(N, dt)
                frequencies(out_idx) = i * (sampling_rate / n);

                // Нормировка амплитуды. 
                // Модуль комплексного числа делится на N.
                // Для всех частот, кроме постоянной составляющей (i = 0), амплитуда удваивается 
                // из-за сложения мощностей отрицательных и положительных частот.
                // Эквивалент: 2. / N * np.abs(sigFFT)
                double mag = std::abs(freq_domain(i)) / n;
                amplitudes(out_idx) = (i == 0) ? mag : mag * 2.0;
            }

            // 4. Упаковка результатов в JSON
            json::JsonDocument doc({"name", "time", "dt", "var_idx", "frequencies", "amplitudes"});
            doc.AddField("name", "FourierSpectrum");
            doc.AddField("dt", dt);
            doc.AddField("frequencies", frequencies);
            doc.AddField("amplitudes", amplitudes);

            return doc;
        }
    }
}

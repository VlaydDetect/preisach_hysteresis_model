// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <numbers>

#include <Eigen/Dense>

#include "Core/Constants.hpp"

#include "Eigen/utils.hpp"

namespace mc
{
    namespace ode
    {
        struct ZeroOneResult
        {
            double K;
            std::vector<double> K_values;
        };
        
        /**
         * @brief Вычисляет коэффициент корреляции Пирсона между двумя векторами.
         */
        inline double calculate_correlation(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
            if (x.size() != y.size() || x.size() == 0) return 0.0;

            Eigen::VectorXd x_centered = x.array() - x.mean();
            bool a = std::ranges::any_of(y, [](const double v){ return isnan(v); });
            double ys_mean = y.mean();
            Eigen::VectorXd y_centered = y.array() - y.mean();

            double num = x_centered.dot(y_centered);
            double den = std::sqrt(x_centered.squaredNorm() * y_centered.squaredNorm());

            if (den == 0) return 0.0;
            return num / den;
        }
        
        /**
         * @brief Ядро 0-1 теста для одной конкретной частоты c.
         * * @param ts Временной ряд (Time Series).
         * @param window_size Размер окна (обычно N/10).
         * @param c Случайная частота из (0, pi).
         * @return double Значение K-статистики (0..1).
         */
        inline double compute_k_statistic(const Eigen::VectorXd& ts, int window_size, double c) {
            Eigen::Index N = ts.size();
            
            // 1. Вычисление трансляционных переменных p и q
            // Eigen не имеет cumsum, делаем вручную для эффективности
            Eigen::VectorXd p(N);
            Eigen::VectorXd q(N);
            
            double current_p = 0.0;
            double current_q = 0.0;

            // Используем raw data pointer для максимальной скорости в цикле
            const double* ts_data = ts.data();
            double* p_data = p.data();
            double* q_data = q.data();

            for (Eigen::Index n = 0; n < N; ++n) {
                double angle = (n + 1) * c; // n+1 т.к. в формулах обычно j=1..N
                double cos_val = std::cos(angle);
                double sin_val = std::sin(angle);
                
                current_p += ts_data[n] * cos_val;
                current_q += ts_data[n] * sin_val;
                
                p_data[n] = current_p;
                q_data[n] = current_q;
            }

            // 2. Вычисление среднеквадратичного смещения (MSD) M_c(n)
            // M_c(n) = lim (1/N) sum [ (p(j+n) - p(j))^2 + (q(j+n) - q(j))^2 ]
            Eigen::VectorXd M_c(window_size);
            Eigen::VectorXd V_osc(window_size);
            
            // Вектор индексов n (1, 2, ..., window_size) для корреляции
            Eigen::VectorXd n_vec(window_size);

            for (int n = 1; n <= window_size; ++n) {
                // Векторизованное вычисление разностей
                // p(j+n) - p(j) соответствует p.tail - p.head
                Eigen::Index len = N - n;
                
                double sum_sq = (p.tail(len) - p.head(len)).squaredNorm() 
                    + (q.tail(len) - q.head(len)).squaredNorm();
                
                // double sum_sq = 0.0;
                // for (int j = 0; j < len; ++j)
                // {
                //     sum_sq += mc::square(p(j + n) - p(j)) + mc::square(q(j + n) - q(j));
                // }
                
                M_c(n - 1) = sum_sq / static_cast<double>(len); // Усреднение
                // V_osc(n - 1) = mc::square(ts.mean()) * (1 - std::cos(n * c)) / (1 - std::cos(c));
                n_vec(n - 1) = static_cast<double>(n);
            }
            
            // Eigen::VectorXd D_c = M_c.array() - V_osc.array();

            // 3. Вычисление корреляции между n и M_c(n)
            // Для хаоса рост линейный (K ~ 1), для регулярного - ограниченный (K ~ 0)
            return calculate_correlation(n_vec, M_c);
        }
        
        /**
         * @brief Основная функция 0-1 теста. 
         * Вычисляет медианное значение K по нескольким случайным частотам для исключения резонансов.
         * * @param time_series Входной временной ряд.
         * @param window_size Размер окна (n_cut). Рекомендуется N/10.
         * @param stride
         * @param num_samples Количество случайных частот c (по умолчанию 5).
         * @return double Результат теста (ближе к 0 -> регулярность, ближе к 1 -> хаос).
         */
        inline double ZeroOneTest_Fast_New(const Eigen::VectorXd& time_series, uint32_t stride = 1, std::optional<int> window_size = std::nullopt, int num_samples = 100)
        {
            Eigen::Index raw_size = time_series.size();
            Eigen::Index N = raw_size / stride;
            
            if (!window_size.has_value())
            {
                window_size = N / 10;
            }
            int n_cut = *window_size;
            
            if (N < 1000) {
                std::cerr << "Warning: After downsampling, N=" << N << ". This might be too short for reliable results.\n";
            }
            
            Eigen::VectorXd phi(N);
            // 1. Прореживание (Downsampling) для устранения оверсэмплинга 
            // Берем каждую stride-ую точку.
            for (Eigen::Index i = 0; i < N; ++i) {
                phi[i] = time_series[i * stride];
            }
            
            double E_phi = phi.mean();
            double E_phi_sq = E_phi * E_phi;
            // Вектор индексов n = 1..n_cut для вычисления корреляции
            Eigen::VectorXd n_vec = Eigen::VectorXd::LinSpaced(n_cut, 1.0, n_cut);
            
            // Генератор случайных частот c in (pi/5, 4pi/5) чтобы избежать резонансов.
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(std::numbers::pi / 5.0, 4.0 * std::numbers::pi / 5.0);
            
            std::vector<double> k_values;
            k_values.reserve(num_samples);
            
            Eigen::VectorXd p(N);
            Eigen::VectorXd q(N);
            
            double current_p = 0.0;
            double current_q = 0.0;
            const double* phi_data = phi.data();
            
            Eigen::VectorXd Dc(n_cut);

            for (int i = 0; i < num_samples; ++i) {
                double c = dist(gen);
                
                for (Eigen::Index j = 0; j < N; ++j) {
                    // j+1 используется, т.к. в статье сумма от j=1..n
                    double angle = (j + 1) * c; 
                    double cos_val = std::cos(angle);
                    double sin_val = std::sin(angle);
                
                    current_p += phi_data[j] * cos_val;
                    current_q += phi_data[j] * sin_val;
                
                    p[j] = current_p;
                    q[j] = current_q;
                }
                
                // Терм для коррекции осцилляций: (1 - cos(c)) в знаменателе
                double denom_osc = 1.0 - std::cos(c);
                
                for (int n = 1; n <= n_cut; ++n) {
                    Eigen::Index len = N - n;
                
                    // Векторизованное MSD (обычное M_c)
                    // M_c(n) = mean( (p(j+n)-p(j))^2 + ... )
                    double sum_sq = (p.tail(len) - p.head(len)).squaredNorm() + 
                                    (q.tail(len) - q.head(len)).squaredNorm();
                    double Mc = sum_sq / static_cast<double>(len);

                    // Вычисляем осциллирующую поправку V_osc(c, n) [cite: 77]
                    // V_osc = (E_phi)^2 * (1 - cos(nc)) / (1 - cos(c))
                    double V_osc = E_phi_sq * (1.0 - std::cos(n * c)) / denom_osc;

                    // Сохраняем модифицированное смещение
                    Dc(n - 1) = Mc - V_osc;
                }
                
                double corr = calculate_correlation(n_vec, Dc);
                k_values.push_back(corr);
            }

            // Возвращаем медиану
            std::ranges::sort(k_values);
            if (num_samples % 2 == 0) {
                return std::abs((k_values[num_samples / 2 - 1] + k_values[num_samples / 2]) / 2.0);
            } else {
                return std::abs(k_values[num_samples / 2]);
            }
        }
    }
}

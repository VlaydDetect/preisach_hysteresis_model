// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/find_extrema.hpp"
#include "DynamicSystem.hpp"
#include <format>
#include <span>
#include <string>
#include <vector>

namespace mc
{
    namespace ode
    {
        struct BifurcationData
        {
            std::vector<double> max_params;
            std::vector<double> max_values;
            std::vector<double> min_params;
            std::vector<double> min_values;
        };

        struct BifurcationResult
        {
            std::string parameter_name;
            std::vector<double> parameter_values;
            BifurcationData extrema;
        };

        namespace detail
        {
            // Используем RowMajor для обеспечения кэш-дружественного последовательного доступа к памяти
            using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            inline BifurcationData extract_bifurcation_extrema(
                std::span<const double> bifurcation_params,
                const MatrixRowMajor &time_series_matrix)
            {
                const Eigen::Index num_params = time_series_matrix.rows();
                const Eigen::Index time_steps = time_series_matrix.cols();

                if (bifurcation_params.size() != num_params)
                {
                    throw std::invalid_argument(
                        std::format("Несоответствие размерностей: параметры ({}), матрица ({})",
                                    bifurcation_params.size(), num_params)
                        );
                }

                BifurcationData result;

                // Резервируем память, предполагая, что экстремумы составляют около 20% от временного ряда.
                // Это предотвращает лишние аллокации при добавлении элементов.
                const std::size_t estimated_capacity = static_cast<std::size_t>(num_params * time_steps * 0.1);
                result.max_params.reserve(estimated_capacity);
                result.max_values.reserve(estimated_capacity);
                result.min_params.reserve(estimated_capacity);
                result.min_values.reserve(estimated_capacity);

                // Используем унифицированный безопасный метод для каждой строки
                for (Eigen::Index i = 0; i < num_params; ++i)
                {
                    const double current_param = bifurcation_params[i];

                    // Передаем строку как Eigen::Ref (без копирования)
                    auto [maxima, minima] = mc::find_extrema_1d(time_series_matrix.row(i));

                    for (const Eigen::Index idx : maxima)
                    {
                        result.max_params.push_back(current_param);
                        result.max_values.push_back(time_series_matrix(i, idx));
                    }

                    for (const Eigen::Index idx : minima)
                    {
                        result.min_params.push_back(current_param);
                        result.min_values.push_back(time_series_matrix(i, idx));
                    }
                }

                // Освобождаем неиспользуемую зарезервированную память
                result.max_params.shrink_to_fit();
                result.max_values.shrink_to_fit();
                result.min_params.shrink_to_fit();
                result.min_values.shrink_to_fit();

                return result;
            }
        }

        template <int N, typename Context, typename SetParameter>
        BifurcationResult BifurcationDiagram(DynamicalSystem<N, Context> &system,
                                              const std::string &param_name, std::span<const double> param_values,
                                              SetParameter set_parameter, double researchTime,
                                              double timeForward, uint32_t var_idx = 0)
        {
            system.ResetArgs();
            system.Reset();

            detail::MatrixRowMajor trajs =
                detail::MatrixRowMajor::Zero(param_values.size(),
                                             static_cast<Eigen::Index>(researchTime / system.GetDeltaTime()) + 1);

            for (std::size_t i = 0; i < param_values.size(); ++i)
            {
                system.Reset();
                auto args = system.GetArgs();
                std::invoke(set_parameter, args, param_values[i]);
                system.SetArgs(args);

                system.Forward(timeForward);

                const auto traj = system.Forward(researchTime);
                trajs.row(i) = traj.col(var_idx).transpose();
            }

            auto extrema = detail::extract_bifurcation_extrema(param_values, trajs);

            system.ResetArgs();
            system.Reset();
            
            return {param_name, {param_values.begin(), param_values.end()}, std::move(extrema)};
        }
    }
}

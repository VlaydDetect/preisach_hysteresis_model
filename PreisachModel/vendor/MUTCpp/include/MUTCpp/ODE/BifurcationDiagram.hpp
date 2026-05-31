// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/find_extrema.hpp"
#include "DSArgs.hpp"
#include "Json/JsonDocument.hpp"

namespace mc
{
    namespace ode
    {
        namespace detail
        {
            struct BifurcationData
            {
                std::vector<double> max_params;
                std::vector<double> max_values;

                std::vector<double> min_params;
                std::vector<double> min_values;
            };

            // Используем RowMajor для обеспечения кэш-дружественного последовательного доступа к памяти
            using MatrixRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            inline BifurcationData extract_bifurcation_extrema(
                const std::vector<Vote> &bifurcation_params,
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
                    const double current_param = bifurcation_params[i].toDouble();

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

        inline json::JsonDocument BifurcationDiagram(Ref<DynamicalSystem> system,
                                                     const std::pair<std::string, std::vector<Vote>> &bifurcationParam,
                                                     double researchTime,
                                                     double timeForward,
                                                     uint32_t var_idx = 0)
        {
            system->ResetArgs();
            system->Reset();

            const auto &[param_name, param_value] = bifurcationParam;

            json::JsonDocument message({"name", param_name, "max_params", "max_values", "min_params", "min_values"});
            message.AddField("name", "BifurcationDiagram");

            WriteVotesToDoc(message, param_name, param_value, param_value[0].Type());

            Eigen::MatrixXd trajs =
                detail::MatrixRowMajor::Zero(param_value.size(),
                                             static_cast<Eigen::Index>(researchTime / system->GetDeltaTime()) + 1);

            for (int i = 0; i < param_value.size(); ++i)
            {
                system->Reset();
                system->SetArg(param_name, param_value[i]);

                system->Forward(timeForward);

                const auto traj = system->Forward(researchTime);
                trajs.row(i) = traj.col(var_idx);
            }

            const auto biff_data = detail::extract_bifurcation_extrema(param_value, trajs);
            message.AddField("max_params", biff_data.max_params);
            message.AddField("max_values", biff_data.max_values);
            message.AddField("min_params", biff_data.min_params);
            message.AddField("min_values", biff_data.min_values);

            // message.AddField("trajs", trajs);

            system->ResetArgs();
            system->Reset();
            
            return message;
        }
    }
}

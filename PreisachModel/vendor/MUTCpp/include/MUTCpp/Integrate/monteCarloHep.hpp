// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Random/rand.hpp"
#include "Utils/gauss_area.hpp"
#include "hep/mc.hpp"

#include <Eigen/Dense>

namespace mc
{
    inline void display_progress(double progress,
                                 double error_estimate,
                                 double current_estimate,
                                 std::chrono::duration<double> estimated_time_to_completion)
    {
        int barWidth = 70;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] "
            << static_cast<int>(progress * 100.0)
            << "%, E = "
            << std::setprecision(3)
            << error_estimate
            << ", time to completion: "
            << estimated_time_to_completion.count()
            << " seconds, estimate: "
            << std::setprecision(5)
            << current_estimate
            << "     \r";

        // std::cout.flush();
    }

    namespace integrate
    {
        namespace detail
        {
            inline Eigen::Vector2d mapToTriangle(const Eigen::Vector2d &p,
                                                 const Eigen::Vector2d &v1,
                                                 const Eigen::Vector2d &v2,
                                                 const Eigen::Vector2d &v3)
            {
                AL_PROFILE_FUNC("mc::integrate::detail::mapToTriangle");

                Eigen::Matrix2d M;
                M.col(0) = v2 - v1;
                M.col(1) = v3 - v1;

                return M * p + v1;
            }
        }

        inline double monteCarloTriangle(const std::function<double(double, double)>& f, const Eigen::Vector2d& v1,
                                         const Eigen::Vector2d& v2, const Eigen::Vector2d& v3)
        {
            AL_PROFILE_FUNC("mc::integrate::monteCarloTriangle");
            auto fn = [v1, v2, v3, f](const std::vector<double> &point) -> double
            {
                const double x = point[0];
                const double y = point[1];

                auto p = detail::mapToTriangle({x, y}, v1, v2, v3);

                const double value = f(p[0], p[1]);

                return value;
            };
            std::vector<std::pair<double, double>> bounds{{0., 1.}, {0., 1.}};
            double error_goal = 0.005;
            naive_monte_carlo mc(fn, bounds, error_goal, false);
            std::future<double> task = mc.integrate();

            while (task.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
            {
                display_progress(mc.progress(),
                                 mc.current_error_estimate(),
                                 mc.current_estimate(),
                                 mc.estimated_time_to_completion());
            }

            double result = task.get();
            const double area = gauss_area({v2, v3}, v1).total_area;
            return result * area;
        }

        inline double monteCarloTriangle(const std::function<double(double, double)> &f, const std::array<Eigen::Vector2d, 3>& triangle)
        {
            return monteCarloTriangle(f, triangle[0], triangle[1], triangle[2]);
        }
    }
}

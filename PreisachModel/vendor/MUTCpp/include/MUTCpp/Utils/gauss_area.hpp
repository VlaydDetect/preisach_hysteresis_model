// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>
#include "Core/Types.hpp"
#include "Functions/abs.hpp"

#include <Eigen/Dense>

namespace mc
{
    struct GaussAreaResult
    {
        double total_area;
        std::vector<std::tuple<double, std::array<Eigen::Vector2d, 3>>> areas_n_triangles;
    };

    inline GaussAreaResult gauss_area(const std::vector<Eigen::Vector2d> &points,
                                      const Eigen::Vector2d &pivot = Eigen::Vector2d::Zero())
    {
        AL_PROFILE_FUNC("mc::utils::gauss_area");

        const uint32 n = points.size();
        double s = 0.;
        std::vector<std::tuple<double, std::array<Eigen::Vector2d, 3>>> areas_n_triangles;

        for (uint32 i = 0; i < n - 1; i++)
        {
            const Eigen::Vector2d &p1 = points[i];
            const Eigen::Vector2d &p2 = points[i + 1];

            std::array triangle = {pivot, p1, p2};
            Eigen::Matrix2d M;
            M.col(0) = pivot - p1;
            M.col(1) = pivot - p2;
            double area = abs(M.determinant()) / 2.;

            s += area;
            areas_n_triangles.push_back({area, triangle});
        }

        return GaussAreaResult(s, areas_n_triangles);
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>
#include "Vector/Vec2.hpp"
#include "Core/Types.hpp"
#include "Functions/abs.hpp"

namespace mc
{
    struct GaussAreaResult
    {
        double total_area;
        std::vector<std::tuple<double, std::array<Vec2, 3>>> areas_n_triangles;
    };

    inline GaussAreaResult gauss_area(const std::vector<Vec2> &points, const Vec2 &pivot = {0.0, 0.0})
    {
        AL_PROFILE_FUNC("mc::utils::gauss_area");
        
        const uint32 n = points.size();
        double s = 0.;
        std::vector<std::tuple<double, std::array<Vec2, 3>>> areas_n_triangles;

        for (uint32 i = 0; i < n-1; i++)
        {
            const Vec2 &p1 = points[i];
            const Vec2 &p2 = points[i + 1];

            std::array<Vec2, 3> triangle = {pivot, p1, p2};
            double area = (pivot.x - p1.x) * (pivot.y - p2.y) - (pivot.y - p1.y) * (pivot.x - p2.x);
            area = abs(area) / 2.;

            s += area;
            areas_n_triangles.push_back({area, triangle});
        }
        
        return GaussAreaResult(s, areas_n_triangles);
    }
}

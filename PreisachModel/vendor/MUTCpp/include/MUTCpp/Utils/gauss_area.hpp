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
        double total_area = 0.0;
        std::vector<std::tuple<double, std::array<Eigen::Vector2d, 3>>> areas_n_triangles;
    };

    inline GaussAreaResult gauss_area(const std::vector<Eigen::Vector2d> &points, const Eigen::Vector2d &pivot)
    {
        AL_PROFILE_FUNC("mc::utils::gauss_area");
        
        GaussAreaResult result;
        const size_t n = points.size();
            
        if (n < 2) return result;
        
        result.areas_n_triangles.reserve(n - 1);
        
        // 2. СОВРЕМЕННЫЙ C++23: std::views::adjacent<2> элегантно выдает пары (p1, p2)
        // по ссылкам, исключая ручное управление индексами массива
        for (const auto& [p1, p2] : points | std::views::adjacent<2>)
        {
            // 3. ОПТИМИЗАЦИЯ МАТЕМАТИКИ: Вместо инстанцирования Eigen::Matrix2d и вызова .determinant(),
            // мы напрямую вычисляем 2D псевдо-скалярное произведение (косое произведение).
            const double cross_product = (p1.x() - pivot.x()) * (p2.y() - pivot.y()) - 
                                         (p1.y() - pivot.y()) * (p2.x() - pivot.x());
            
            const double area = std::abs(cross_product) * 0.5;
            result.total_area += area;
            // 4. Используем emplace_back для создания tuple и array in-place
            result.areas_n_triangles.emplace_back(area, std::array{pivot, p1, p2});
        }

        // NRVO (Named Return Value Optimization), копирования структуры не будет
        return result;
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/linspace.hpp"
#include "Utils/DenseOperations.hpp"
#include <limits>
#include <utility>

namespace mc
{
    template <typename Derived, typename Edges>
    Eigen::RowVectorX<uint32> histogram(const Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Edges> &binEdges)
    {
        if (binEdges.size() < 2)
        {
            throw std::invalid_argument("At least two bin edges are required");
        }
        Eigen::RowVectorXd edges = detail::rowMajor(binEdges).transpose().template cast<double>();
        std::sort(edges.begin(), edges.end());
        Eigen::RowVectorX<uint32> counts = Eigen::RowVectorX<uint32>::Zero(edges.size() - 1);
        for (const auto value : detail::rowMajor(input))
        {
            if (value >= edges(0) && value <= edges(edges.size() - 1))
            {
                const auto upper = std::upper_bound(edges.begin(), edges.end(), value);
                const auto bin = (std::min)(std::distance(edges.begin(), upper) - 1, counts.size() - 1);
                ++counts(bin);
            }
        }
        return counts;
    }

    template <typename Derived>
    auto histogram(const Eigen::DenseBase<Derived> &input, uint32 count = 10)
    {
        if (count == 0 || count == (std::numeric_limits<uint32>::max)() || input.size() == 0)
        {
            throw std::invalid_argument("Histogram requires data and a positive bin count");
        }
        auto edges = linspace(static_cast<double>(input.minCoeff()), static_cast<double>(input.maxCoeff()), count + 1);
        auto counts = histogram(input, edges);
        return std::pair(std::move(counts), std::move(edges));
    }
}

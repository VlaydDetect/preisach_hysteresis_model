// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace mc
{
    template <typename Time, typename Values>
    std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd> ContinueFunction(
        const Eigen::DenseBase<Time> &time, const Eigen::DenseBase<Values> &values, double dt)
    {
        if (time.size() != values.size() || !(dt > 0.0) || !std::isfinite(dt))
        {
            throw std::invalid_argument("Time and values must have equal sizes and dt must be positive and finite");
        }
        const auto t = detail::rowMajor(time);
        const auto v = detail::rowMajor(values);
        std::vector<double> interpolatedTime;
        std::vector<double> interpolatedValues;
        for (Eigen::Index i = 1; i < t.size(); ++i)
        {
            const double start = static_cast<double>(t(i - 1));
            const double span = std::abs(static_cast<double>(t(i)) - start);
            const double count = span / dt;
            if (!std::isfinite(count) || count >= static_cast<double>((std::numeric_limits<Eigen::Index>::max)()))
            {
                throw std::invalid_argument("Too many interpolation samples");
            }
            const double first = static_cast<double>(v(i - 1));
            const double change = static_cast<double>(v(i)) - first;
            for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(count); ++j)
            {
                const double elapsed = static_cast<double>(j) * dt;
                interpolatedTime.push_back(start + elapsed);
                interpolatedValues.push_back(first + elapsed / span * change);
            }
        }
        return {Eigen::Map<const Eigen::RowVectorXd>(interpolatedTime.data(), static_cast<Eigen::Index>(interpolatedTime.size())),
                Eigen::Map<const Eigen::RowVectorXd>(interpolatedValues.data(), static_cast<Eigen::Index>(interpolatedValues.size()))};
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/where.hpp"
#include <vector>

namespace mc
{
    template <typename Mask, typename Choice>
    Eigen::MatrixX<typename Choice::Scalar> select(const std::vector<const Mask *> &conditions,
        const std::vector<const Choice *> &choices, typename Choice::Scalar defaultValue = {})
    {
        if (choices.empty() || choices.size() != conditions.size() || !conditions.front())
        {
            throw std::invalid_argument("Select requires the same nonzero number of conditions and choices");
        }
        using T = typename Choice::Scalar;
        Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Constant(conditions.front()->rows(), conditions.front()->cols(), defaultValue);
        for (std::size_t i = choices.size(); i-- > 0;)
        {
            if (!conditions[i] || !choices[i]) throw std::invalid_argument("Select received a null condition or choice");
            result = where(*conditions[i], choices[i]->matrix(), result);
        }
        return result;
    }

    template <typename Mask, typename Choice>
    auto select(const std::vector<Mask> &conditions, const std::vector<Choice> &choices,
        typename Choice::Scalar defaultValue = {})
    {
        std::vector<const Mask *> masks;
        std::vector<const Choice *> values;
        for (const auto &condition : conditions) masks.push_back(&condition);
        for (const auto &choice : choices) values.push_back(&choice);
        return select(masks, values, defaultValue);
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/find.hpp"

namespace mc
{
    template <typename Derived, typename Mask, typename Values>
    void place(Eigen::DenseBase<Derived> &input, const Eigen::DenseBase<Mask> &mask, const Eigen::DenseBase<Values> &values)
    {
        if (mask.size() != input.size()) throw std::invalid_argument("Input and mask sizes must match");
        if (values.size() == 0) return;
        const auto selected = find(mask);
        const Eigen::VectorX<typename Derived::Scalar> replacements = detail::rowMajor(values);
        auto data = detail::rowMajor(input);
        for (Eigen::Index i = 0; i < selected.size(); ++i)
        {
            data(selected(i)) = replacements(i % replacements.size());
        }
    }
}

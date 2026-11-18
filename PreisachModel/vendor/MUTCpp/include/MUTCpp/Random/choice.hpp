// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Random/permutation.hpp"
#include "Random/randInt.hpp"

namespace mc::random
{
    template <typename Derived>
    typename Derived::Scalar choice(const Eigen::DenseBase<Derived> &input)
    {
        if (input.size() == 0) throw std::invalid_argument("Cannot sample an empty array");
        const auto index = randInt<Eigen::Index>(0, input.size());
        return mc::detail::rowMajor(input)(index);
    }

    template <typename Derived>
    Eigen::RowVectorX<typename Derived::Scalar> choice(const Eigen::DenseBase<Derived> &input, Eigen::Index count, Replace replace = Replace::YES)
    {
        if (count < 0 || (replace == Replace::NO && count > input.size()))
        {
            throw std::invalid_argument("Invalid sample count");
        }
        if (replace == Replace::NO)
        {
            const auto shuffled = permutation(input);
            return mc::detail::rowMajor(shuffled).head(count).transpose();
        }
        Eigen::RowVectorX<typename Derived::Scalar> result(count);
        for (Eigen::Index i = 0; i < count; ++i) result(i) = choice(input);
        return result;
    }
}

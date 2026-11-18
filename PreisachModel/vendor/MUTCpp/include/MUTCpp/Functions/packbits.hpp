// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <type_traits>

#include <Eigen/Core>

#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/Types.hpp"

namespace mc
{
    namespace detail
    {
        template <typename Derived>
        Eigen::MatrixX<uint8> packBitRows(const Eigen::DenseBase<Derived> &input)
        {
            const Eigen::Index bytes = input.cols() / 8 + (input.cols() % 8 != 0);
            auto result = Eigen::MatrixX<uint8>::Zero(input.rows(), bytes).eval();
            for (Eigen::Index row = 0; row < input.rows(); ++row)
            {
                for (Eigen::Index col = 0; col < input.cols(); ++col)
                {
                    const auto bit = static_cast<uint8>(input(row, col)) != 0;
                    result(row, col / 8) |= static_cast<uint8>(bit << (col % 8));
                }
            }
            return result;
        }
    }

    /// Pack truth values into bytes starting at the least significant bit.
    template <typename Derived>
        requires std::is_integral_v<typename Derived::Scalar>
    Eigen::MatrixX<uint8> packbits(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
        case Axis::NONE:
            return detail::packBitRows(input.derived().template reshaped<Eigen::RowMajor>().transpose());
        case Axis::COL:
            return detail::packBitRows(input);
        case Axis::ROW:
            return detail::packBitRows(input.derived().transpose()).transpose();
        }
        THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
    }
}

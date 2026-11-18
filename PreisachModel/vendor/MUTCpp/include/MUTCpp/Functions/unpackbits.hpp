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
        Eigen::MatrixX<uint8> unpackBitRows(const Eigen::DenseBase<Derived> &input)
        {
            Eigen::MatrixX<uint8> result(input.rows(), input.cols() * 8);
            for (Eigen::Index row = 0; row < input.rows(); ++row)
            {
                for (Eigen::Index col = 0; col < result.cols(); ++col)
                {
                    result(row, col) = static_cast<uint8>((input(row, col / 8) >> (col % 8)) & 1);
                }
            }
            return result;
        }
    }

    /// Unpack each byte starting at its least significant bit.
    template <typename Derived>
        requires std::is_same_v<typename Derived::Scalar, uint8>
    Eigen::MatrixX<uint8> unpackbits(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
        case Axis::NONE:
            return detail::unpackBitRows(input.derived().template reshaped<Eigen::RowMajor>().transpose());
        case Axis::COL:
            return detail::unpackBitRows(input);
        case Axis::ROW:
            return detail::unpackBitRows(input.derived().transpose()).transpose();
        }
        THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
    }
}

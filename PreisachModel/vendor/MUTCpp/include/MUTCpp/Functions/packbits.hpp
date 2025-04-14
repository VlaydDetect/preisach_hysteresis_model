// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <type_traits>

#include "Core/Error.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Packs the elements of a binary-valued array into bits in an uint8 array. Works on LittleEndian machines.
    /// @param a: An array of integers or booleans whose elements should be packed to bits.
    /// @param axis: The dimension over which bit-packing is done. None implies packing the flattened array.
    /// @return Matrix<uint8>
    template <typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_same_v<dtype, bool>, int>  = 0>
    Matrix<uint8> packbits(const Matrix<dtype> &a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
        case Axis::NONE:
        {
            const auto numFullValues = a.size() / 8;
            const auto leftOvers = a.size() % 8;
            const auto resultSize = leftOvers == 0 ? numFullValues : numFullValues + 1;

            Matrix<uint8> result(1, resultSize);
            result.fill(0);

            for (typename Matrix<dtype>::size_type i = 0; i < numFullValues; ++i)
            {
                const auto startIdx = i * 8;
                for (auto bit = 0; bit < 8; ++bit)
                {
                    auto value = static_cast<uint8>(a[startIdx + bit]);
                    value = value == 0 ? 0 : 1;
                    result[i] |= (value << bit);
                }
            }

            if (leftOvers != 0)
            {
                const auto startIdx = numFullValues * 8;
                for (std::remove_const_t<decltype(leftOvers)> bit = 0; bit < leftOvers; ++bit)
                {
                    auto value = static_cast<uint8>(a[startIdx + bit]);
                    value = value == 0 ? 0 : 1;
                    result.back() |= (value << bit);
                }
            }

            return result;
        }
        case Axis::COL:
        {
            const auto aShape = a.shape();
            const auto numFullValues = aShape.cols / 8;
            const auto leftOvers = aShape.cols % 8;
            const auto resultSize = leftOvers == 0 ? numFullValues : numFullValues + 1;

            Matrix<uint8> result(aShape.rows, resultSize);
            const auto resultCSlice = result.cSlice();
            const auto aCSlice = a.cSlice();

            for (typename Matrix<dtype>::size_type row = 0; row < aShape.rows; ++row)
            {
                result.put(row, resultCSlice, packbits(a(row, aCSlice)));
            }

            return result;
        }
        case Axis::ROW:
        {
            return packbits(a.transpose(), Axis::COL).transpose();
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}

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
    /// Unpacks elements of a uint8 array into a binary-valued output array. Works on LittleEndian machines.
    ///
    /// Each element of a represents a bit - field that should be unpacked into a binary -
    /// valued output array.The shape of the output array is either 1 - D(if axis is None) or
    /// the same shape as the input array with unpacking done along the axis specified.
    /// @param a: An array of uint8 whose elements should be unpacked to bits.
    /// @param axis: The dimension over which bit-unpacking is done. None implies unpacking the flattened array.
    /// @return Matrix<uint8>
    inline Matrix<uint8> unpackbits(const Matrix<uint8> &a, Axis axis = Axis::NONE)
    {
        switch (axis)
        {
        case Axis::NONE:
        {
            Matrix<uint8> result(1, a.size() * 8);

            for (Matrix<uint8>::size_type byte = 0; byte < a.size(); ++byte)
            {
                const auto startIdx = byte * 8;
                const auto byteValue = a[byte];

                for (uint8 bit = 0; bit < 8; ++bit)
                {
                    result[startIdx + bit] = static_cast<uint8>((byteValue & (uint8{1} << bit)) >> bit);
                }
            }

            return result;
        }
        case Axis::COL:
        {
            const auto aShape = a.shape();
            Matrix<uint8> result(aShape.m_Rows, aShape.m_Cols * 8);
            const auto resultCSlice = result.cSlice();
            const auto aCSlice = a.cSlice();

            for (Matrix<uint8>::size_type row = 0; row < aShape.m_Rows; ++row)
            {
                result.put(row, resultCSlice, unpackbits(a(row, aCSlice)));
            }

            return result;
        }
        case Axis::ROW:
        {
            return unpackbits(a.transpose(), Axis::COL).transpose();
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}

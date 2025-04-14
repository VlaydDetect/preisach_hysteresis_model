// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Trim the leading and/or trailing zeros from a 1-D array or sequence.
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.trim_zeros.html
    /// @param inArray
    /// @param inTrim: ("f" = front, "b" = back, "fb" = front and back)
    template <typename dtype>
    Matrix<dtype> trim_zeros(const Matrix<dtype> &inArray, const std::string &inTrim = "fb")
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inTrim == "f")
        {
            uint32 place = 0;
            for (auto value : inArray)
            {
                if (!utils::essentiallyEqual(value, dtype{0}))
                {
                    break;
                }

                ++place;
            }

            if (place == inArray.size())
            {
                return Matrix<dtype>(0);
            }

            Matrix<dtype> returnArray(1, inArray.size() - place);
            algo::copy(inArray.cbegin() + place, inArray.cend(), returnArray.begin());

            return returnArray;
        }

        if (inTrim == "b")
        {
            uint32 place = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (!utils::essentiallyEqual(inArray[i], dtype{0}))
                {
                    break;
                }

                --place;
            }

            if (place == 0 || (place == 1 && utils::essentiallyEqual(inArray[0], dtype{0})))
            {
                return Matrix<dtype>(0);
            }

            Matrix<dtype> returnArray(1, place);
            algo::copy(inArray.cbegin(), inArray.cbegin() + place, returnArray.begin());

            return returnArray;
        }

        if (inTrim == "fb")
        {
            uint32 placeBegin = 0;
            for (auto value : inArray)
            {
                if (!utils::essentiallyEqual(value, dtype{0}))
                {
                    break;
                }

                ++placeBegin;
            }

            if (placeBegin == inArray.size())
            {
                return Matrix<dtype>(0);
            }

            uint32 placeEnd = inArray.size();
            for (uint32 i = inArray.size() - 1; i > 0; --i)
            {
                if (!utils::essentiallyEqual(inArray[i], dtype{0}))
                {
                    break;
                }

                --placeEnd;
            }

            if (placeEnd == 0 || (placeEnd == 1 && utils::essentiallyEqual(inArray[0], dtype{0})))
            {
                return Matrix<dtype>(0);
            }

            Matrix<dtype> returnArray(1, placeEnd - placeBegin);
            algo::copy(inArray.cbegin() + placeBegin, inArray.cbegin() + placeEnd, returnArray.begin());

            return returnArray;
        }

        THROW_INVALID_ARGUMENT_ERROR("trim options are 'f' = front, 'b' = back, 'fb' = front and back.");
        return {};
    }
}

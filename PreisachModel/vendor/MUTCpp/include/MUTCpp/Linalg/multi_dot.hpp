// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <initializer_list>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Functions/dot.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        /// Compute the dot product of two or more arrays in a single
        /// function call.
        /// @param inList: list of arrays
        /// @return Matrix
        template <typename dtype>
        Matrix<dtype> multi_dot(const std::initializer_list<Matrix<dtype>> &inList)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            typename std::initializer_list<Matrix<dtype>>::iterator iter = inList.begin();

            if (inList.size() == 0)
            {
                THROW_INVALID_ARGUMENT_ERROR("input empty list of arrays.");
            }
            else if (inList.size() == 1)
            {
                return iter->copy();
            }

            Matrix<dtype> returnArray = dot<dtype>(*iter, *(iter + 1));
            iter += 2;
            for (; iter < inList.end(); ++iter)
            {
                returnArray = dot(returnArray, *iter);
            }

            return returnArray;
        }
    }
}

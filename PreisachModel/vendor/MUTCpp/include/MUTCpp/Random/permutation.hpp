// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>

#include "Core/StaticAssert.hpp"
#include "Functions/arange.hpp"
#include "Matrix.hpp"
#include "Random/generator.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> permutation(GeneratorType &generator, dtype inValue)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                Matrix<dtype> returnArray = arange(inValue);
                std::shuffle(returnArray.begin(), returnArray.end(), generator);
                return returnArray;
            }

            /// Randomly permute a sequence, or return a permuted range.
            /// If x is an integer, randomly permute np.arange(x).
            /// If x is an array, make a copy and shuffle the elements randomly.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> permutation(GeneratorType &generator, const Matrix<dtype> &inArray)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                Matrix<dtype> returnArray(inArray);
                std::shuffle(returnArray.begin(), returnArray.end(), generator);
                return returnArray;
            }
        }

        /// Randomly permute a sequence, or return a permuted range.
        /// If x is an integer, randomly permute np.arange(x).
        /// If x is an array, make a copy and shuffle the elements randomly.
        template <typename dtype>
        Matrix<dtype> permutation(dtype inValue)
        {
            return detail::permutation(generator_, inValue);
        }

        /// Randomly permute a sequence, or return a permuted range.
        /// If x is an integer, randomly permute np.arange(x).
        /// If x is an array, make a copy and shuffle the elements randomly.
        template <typename dtype>
        Matrix<dtype> permutation(const Matrix<dtype> &inArray)
        {
            return detail::permutation(generator_, inArray);
        }
    }
}

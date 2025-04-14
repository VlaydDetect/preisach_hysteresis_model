// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>

#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Random/permutation.hpp"
#include "Random/randInt.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Chooses a random sample from an input array.
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype choice(GeneratorType& generator, const Matrix<dtype>& inArray)
            {
                uint32 randIdx = detail::randInt<uint32>(generator, inArray.size());
                return inArray[randIdx];
            }

            /// Chooses inNum random samples from an input array.
            /// @param generator: instance of a random number generator
            /// @param inArray
            /// @param inNum
            /// @param replace: Whether the sample is with or without replacement
            template<typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> choice(GeneratorType&        generator,
                                  const Matrix<dtype>& inArray,
                                  uint32                inNum,
                                  Replace               replace = Replace::YES)
            {
                if (replace == Replace::NO && inNum > inArray.size())
                {
                    THROW_INVALID_ARGUMENT_ERROR("when Replace::NO 'inNum' must be <= inArray.size()");
                }

                if (replace == Replace::YES)
                {
                    Matrix<dtype> outArray(1, inNum);
                    std::for_each(outArray.begin(),
                                  outArray.end(),
                                  [&generator, &inArray](dtype& value) -> void { value = choice(generator, inArray); });

                    return outArray;
                }

                return detail::permutation(generator, inArray)[Slice(inNum)];
            }
        }

        /// Chooses a random sample from an input array.
        template<typename dtype>
        dtype choice(const Matrix<dtype>& inArray)
        {
            return detail::choice(generator_, inArray);
        }

        /// Chooses inNum random samples from an input array.
        /// @param inArray
        /// @param inNum
        /// @param replace: Whether the sample is with or without replacement
        /// @return Matrix
        template<typename dtype>
        Matrix<dtype> choice(const Matrix<dtype>& inArray, uint32 inNum, Replace replace = Replace::YES)
        {
            return detail::choice(generator_, inArray, inNum, replace);
        }
    }
}

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
    /// Count the number of occurrences of each value in an array of non-negative ints.
    /// Negative values will be counted in the zero bin.
    ///
    /// The number of bins (of size 1) is one larger than the largest value in x.
    /// If minlength is specified, there will be at least this number of bins in
    /// the output array (though it will be longer if necessary, depending on the
    /// contents of x).Each bin gives the number of occurrences of its index value
    /// in x.
    template <typename dtype>
    Matrix<dtype> bincount(const Matrix<dtype> &inArray, uint16 inMinLength = 1)
    {
        STATIC_ASSERT_INTEGER(dtype);

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return Matrix<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            THROW_INVALID_ARGUMENT_ERROR(
                "array values too large, will result in gigantic array that will take up alot of memory...");
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        Matrix<dtype> clippedArray = inArray.clip(0, maxValue);

        Matrix<dtype> outArray(1, outArraySize);
        outArray.zeros();
        std::for_each(clippedArray.cbegin(),
                      clippedArray.cend(),
                      [&outArray](dtype value) noexcept -> void { ++outArray[value]; });

        return outArray;
    }

    //============================================================================
    // Method Description:
    /// Count number of occurrences of each value in array of non-negative ints.
    /// Negative values will be counted in the zero bin.
    ///
    /// The number of bins(of size 1) is one larger than the largest value in x.
    /// If minlength is specified, there will be at least this number of bins in
    /// the output array(though it will be longer if necessary, depending on the
    /// contents of x).Each bin gives the number of occurrences of its index value
    /// in x.If weights is specified the input array is weighted by it, i.e. if a
    /// value n is found at position i, out[n] += weight[i] instead of out[n] += 1.
    /// Weights array shall be of the same shape as inArray.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.bincount.html
    ///
    /// @param inArray
    /// @param inWeights
    /// @param inMinLength
    /// @return Matrix
    ///
    template <typename dtype>
    Matrix<dtype> bincount(const Matrix<dtype> &inArray, const Matrix<dtype> &inWeights, uint16 inMinLength = 1)
    {
        STATIC_ASSERT_INTEGER(dtype);

        if (inArray.shape() != inWeights.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("weights array must be the same shape as the input array.");
        }

        dtype maxValue = inArray.max().item();
        if (maxValue < 0)
        {
            // no positive values so just return an empty array
            return Matrix<dtype>(0);
        }

        if (maxValue + 1 > DtypeInfo<dtype>::max())
        {
            THROW_INVALID_ARGUMENT_ERROR(
                "array values too large, will result in gigantic array that will take up alot of memory...");
        }

        const uint16 outArraySize = std::max(static_cast<uint16>(maxValue + 1), inMinLength);
        Matrix<dtype> clippedArray = inArray.clip(0, maxValue);

        Matrix<dtype> outArray(1, outArraySize);
        outArray.zeros();
        uint32 counter = 0;
        std::for_each(clippedArray.cbegin(),
                      clippedArray.cend(),
                      [&outArray, &inWeights, &counter](dtype value) noexcept -> void
                      {
                          outArray[value] += inWeights[counter++];
                      });

        return outArray;
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>
#include <utility>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Functions/linspace.hpp"
#include "Functions/sort.hpp"
#include "Functions/zeros.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the histogram of a set of data.
    /// @param inArray
    /// @param inBinEdges: monotonically increasing array of bin edges, including the
    /// rightmost edge, allowing for non-uniform bin widths.
    /// @return array of histogram counts
    template <typename dtype>
    Matrix<uint32> histogram(const Matrix<dtype> &inArray, const Matrix<double> &inBinEdges)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inBinEdges.size() < 2)
        {
            THROW_INVALID_ARGUMENT_ERROR("number of bin edges must be >= 2.");
        }

        // sort just in case the user hasn't already
        const auto binEdges = sort(inBinEdges);

        Matrix<uint32> histo = zeros<uint32>(1, binEdges.size() - 1);
        for (const auto value : inArray)
        {
            if (value < binEdges.front() || value > binEdges.back())
            {
                continue;
            }

            // binary search to find the bin idx
            constexpr bool keepSearching = true;
            uint32 lowIdx = 0;
            uint32 highIdx = binEdges.size() - 1;
            while (keepSearching)
            {
                const uint32 idx = (lowIdx + highIdx) / 2; // integer division
                if (lowIdx == highIdx || lowIdx == highIdx - 1)
                {
                    // we found the bin
                    ++histo[lowIdx];
                    break;
                }

                if (value > binEdges[idx])
                {
                    lowIdx = idx;
                }
                else if (value < binEdges[idx])
                {
                    highIdx = idx;
                }
                else
                {
                    // we found the bin
                    ++histo[idx];
                    break;
                }
            }
        }

        return histo;
    }

    /// Compute the histogram of a set of data.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.histogram.html
    ///
    ///
    /// @param inArray
    /// @param inNumBins (default 10)
    ///
    /// @return a pair of matrices; first is histogram counts, seconds is the bin edges
    ///
    template <typename dtype>
    std::pair<Matrix<uint32>, Matrix<double>> histogram(const Matrix<dtype> &inArray, uint32 inNumBins = 10)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inNumBins == 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("number of bins must be positive.");
        }

        const Matrix<double> binEdges = linspace(static_cast<double>(inArray.min().item()),
                                                 static_cast<double>(inArray.max().item()),
                                                 inNumBins + 1,
                                                 EndPoint::YES);

        const auto histo = histogram(inArray, binEdges);
        return std::make_pair(histo, binEdges);
    }
}

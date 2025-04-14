// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/ComplexOperators.hpp"
#include "Core/Algorithm.hpp"
#include "Core/TypeTraits.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/mean.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Compute the average along the specified axis.
    template<typename dtype>
    auto average(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return mean(inArray, inAxis);
    }

    /// Compute the weighted average along the specified axis.
    template<typename dtype>
    Matrix<double> average(const Matrix<dtype>& inArray, const Matrix<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                Matrix<double> weightedArray(inArray.shape());
                algo::transform(inArray.cbegin(),
                                          inArray.cend(),
                                          inWeights.cbegin(),
                                          weightedArray.begin(),
                                          std::multiplies<double>()); // NOLINT(modernize-use-transparent-functors)

                double          sum         = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.);
                Matrix<double> returnArray = { sum /= inWeights.template astype<double>().sum().item() };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weights value are not consistant.");
                }

                double          weightSum = inWeights.template astype<double>().sum().item();
                Matrix<double> returnArray(1, arrayShape.m_Rows);
                for (uint32 row = 0; row < arrayShape.m_Rows; ++row)
                {
                    Matrix<double> weightedArray(1, arrayShape.m_Cols);
                    algo::transform(inArray.cbegin(row),
                                              inArray.cend(row),
                                              inWeights.cbegin(),
                                              weightedArray.begin(),
                                              std::multiplies<double>()); // NOLINT(modernize-use-transparent-functors)

                    double sum          = std::accumulate(weightedArray.begin(), weightedArray.end(), 0.);
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return average(inArray.transpose(), inWeights, Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {};
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Compute the weighted average along the specified axis.
    ///
    /// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.average.html
    ///
    /// @param inArray
    /// @param inWeights
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<double>>
        average(const Matrix<std::complex<dtype>>& inArray, const Matrix<dtype>& inWeights, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto multiplies = [](const std::complex<dtype>& lhs, dtype rhs) -> std::complex<double>
        { return complex_cast<double>(lhs) * static_cast<double>(rhs); };

        switch (inAxis)
        {
            case Axis::NONE:
            {
                if (inWeights.shape() != inArray.shape())
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weight values are not consistant.");
                }

                Matrix<std::complex<double>> weightedArray(inArray.shape());
                algo::transform(inArray.cbegin(),
                                          inArray.cend(),
                                          inWeights.cbegin(),
                                          weightedArray.begin(),
                                          multiplies);

                std::complex<double> sum =
                    std::accumulate(weightedArray.begin(), weightedArray.end(), std::complex<double>(0.));
                Matrix<std::complex<double>> returnArray = { sum /= inWeights.template astype<double>().sum().item() };

                return returnArray;
            }
            case Axis::COL:
            {
                const Shape arrayShape = inArray.shape();
                if (inWeights.size() != arrayShape.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array and weights value are not consistant.");
                }

                double                        weightSum = inWeights.template astype<double>().sum().item();
                Matrix<std::complex<double>> returnArray(1, arrayShape.m_Rows);
                for (uint32 row = 0; row < arrayShape.m_Rows; ++row)
                {
                    Matrix<std::complex<double>> weightedArray(1, arrayShape.m_Cols);
                    algo::transform(inArray.cbegin(row),
                                              inArray.cend(row),
                                              inWeights.cbegin(),
                                              weightedArray.begin(),
                                              multiplies);

                    const std::complex<double> sum =
                        std::accumulate(weightedArray.begin(), weightedArray.end(), std::complex<double>(0.));
                    returnArray(0, row) = sum / weightSum;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return average(inArray.transpose(), inWeights, Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
}

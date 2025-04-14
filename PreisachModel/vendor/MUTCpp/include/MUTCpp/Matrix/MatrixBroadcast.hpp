// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <utility>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "MatrixCore.hpp"

namespace mc
{
    namespace broadcast
    {
        //============================================================================
        // Method Description:
        /// Broadcasting template function for in-place operations
        ///
        /// @param function
        /// @param inArray1
        /// @param inArray2
        /// @param additionalFunctionArgs
        ///
        /// @return Matrix
        ///
        template <typename dtypeIn1, typename dtypeIn2, typename Function, typename... AdditionalFunctionArgs>
        Matrix<dtypeIn1> &broadcaster(Matrix<dtypeIn1> &inArray1,
                                      const Matrix<dtypeIn2> &inArray2,
                                      const Function &function,
                                      const AdditionalFunctionArgs &&... additionalFunctionArgs)
        {
            if (inArray1.shape() == inArray2.shape())
            {
                algo::transform(
                    inArray1.cbegin(),
                    inArray1.cend(),
                    inArray2.cbegin(),
                    inArray1.begin(),
                    [&function, &additionalFunctionArgs...](const auto &inValue1, const auto &inValue2) -> dtypeIn1
                    {
                        return function(inValue1,
                                        inValue2,
                                        std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                    });
            }
            else if (inArray2.isscalar())
            {
                const auto value = inArray2.item();
                algo::transform(
                    inArray1.cbegin(),
                    inArray1.cend(),
                    inArray1.begin(),
                    [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeIn1
                    {
                        return function(inValue, value,
                                        std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                    });
            }
            else if (inArray2.isflat())
            {
                if (inArray2.numRows() > 1 && inArray2.numRows() == inArray1.numRows())
                {
                    for (uint32 row = 0; row < inArray1.numRows(); ++row)
                    {
                        const auto value = inArray2[row];
                        algo::transform(
                            inArray1.cbegin(row),
                            inArray1.cend(row),
                            inArray1.begin(row),
                            [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeIn1
                            {
                                return function(inValue,
                                                value,
                                                std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                            });
                    }
                }
                else if (inArray2.numCols() > 1 && inArray2.numCols() == inArray1.numCols())
                {
                    for (uint32 col = 0; col < inArray1.numCols(); ++col)
                    {
                        const auto value = inArray2[col];
                        algo::transform(
                            inArray1.ccolbegin(col),
                            inArray1.ccolend(col),
                            inArray1.colbegin(col),
                            [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeIn1
                            {
                                return function(inValue,
                                                value,
                                                std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                            });
                    }
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
                }
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
            }

            return inArray1;
        }

        //============================================================================
        // Method Description:
        /// Broadcasting template function
        ///
        /// @param function
        /// @param inArray1
        /// @param inArray2
        /// @param additionalFunctionArgs
        ///
        /// @return Matrix
        ///
        template <typename dtypeOut,
                  typename dtypeIn1,
                  typename dtypeIn2,
                  typename Function,
                  typename... AdditionalFunctionArgs>
        Matrix<dtypeOut> broadcaster(const Matrix<dtypeIn1> &inArray1,
                                     const Matrix<dtypeIn2> &inArray2,
                                     const Function &function,
                                     const AdditionalFunctionArgs &&... additionalFunctionArgs)
        {
            if (inArray1.shape() == inArray2.shape())
            {
                return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
                {
                    Matrix<dtypeOut> returnArray(inArray1.shape());
                    algo::transform(
                        inArray1.cbegin(),
                        inArray1.cend(),
                        inArray2.cbegin(),
                        returnArray.begin(),
                        [&function, &additionalFunctionArgs...](const auto &inValue1, const auto &inValue2) -> dtypeOut
                        {
                            return function(inValue1,
                                            inValue2,
                                            std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                        });

                    return returnArray;
                }();
            }
            if (inArray1.isscalar())
            {
                const auto value = inArray1.item();
                return [&inArray2, &value, &function, &additionalFunctionArgs...]
                {
                    Matrix<dtypeOut> returnArray(inArray2.shape());
                    algo::transform(
                        inArray2.cbegin(),
                        inArray2.cend(),
                        returnArray.begin(),
                        [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeOut
                        {
                            return function(inValue,
                                            value,
                                            std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                        });
                    return returnArray;
                }();
            }
            if (inArray2.isscalar())
            {
                const auto value = inArray2.item();
                return [&inArray1, &value, &function, &additionalFunctionArgs...]
                {
                    Matrix<dtypeOut> returnArray(inArray1.shape());
                    algo::transform(
                        inArray1.cbegin(),
                        inArray1.cend(),
                        returnArray.begin(),
                        [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeOut
                        {
                            return function(inValue,
                                            value,
                                            std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                        });
                    return returnArray;
                }();
            }
            if (inArray1.isflat() && inArray2.isflat())
            {
                return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
                {
                    const auto numRows = std::max(inArray1.numRows(), inArray2.numRows());
                    const auto numCols = std::max(inArray1.numCols(), inArray2.numCols());
                    Matrix<dtypeOut> returnArray(numRows, numCols);
                    if (inArray1.numRows() > 1)
                    {
                        for (uint32 row = 0; row < inArray1.numRows(); ++row)
                        {
                            for (uint32 col = 0; col < inArray2.numCols(); ++col)
                            {
                                returnArray(row, col) =
                                    function(inArray1[row],
                                             inArray2[col],
                                             std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                            }
                        }
                    }
                    else
                    {
                        for (uint32 row = 0; row < inArray2.numRows(); ++row)
                        {
                            for (uint32 col = 0; col < inArray1.numCols(); ++col)
                            {
                                returnArray(row, col) =
                                    function(inArray1[col],
                                             inArray2[row],
                                             std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                            }
                        }
                    }
                    return returnArray;
                }();
            }
            if (inArray1.isflat())
            {
                return broadcaster<dtypeOut>(inArray2,
                                             inArray1,
                                             function,
                                             std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
            }
            if (inArray2.isflat())
            {
                if (inArray2.numRows() > 1 && inArray2.numRows() == inArray1.numRows())
                {
                    return [&inArray1, &inArray2, &function, &additionalFunctionArgs...]
                    {
                        Matrix<dtypeOut> returnArray(inArray1.shape());
                        for (uint32 row = 0; row < inArray1.numRows(); ++row)
                        {
                            const auto value = inArray2[row];
                            algo::transform(
                                inArray1.cbegin(row),
                                inArray1.cend(row),
                                returnArray.begin(row),
                                [&value, &function, &additionalFunctionArgs...](const auto &inValue) -> dtypeOut
                                {
                                    return function(inValue,
                                                    value,
                                                    std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...);
                                });
                        }
                        return returnArray;
                    }();
                }
                if (inArray2.numCols() > 1 && inArray2.numCols() == inArray1.numCols())
                {
                    return broadcaster<dtypeOut>(inArray1.transpose(),
                                                 inArray2.transpose(),
                                                 function,
                                                 std::forward<AdditionalFunctionArgs>(additionalFunctionArgs)...)
                        .transpose();
                }
                THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
            }
            else
            {
                THROW_INVALID_ARGUMENT_ERROR("operands could not be broadcast together");
            }

            return {}; // get rid of compiler warning
        }
    }
}

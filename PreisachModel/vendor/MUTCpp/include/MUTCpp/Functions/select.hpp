// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <vector>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return an array drawn from elements in choiceVec, depending on conditions.
    /// @param condVec The vector of conditions which determine from which array in choiceVec
    /// the output elements are taken. When multiple conditions are satisfied,
    /// the first one encountered in choiceVec is used.
    /// @param choiceVec The vector of array pointers from which the output elements are taken.
    /// It has to be of the same length as condVec.
    /// @param defaultValue The element inserted in output when all conditions evaluate to False
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> select(const std::vector<const Matrix<bool> *> &condVec,
                         const std::vector<const Matrix<dtype> *> &choiceVec,
                         dtype defaultValue = dtype{0})
    {
        if (choiceVec.size() != condVec.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("condVec and choiceVec need to be the same size");
        }

        if (choiceVec.size() == 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("choiceVec is size 0");
        }

        auto theShape = condVec.front()->shape();
        for (const auto cond : condVec)
        {
            const auto &theCond = *cond;
            if (theCond.shape() != theShape)
            {
                THROW_INVALID_ARGUMENT_ERROR("all Matrixs of the condVec must be the same shape");
            }
        }

        for (const auto choice : choiceVec)
        {
            const auto &theChoice = *choice;
            if (theChoice.shape() != theShape)
            {
                THROW_INVALID_ARGUMENT_ERROR(
                    "all Matrixs of the choiceVec must be the same shape, and the same as condVec");
            }
        }

        using size_type = typename Matrix<dtype>::size_type;
        constexpr auto nullChoice = std::numeric_limits<size_type>::max();

        Matrix<size_type> choiceIndices(theShape);
        choiceIndices.fill(nullChoice);
        for (size_type condIdx = 0; condIdx < condVec.size(); ++condIdx)
        {
            const auto &theCond = *condVec[condIdx];
            for (size_type i = 0; i < theCond.size(); ++i)
            {
                if (theCond[i] && choiceIndices[i] == nullChoice)
                {
                    choiceIndices[i] = condIdx;
                }
            }
        }

        Matrix<dtype> result(theShape);
        result.fill(defaultValue);
        for (size_type i = 0; i < choiceIndices.size(); ++i)
        {
            const auto choiceIndex = choiceIndices[i];
            if (choiceIndex != nullChoice)
            {
                const auto &theChoice = *choiceVec[choiceIndex];
                result[i] = theChoice[i];
            }
        }

        return result;
    }

    /// Return an array drawn from elements in choiceList, depending on conditions.
    /// @param condList The list of conditions which determine from which array in choiceList
    /// the output elements are taken. When multiple conditions are satisfied,
    /// the first one encountered in choiceList is used.
    /// @param choiceList The list of array pointers from which the output elements are taken.
    /// It has to be of the same length as condVec.
    /// @param defaultValue The element inserted in output when all conditions evaluate to False
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> select(const std::vector<Matrix<bool>> &condList,
                         const std::vector<Matrix<dtype>> &choiceList,
                         dtype defaultValue = dtype{0})
    {
        std::vector<const Matrix<bool> *> condVec(condList.size());
        algo::transform(condList.begin(),
                        condList.end(),
                        condVec.begin(),
                        [](auto &cond) noexcept -> const Matrix<bool> *{ return &cond; });

        std::vector<const Matrix<dtype> *> choiceVec(choiceList.size());
        algo::transform(choiceList.begin(),
                        choiceList.end(),
                        choiceVec.begin(),
                        [](auto &choice) noexcept -> const Matrix<dtype> *{ return &choice; });

        return select(condVec, choiceVec, defaultValue);
    }
}

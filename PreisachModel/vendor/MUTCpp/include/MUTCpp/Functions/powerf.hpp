// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Raises the elements of the array to the input floating point power
    template<typename dtype1, typename dtype2>
    auto powerf(dtype1 inValue, dtype2 inExponent) noexcept
    {
        return utils::powerf(inValue, inExponent);
    }
    
    /// Raises the elements of the array to the input floating point power
    template<typename dtype1, typename dtype2>
    auto powerf(const Matrix<dtype1>& inArray, dtype2 inExponent)
    {
        Matrix<decltype(powerf(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  returnArray.begin(),
                                  [inExponent](dtype1 inValue) noexcept -> auto
                                  { return mc::powerf(inValue, inExponent); });

        return returnArray;
    }

    /// Raises the elements of the array to the input floating point powers
    template<typename dtype1, typename dtype2>
    auto powerf(const Matrix<dtype1>& inArray, const Matrix<dtype2>& inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        Matrix<decltype(powerf(dtype1{ 0 }, dtype2{ 0 }))> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                                  inArray.cend(),
                                  inExponents.cbegin(),
                                  returnArray.begin(),
                                  [](dtype1 inValue, dtype2 inExponent) noexcept -> auto
                                  { return mc::powerf(inValue, inExponent); });

        return returnArray;
    }
}

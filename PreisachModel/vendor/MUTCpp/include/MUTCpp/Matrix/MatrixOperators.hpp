// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <complex>
#include <functional>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/TypeTraits.hpp"
#include "MatrixBroadcast.hpp"
#include "MatrixCore.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/essentiallyEqualComplex.hpp"

namespace mc
{
    //============================================================================
    // Method Description:
    /// Adds the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator+=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster(lhs, rhs, std::plus<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Adds the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator+=(Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        { return val1 + val2; };
        return broadcast::broadcaster(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator+=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value += rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator+=(Matrix<std::complex<dtype>>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value) -> std::complex<dtype> { return value += rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator+(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::plus<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Adds the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(const Matrix<dtype>& lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 + val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Adds the elements of two arrays (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(const Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator+(Matrix<dtype> lhs, dtype rhs)
    {
        lhs += rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (5)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator+(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (6)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(const Matrix<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value) -> std::complex<dtype> { return value + rhs; };

        Matrix<std::complex<dtype>> returnArray(lhs.shape());

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (7)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(const std::complex<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (8)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(Matrix<std::complex<dtype>> lhs, dtype rhs)
    {
        lhs += rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Adds the scalar to the array (9)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator+(dtype lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        return rhs + lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator-=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster(lhs, rhs, std::minus<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Subtracts the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator-=(Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        { return val1 - val2; };
        return broadcast::broadcaster(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator-=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value -= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator-=(Matrix<std::complex<dtype>>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value) -> std::complex<dtype> { return value -= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator-(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::minus<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Subtracts the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(const Matrix<dtype>& lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 - val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Subtracts the elements of two arrays (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(const Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 - val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator-(Matrix<dtype> lhs, dtype rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (5)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator-(dtype lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [lhs](dtype value) -> dtype { return lhs - value; };

        Matrix<dtype> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (6)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(const Matrix<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value) -> std::complex<dtype> { return value - rhs; };

        Matrix<std::complex<dtype>> returnArray(lhs.shape());

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (7)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(const std::complex<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](dtype value) -> std::complex<dtype> { return lhs - value; };

        Matrix<std::complex<dtype>> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (8)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(Matrix<std::complex<dtype>> lhs, dtype rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Subtracts the scalar from the array (9)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator-(dtype lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](std::complex<dtype> value) -> std::complex<dtype> { return lhs - value; };

        Matrix<std::complex<dtype>> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Negative Operator
    ///
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator-(const Matrix<dtype>& inArray)
    {
        const auto function = [](dtype value) -> dtype { return -value; };

        auto returnArray = Matrix<dtype>(inArray.shape());
        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);
        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator*=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster(lhs, rhs, std::multiplies<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Multiplies the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator*=(Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        { return val1 * val2; };
        return broadcast::broadcaster(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator*=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value *= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator*=(Matrix<std::complex<dtype>>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value) -> std::complex<dtype> { return value *= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator*(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::multiplies<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Multiplies the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(const Matrix<dtype>& lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 * val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Multiplies the elements of two arrays (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(const Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator*(Matrix<dtype> lhs, dtype rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (5)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator*(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (6)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(const Matrix<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value) -> std::complex<dtype> { return value * rhs; };

        Matrix<std::complex<dtype>> returnArray(lhs.shape());

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (7)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(const std::complex<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (8)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(Matrix<std::complex<dtype>> lhs, dtype rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Multiplies the scalar to the array (9)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator*(dtype lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        return rhs * lhs;
    }

    //============================================================================
    // Method Description:
    /// Divides the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator/=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster(lhs, rhs, std::divides<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Divides the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator/=(Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const std::complex<dtype>& val1, dtype val2) -> std::complex<dtype>
        { return val1 / val2; };
        return broadcast::broadcaster(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator/=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value /= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>>& operator/=(Matrix<std::complex<dtype>>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](std::complex<dtype>& value) -> std::complex<dtype> { return value /= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Divides the elements of two arrays (1)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator/(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::divides<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Divides the elements of two arrays (2)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(const Matrix<dtype>& lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 / val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Divides the elements of two arrays (3)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(const Matrix<std::complex<dtype>>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](const auto& val1, const auto& val2) -> std::complex<dtype> { return val1 / val2; };
        return broadcast::broadcaster<std::complex<dtype>>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (4)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator/(Matrix<dtype> lhs, dtype rhs)
    {
        lhs /= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (5)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator/(dtype lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [lhs](dtype value) -> dtype { return lhs / value; };

        Matrix<dtype> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (6)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(const Matrix<dtype>& lhs, const std::complex<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [rhs](dtype value) -> std::complex<dtype> { return value / rhs; };

        Matrix<std::complex<dtype>> returnArray(lhs.shape());

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (7)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(const std::complex<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](dtype value) -> std::complex<dtype> { return lhs / value; };

        Matrix<std::complex<dtype>> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (8)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(Matrix<std::complex<dtype>> lhs, dtype rhs)
    {
        lhs /= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Divides the scalar from the array (9)
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<std::complex<dtype>> operator/(dtype lhs, const Matrix<std::complex<dtype>>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [lhs](const std::complex<dtype>& value) -> std::complex<dtype> { return lhs / value; };

        Matrix<std::complex<dtype>> returnArray(rhs.shape());

        algo::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Modulus the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_floating_point_v<dtype>, int> = 0>
    Matrix<dtype>& operator%=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        if constexpr (std::is_integral_v<dtype>)
        {
            return broadcast::broadcaster(lhs, rhs, std::modulus<dtype>());
        }
        else
        {
            const auto function = [](const dtype value1, const dtype value2) -> dtype
            { return std::fmod(value1, value2); };
            return broadcast::broadcaster(lhs, rhs, function);
        }
    }

    //============================================================================
    // Method Description:
    /// Modulus the scalar to the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_floating_point_v<dtype>, int> = 0>
    Matrix<dtype>& operator%=(Matrix<dtype>& lhs, dtype rhs)
    {
        if constexpr (std::is_integral_v<dtype>)
        {
            const auto function = [rhs](dtype& value) -> dtype { return value %= rhs; };
            algo::for_each(lhs.begin(), lhs.end(), function);
        }
        else
        {
            const auto function = [rhs](dtype& value) -> void { value = std::fmod(value, rhs); };
            algo::for_each(lhs.begin(), lhs.end(), function);
        }

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the modulus of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype> || std::is_floating_point_v<dtype>, int> = 0>
    Matrix<dtype> operator%(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        if constexpr (std::is_integral_v<dtype>)
        {
            return broadcast::broadcaster<dtype>(lhs, rhs, std::modulus<dtype>());
        }
        else
        {
            const auto function = [](dtype value1, dtype value2) -> dtype { return std::fmod(value1, value2); };
            return broadcast::broadcaster<dtype>(lhs, rhs, function);
        }
    }

    //============================================================================
    // Method Description:
    /// Modulus of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator%(Matrix<dtype> lhs, dtype rhs)
    {
        lhs %= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Modulus of the scalar and the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
    Matrix<dtype> operator%(dtype lhs, const Matrix<dtype>& rhs)
    {
        Matrix<dtype> returnArray(rhs.shape());
        algo::transform(rhs.begin(),
                                  rhs.end(),
                                  returnArray.begin(),
                                  [lhs](dtype value) -> dtype { return lhs % value; });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Modulus of the scalar and the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int> = 0>
    Matrix<dtype> operator%(dtype lhs, const Matrix<dtype>& rhs)
    {
        Matrix<dtype> returnArray(rhs.shape());
        algo::transform(rhs.begin(),
                                  rhs.end(),
                                  returnArray.begin(),
                                  [lhs](dtype value) -> dtype { return std::fmod(lhs, value); });

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Bitwise or the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator|=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster(lhs, rhs, std::bit_or<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Bitwise or the scalar to the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator|=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value |= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise or of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator|(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::bit_or<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise or of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator|(Matrix<dtype> lhs, dtype rhs)
    {
        lhs |= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise or of the sclar and the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator|(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs | lhs;
    }

    //============================================================================
    // Method Description:
    /// Bitwise and the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator&=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster(lhs, rhs, std::bit_and<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Bitwise and the scalar to the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator&=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value &= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise and of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator&(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::bit_and<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise and of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator&(Matrix<dtype> lhs, dtype rhs)
    {
        lhs &= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise and of the sclar and the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator&(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs & lhs;
    }

    //============================================================================
    // Method Description:
    /// Bitwise xor the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator^=(Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster(lhs, rhs, std::bit_xor<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Bitwise xor the scalar to the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator^=(Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [rhs](dtype& value) -> dtype { return value ^= rhs; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise xor of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator^(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_INTEGER(dtype);

        return broadcast::broadcaster<dtype>(lhs, rhs, std::bit_xor<dtype>());
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise xor of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator^(Matrix<dtype> lhs, dtype rhs)
    {
        lhs ^= rhs;
        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise xor of the sclar and the array
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator^(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs ^ lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the bitwise not of the array
    ///
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator~(const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [](dtype value) -> dtype { return ~value; };

        Matrix<dtype> returnArray(inArray.shape());

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Takes the and of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator&&(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype value1, dtype value2) -> bool
        { return !utils::essentiallyEqual(value1, dtype{ 0 }) && !utils::essentiallyEqual(value2, dtype{ 0 }); };

        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Takes the and of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator&&(const Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [rhs](dtype value) -> bool { return value && rhs; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Takes the and of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator&&(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs && lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the or of the elements of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator||(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype value1, dtype value2) -> bool
        { return !utils::essentiallyEqual(value1, dtype{ 0 }) || !utils::essentiallyEqual(value2, dtype{ 0 }); };

        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Takes the or of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator||(const Matrix<dtype>& lhs, dtype rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [rhs](dtype value) -> bool { return value || rhs; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Takes the or of the array and the scalar
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator||(dtype lhs, const Matrix<dtype>& rhs)
    {
        return rhs || lhs;
    }

    //============================================================================
    // Method Description:
    /// Takes the not of the array
    ///
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator!(const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        Matrix<bool> returnArray(inArray.shape());

        const auto function = [](dtype value) -> dtype { return !value; };

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator==(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        const auto equalTo = [](dtype lhs_, dtype rhs_) noexcept -> bool
        { return utils::essentiallyEqual(lhs_, rhs_); };

        return broadcast::broadcaster<bool>(lhs, rhs, equalTo);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// an array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator==(const Matrix<dtype>& lhs, dtype inValue)
    {
        Matrix<bool> returnArray(lhs.shape());

        const auto equalTo = [inValue](dtype value) noexcept -> bool
        { return utils::essentiallyEqual(inValue, value); };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), equalTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// an array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator==(dtype inValue, const Matrix<dtype>& inArray)
    {
        return inArray == inValue;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator!=(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        const auto notEqualTo = [](dtype lhs_, dtype rhs_) noexcept -> bool
        { return !utils::essentiallyEqual(lhs_, rhs_); };

        return broadcast::broadcaster<bool>(lhs, rhs, notEqualTo);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// an array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator!=(const Matrix<dtype>& lhs, dtype inValue)
    {
        Matrix<bool> returnArray(lhs.shape());

        const auto notEqualTo = [inValue](dtype value) noexcept -> bool
        { return !utils::essentiallyEqual(inValue, value); };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), notEqualTo);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// an array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator!=(dtype inValue, const Matrix<dtype>& inArray)
    {
        return inArray != inValue;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [](dtype lhs_, dtype rhs_) noexcept -> bool { return lhs_ < rhs_; };
        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<(const Matrix<dtype>& lhs, dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return value < inValue; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<(dtype inValue, const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return inValue < value; };

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [](dtype lhs_, dtype rhs_) noexcept -> bool { return lhs_ > rhs_; };
        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>(const Matrix<dtype>& lhs, dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return value > inValue; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>(dtype inValue, const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return inValue > value; };

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<=(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [](dtype lhs_, dtype rhs_) noexcept -> bool { return lhs_ <= rhs_; };
        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<=(const Matrix<dtype>& lhs, dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return value <= inValue; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator<=(dtype inValue, const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return inValue <= value; };

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// of two arrays
    ///
    /// @param lhs
    /// @param rhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>=(const Matrix<dtype>& lhs, const Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto function = [](dtype lhs_, dtype rhs_) noexcept -> bool { return lhs_ >= rhs_; };
        return broadcast::broadcaster<bool>(lhs, rhs, function);
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param lhs
    /// @param inValue
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>=(const Matrix<dtype>& lhs, dtype inValue)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(lhs.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return value >= inValue; };

        algo::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Returns an array of booleans of element wise comparison
    /// the array and a scalar
    ///
    /// @param inValue
    /// @param inArray
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<bool> operator>=(dtype inValue, const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        Matrix<bool> returnArray(inArray.shape());

        const auto function = [inValue](dtype value) noexcept -> bool { return inValue >= value; };

        algo::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Bitshifts left the elements of the array
    ///
    /// @param lhs
    /// @param inNumBits
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator<<=(Matrix<dtype>& lhs, uint8 inNumBits)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [inNumBits](dtype& value) -> void { value <<= inNumBits; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Bitshifts left the elements of the array
    ///
    /// @param lhs
    /// @param inNumBits
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator<<(const Matrix<dtype>& lhs, uint8 inNumBits)
    {
        STATIC_ASSERT_INTEGER(dtype);

        Matrix<dtype> returnArray(lhs);
        returnArray <<= inNumBits;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// Bitshifts right the elements of the array
    ///
    /// @param lhs
    /// @param inNumBits
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator>>=(Matrix<dtype>& lhs, uint8 inNumBits)
    {
        STATIC_ASSERT_INTEGER(dtype);

        const auto function = [inNumBits](dtype& value) -> void { value >>= inNumBits; };

        algo::for_each(lhs.begin(), lhs.end(), function);

        return lhs;
    }

    //============================================================================
    // Method Description:
    /// Bitshifts right the elements of the array
    ///
    /// @param lhs
    /// @param inNumBits
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator>>(const Matrix<dtype>& lhs, uint8 inNumBits)
    {
        STATIC_ASSERT_INTEGER(dtype);

        Matrix<dtype> returnArray(lhs);
        returnArray >>= inNumBits;
        return returnArray;
    }

    //============================================================================
    // Method Description:
    /// prefix incraments the elements of an array
    ///
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator++(Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype& value) -> void { ++value; };

        algo::for_each(rhs.begin(), rhs.end(), function);

        return rhs;
    }

    //============================================================================
    // Method Description:
    /// postfix increments the elements of an array
    ///
    /// @param lhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator++(Matrix<dtype>& lhs, int)
    {
        auto copy = Matrix<dtype>(lhs);
        ++lhs;
        return copy;
    }

    //============================================================================
    // Method Description:
    /// prefix decrements the elements of an array
    ///
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype>& operator--(Matrix<dtype>& rhs)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        const auto function = [](dtype& value) -> void { --value; };

        algo::for_each(rhs.begin(), rhs.end(), function);

        return rhs;
    }

    //============================================================================
    // Method Description:
    /// postfix decrements the elements of an array
    ///
    /// @param lhs
    /// @return Matrix
    ///
    template<typename dtype>
    Matrix<dtype> operator--(Matrix<dtype>& lhs, int)
    {
        auto copy = Matrix<dtype>(lhs);
        --lhs;
        return copy;
    }

    //============================================================================
    // Method Description:
    /// io operator for the Matrix class
    ///
    /// @param inOStream
    /// @param inArray
    /// @return std::ostream
    ///
    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const Matrix<dtype>& inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        inOStream << inArray.str();
        return inOStream;
    }
}

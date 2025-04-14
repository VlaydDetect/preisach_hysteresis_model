// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <complex>

#include "StaticAssert.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    /// Less than operator for std::complex<T>
    /// @return bool true if lhs < rhs
    template <typename T>
    bool operator<(const std::complex<T> &lhs, const std::complex<T> &rhs) noexcept
    {
        if (!utils::essentiallyEqual(lhs.real(), rhs.real()))
        {
            return lhs.real() < rhs.real();
        }

        return lhs.imag() < rhs.imag();
    }

    /// Less than or equal operator for std::complex<T>
    /// @return bool true if lhs <= rhs
    template <typename T>
    bool operator<=(const std::complex<T> &lhs, const std::complex<T> &rhs) noexcept
    {
        if (!utils::essentiallyEqual(lhs.real(), rhs.real()))
        {
            return lhs.real() <= rhs.real();
        }

        return lhs.imag() <= rhs.imag();
    }

    /// Greater than operator for std::complex<T>
    /// @return bool true if lhs > rhs
    template <typename T>
    bool operator>(const std::complex<T> &lhs, const std::complex<T> &rhs) noexcept
    {
        return !(lhs <= rhs);
    }

    /// Greater than or equal operator for std::complex<T>
    /// @return bool true if lhs >= rhs
    template <typename T>
    bool operator>=(const std::complex<T> &lhs, const std::complex<T> &rhs) noexcept
    {
        return !(lhs < rhs);
    }

    /// Greater than or equal operator for std::complex<T>
    /// @return std::complex<Out>
    template <typename Out, typename In>
    std::complex<Out> complex_cast(const std::complex<In> &value) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(Out);

        return std::complex<Out>(static_cast<Out>(value.real()), static_cast<Out>(value.imag()));
    }

    template <typename T, typename dtype>
    std::complex<T> operator*(const dtype &a, const std::complex<T> &b)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (a == static_cast<dtype>(0))
            return {static_cast<T>(0), static_cast<T>(0)};

        return { b.real() * static_cast<T>(a), b.imag() * static_cast<T>(a) };
    }

    template <typename T, typename dtype>
    std::complex<T> operator*(const std::complex<T> &b, const dtype &a)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (a == static_cast<dtype>(0))
            return {static_cast<T>(0), static_cast<T>(0)};

        return { b.real() * static_cast<T>(a), b.imag() * static_cast<T>(a) };
    }
}

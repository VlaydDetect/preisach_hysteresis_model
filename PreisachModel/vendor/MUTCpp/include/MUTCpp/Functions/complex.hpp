// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns a std::complex from the input real and imag components
    /// @param inReal: the real component of the complex number
    template<typename dtype>
    auto complex(dtype inReal)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::complex<dtype>(inReal);
    }
    
    /// Returns a std::complex from the input real and imag components
    /// @param inReal: the real component of the complex number
    /// @param inImag: the imaginary component of the complex number
    template<typename dtype>
    auto complex(dtype inReal, dtype inImag)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::complex<dtype>(inReal, inImag);
    }

    /// Returns a std::complex from the input real and imag components
    /// @param inReal: the real component of the complex number
    /// @return Matrix
    template<typename dtype>
    auto complex(const Matrix<dtype>& inReal)
    {
        Matrix<decltype(mc::complex(dtype{ 0 }))> returnArray(inReal.shape());
        algo::transform(inReal.cbegin(),
                                  inReal.cend(),
                                  returnArray.begin(),
                                  [](dtype real) -> auto { return mc::complex(real); });

        return returnArray;
    }

    /// Returns a std::complex from the input real and imag components
    /// @param inReal: the real component of the complex number
    /// @param inImag: the imaginary component of the complex number
    /// @return Matrix
    template<typename dtype>
    auto complex(const Matrix<dtype>& inReal, const Matrix<dtype>& inImag)
    {
        if (inReal.shape() != inImag.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("Input real array must be the same shape as input imag array");
        }

        Matrix<decltype(mc::complex(dtype{ 0 }, dtype{ 0 }))> returnArray(inReal.shape());
        algo::transform(inReal.cbegin(),
                                  inReal.cend(),
                                  inImag.cbegin(),
                                  returnArray.begin(),
                                  [](dtype real, dtype imag) -> auto { return mc::complex(real, imag); });

        return returnArray;
    }
}

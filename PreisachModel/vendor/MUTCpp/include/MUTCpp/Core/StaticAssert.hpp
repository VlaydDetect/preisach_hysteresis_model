// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <type_traits>
#include "Core/TypeTraits.hpp"

#define STATIC_ASSERT_VALID_DTYPE(dtype) \
static_assert(mc::is_valid_dtype_v<dtype>, "Template type is not a valid dtype for NdArray")

#define STATIC_ASSERT_ARITHMETIC(dtype) \
static_assert(std::is_arithmetic_v<dtype>, "Can only be used with arithmetic types")

#define STATIC_ASSERT_INTEGER(dtype) static_assert(std::is_integral_v<dtype>, "Can only be used with integer types")

#define STATIC_ASSERT_UNSIGNED_INTEGER(dtype) \
static_assert(std::is_integral_v<dtype> && std::is_unsigned_v<dtype>, "Can only be used with unsigned integer types")

#define STATIC_ASSERT_FLOAT(dtype) static_assert(std::is_floating_point_v<dtype>, "Can only be used with float types")

#define STATIC_ASSERT_COMPLEX(dtype) static_assert(mc::is_complex_v<dtype>, "Can only be used with std::complex types")

#define STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype)                        \
static_assert(std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>, \
"Can only be used with arithmetic types or std::complex types")

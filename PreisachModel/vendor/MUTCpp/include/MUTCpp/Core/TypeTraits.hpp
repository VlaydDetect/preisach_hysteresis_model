// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <complex>
#include <type_traits>

namespace mc {
    // ================= all_arithmetic =================
    /// Template class for determining if all the types are arithmetic
    template<typename... Ts>
    struct all_arithmetic;

    /// Template class specialization for determining if all of the types are arithmetic
    template<typename Head, typename... Tail>
    struct all_arithmetic<Head, Tail...> {
        static constexpr bool value = std::is_arithmetic_v<Head> && all_arithmetic<Tail...>::value;
    };

    /// Template class specialization for determining if all of the types are arithmetic
    template<typename T>
    struct all_arithmetic<T> {
        static constexpr bool value = std::is_arithmetic_v<T>;
    };

    /// all_arithmetic helper
    template<typename... Ts>
    constexpr bool all_arithmetic_v = all_arithmetic<Ts...>::value;
    // ==================================================

    // ==================== all_same ====================
    /// Template class for determining if all the types are the same as another type
    template<typename T1, typename... Ts>
    struct all_same;


    /// Template class specialization for determining if all the types are the same as another type
    template<typename T1, typename Head, typename... Tail>
    struct all_same<T1, Head, Tail...> {
        static constexpr bool value = std::is_same_v<T1, Head> && all_same<T1, Tail...>::value;
    };

    /// Template class specialization for determining if all of the types are the same as another type
    template<typename T1, typename T2>
    struct all_same<T1, T2> {
        static constexpr bool value = std::is_same_v<T1, T2>;
    };

    /// all_same helper
    template<typename... Ts>
    constexpr bool all_same_v = all_same<Ts...>::value;
    // ==================================================

    // ==================================================
    /// Template class for determining if dtype is a valid dtype for NdArray
    template<typename dtype>
    struct is_valid_dtype {
        static constexpr bool value =
                std::is_default_constructible_v<dtype> && std::is_nothrow_copy_constructible_v<dtype> &&
                std::is_nothrow_move_constructible_v<dtype> && std::is_nothrow_copy_assignable_v<dtype> &&
                std::is_nothrow_move_assignable_v<dtype> && std::is_nothrow_destructible_v<dtype> &&
                !std::is_void_v<dtype> && !std::is_pointer_v<dtype> && !std::is_array_v<dtype> &&
                !std::is_union_v<dtype> && !std::is_function_v<dtype> && !std::is_abstract_v<dtype>;
    };

    /// is_valid_dtype helper
    template<class dtype>
    constexpr bool is_valid_dtype_v = is_valid_dtype<dtype>::value;
    // ==================================================

    // Forward declare
    template<typename dtype, class Allocator>
    class Matrix;

    /// Template class for determining if dtype is a valid index type for NdArray
    ///
    template<typename>
    struct is_matrix_int : std::false_type {
    };

    /// Template class for determining if dtype is a valid index typefor NdArray
    ///

    template<typename dtype, typename Allocator>
    struct is_matrix_int<Matrix<dtype, Allocator> > {
        static constexpr bool value = std::is_integral_v<dtype>;
    };


    /// is_ndarray_int helper
    ///
    template<typename T>
    constexpr bool is_matrix_int_v = is_matrix_int<T>::value;


    /// is_ndarray_int
    ///
    template<typename T>
    using matrix_int_concept = std::enable_if_t<is_matrix_int_v<T>, int>;


    /// Template class for determining if type is std::complex<>
    ///
    template<class T>
    struct is_complex {
        static constexpr bool value = false;
    };


    /// Template class specialization for determining if type is std::complex<>
    ///
    template<class T>
    struct is_complex<std::complex<T> > {
        static constexpr bool value = true;
    };

    /// is_complex helper
    ///
    template<class T>
    constexpr bool is_complex_v = is_complex<T>::value;

    /// type trait to test if one value is larger than another at compile time
    ///
    template<std::size_t Value1, std::size_t Value2>
    struct greaterThan {
        static constexpr bool value = Value1 > Value2;
    };

    /// greaterThan helper
    ///
    template<std::size_t Value1, std::size_t Value2>
    constexpr bool greaterThan_v = greaterThan<Value1, Value2>::value;
}

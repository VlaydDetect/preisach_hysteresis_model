// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <array>
#include <deque>
#include <forward_list>
#include <initializer_list>
#include <iterator>
#include <list>
#include <set>
#include <vector>

#include "Core/Enums.hpp"
#include "Core/TypeTraits.hpp"
#include "Matrix.hpp"

namespace mc
{

    /// Convert the list initializer to an array.
    /// eg: Matrix<int> myArray = mc::asarray<int>({1,2,3});
    template <typename dtype, std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
    Matrix<dtype> asarray(std::initializer_list<dtype> inList)
    {
        return Matrix<dtype>(inList);
    }


    /// Convert the list initializer to an array.
    /// eg: Matrix<int> myArray = mc::asarray<int>({{1,2,3}, {4, 5, 6}});
    template <typename dtype>
    Matrix<dtype> asarray(std::initializer_list<std::initializer_list<dtype>> inList)
    {
        return Matrix<dtype>(inList);
    }


    /// Convert the std::array to an array.
    /// @param inArray
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    ///
    template <typename dtype, size_t ArraySize, std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
    Matrix<dtype> asarray(std::array<dtype, ArraySize> &inArray, PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return Matrix<dtype>(inArray, pointerPolicy);
    }


    /// Convert the std::array to an array.
    /// @param inArray
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    ///
    template <typename dtype, size_t Dim0Size, size_t Dim1Size>
    Matrix<dtype> asarray(std::array<std::array<dtype, Dim1Size>, Dim0Size> &inArray,
                          PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return Matrix<dtype>(inArray, pointerPolicy);
    }


    /// Convert the vector to an array.
    /// @param inVector
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    ///
    template <typename dtype, std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
    Matrix<dtype> asarray(std::vector<dtype> &inVector, PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return Matrix<dtype>(inVector, pointerPolicy);
    }


    /// Convert the vector to an array. Makes a copy of the data.
    template <typename dtype>
    Matrix<dtype> asarray(const std::vector<std::vector<dtype>> &inVector)
    {
        return Matrix<dtype>(inVector);
    }


    /// Convert the vector to an array.
    /// @param inVector
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    ///
    template <typename dtype, size_t Dim1Size>
    Matrix<dtype> asarray(std::vector<std::array<dtype, Dim1Size>> &inVector,
                          PointerPolicy pointerPolicy = PointerPolicy::COPY)
    {
        return Matrix<dtype>(inVector, pointerPolicy);
    }


    /// Convert the vector to an array.
    template <typename dtype, std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
    Matrix<dtype> asarray(const std::deque<dtype> &inDeque)
    {
        return Matrix<dtype>(inDeque);
    }


    /// Convert the vector to an array. Makes a copy of the data.
    template <typename dtype>
    Matrix<dtype> asarray(const std::deque<std::deque<dtype>> &inDeque)
    {
        return Matrix<dtype>(inDeque);
    }


    /// Convert the set to an array. Makes a copy of the data.
    template <typename dtype, typename dtypeComp>
    Matrix<dtype> asarray(const std::set<dtype, dtypeComp> &inSet)
    {
        return Matrix<dtype>(inSet);
    }


    /// Convert the list to an array. Makes a copy of the data.
    template <typename dtype>
    Matrix<dtype> asarray(const std::list<dtype> &inList)
    {
        return Matrix<dtype>(inList);
    }


    /// Convert the forward_list to an array. Makes a copy of the data.
    template <typename Iterator>
    auto asarray(Iterator iterBegin, Iterator iterEnd)
    {
        return Matrix<typename std::iterator_traits<Iterator>::value_type>(iterBegin, iterEnd);
    }


    /// Convert the forward_list to an array. Makes a copy of the data.
    template <typename dtype>
    Matrix<dtype> asarray(const dtype *iterBegin, const dtype *iterEnd)
    {
        return Matrix<dtype>(iterBegin, iterEnd);
    }


    /// Convert the c-style array to an array. Makes a copy of the data.
    /// @param ptr to array
    /// @param size: the number of elements in the array
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> asarray(const dtype *ptr, uint32 size)
    {
        return Matrix<dtype>(ptr, size);
    }


    /// Convert the c-style array to an array. Makes a copy of the data.
    /// @param ptr to array
    /// @param numRows: number of rows of the buffer
    /// @param numCols: number of cols of the buffer
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> asarray(const dtype *ptr, uint32 numRows, uint32 numCols)
    {
        return Matrix<dtype>(ptr, numRows, numCols);
    }


    /// Convert the c-style array to an array. Makes a copy of the data.
    /// @param ptr to array
    /// @param size: the number of elements in the array
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    template <typename dtype,
              typename UIntType,
              std::enable_if_t<std::is_integral_v<UIntType> && !std::is_same_v<UIntType, bool>, int>  = 0>
    Matrix<dtype> asarray(dtype *ptr, UIntType size, PointerPolicy pointerPolicy = PointerPolicy::COPY) noexcept
    {
        return Matrix<dtype>(ptr, size, pointerPolicy);
    }


    /// Convert the c-style array to an array. Makes a copy of the data.
    /// @param ptr to array
    /// @param numRows: number of rows of the buffer
    /// @param numCols: number of cols of the buffer
    /// @param pointerPolicy: (optional) whether to make a copy and own the data, or
    ///                       act as a non-owning shell. Default Copy
    /// @return Matrix
    template <typename dtype,
              typename UIntType1,
              typename UIntType2,
              std::enable_if_t<std::is_integral_v<UIntType1> && !std::is_same_v<UIntType1, bool>, int>  = 0,
              std::enable_if_t<std::is_integral_v<UIntType2> && !std::is_same_v<UIntType2, bool>, int>  = 0>
    Matrix<dtype> asarray(dtype *ptr,
                          UIntType1 numRows,
                          UIntType2 numCols,
                          PointerPolicy pointerPolicy = PointerPolicy::COPY) noexcept
    {
        return Matrix<dtype>(ptr, numRows, numCols, pointerPolicy);
    }
}

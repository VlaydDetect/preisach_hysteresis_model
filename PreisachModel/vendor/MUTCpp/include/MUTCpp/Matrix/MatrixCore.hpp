// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <forward_list>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Core/Constants.hpp"
#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/ComplexOperators.hpp"
#include "Core/Algorithm.hpp"
#include "Core/TypeTraits.hpp"
#include "Core/Shape.hpp"
#include "Core/Slice.hpp"
#include "Core/Types.hpp"
#include "Matrix/MatrixIterators.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/essentiallyEqualComplex.hpp"
#include "Utils/num2str.hpp"
#include "Utils/mathOperations.hpp"
#include "Utils/value2str.hpp"

namespace mc
{
    namespace type_traits
    {
        /// Template class for determining if dtype is a valid index type for Matrix
        template <typename>
        struct is_matrix_int : std::false_type
        {
        };

        /// Template class for determining if dtype is a valid index typefor Matrix
        template <typename dtype, typename Allocator>
        struct is_matrix_int<Matrix<dtype, Allocator>>
        {
            static constexpr bool value = std::is_integral_v<dtype>;
        };

        /// Template class for determining if dtype is a valid index typefor std::initializer_list
        template <typename dtype>
        struct is_matrix_int<std::initializer_list<dtype>>
        {
            static constexpr bool value = std::is_integral_v<dtype>;
        };

        /// Template class for determining if dtype is a valid index typefor std::vector
        template <typename dtype>
        struct is_matrix_int<std::vector<dtype>>
        {
            static constexpr bool value = std::is_integral_v<dtype>;
        };

        /// is_matrix_int helper
        template <typename T>
        constexpr bool is_matrix_int_v = is_matrix_int<T>::value;

        /// Template class for determining if dtype is an unsigned integer type
        template <typename>
        struct is_matrix_signed_int : std::false_type
        {
        };

        /// Template class for determining if dtype is an unsigned integer type
        template <typename dtype, typename Allocator>
        struct is_matrix_signed_int<Matrix<dtype, Allocator>>
        {
            static constexpr bool value = std::is_signed_v<dtype>;
        };

        /// is_matrix_int helper
        template <typename T>
        constexpr bool is_matrix_signed_int_v = is_matrix_signed_int<T>::value;

        /// is_matrix_int
        template <typename T>
        using matrix_int_concept = std::enable_if_t<is_matrix_int_v<T>, int>;
    }

    /// Holds 1D and 2D arrays, the main work horse of the MUTCpp library
    template <typename dtype, class Allocator = std::allocator<dtype>>
    class Matrix
    {
    private:
        STATIC_ASSERT_VALID_DTYPE(dtype);
        static_assert(std::is_same_v<dtype, typename Allocator::value_type>,
                      "value_type and Allocator::value_type must match");

        using AllocType = typename std::allocator_traits<Allocator>::template rebind_alloc<dtype>;
        using AllocTraits = std::allocator_traits<AllocType>;

    public:
        using self_type = Matrix<dtype, Allocator>;
        using value_type = dtype;
        using allocator_type = Allocator;
        using pointer = typename AllocTraits::pointer;
        using const_pointer = typename AllocTraits::const_pointer;
        using reference = dtype &;
        using const_reference = const dtype &;
        using size_type = uint32;
        using index_type = int32;
        using difference_type = typename AllocTraits::difference_type;

        using iterator = MatrixIterator<dtype, pointer, difference_type>;
        using const_iterator = MatrixConstIterator<dtype, const_pointer, difference_type>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        using column_iterator = MatrixColumnIterator<dtype, size_type, pointer, difference_type>;
        using const_column_iterator = MatrixConstColumnIterator<dtype, size_type, const_pointer, difference_type>;
        using reverse_column_iterator = std::reverse_iterator<column_iterator>;
        using const_reverse_column_iterator = std::reverse_iterator<const_column_iterator>;

        Matrix() = default;

        explicit Matrix(size_type inSquareSize) :
            shape_{inSquareSize, inSquareSize},
            size_{inSquareSize * inSquareSize}
        {
            newArray();
        }

        Matrix(size_type inNumRows, size_type inNumCols) :
            shape_{inNumRows, inNumCols},
            size_{inNumRows * inNumCols}
        {
            newArray();
        }

        explicit Matrix(const Shape &inShape) :
            shape_{inShape},
            size_{shape_.size()}
        {
            newArray();
        }


        Matrix(std::initializer_list<dtype> inList) :
            shape_{1, static_cast<uint32>(inList.size())},
            size_{shape_.size()}
        {
            newArray();
            if (size_ > 0)
            {
                algo::copy(inList.begin(), inList.end(), begin());
            }
        }

        /// Constructor
        /// @param inList: 2D initializer list
        Matrix(const std::initializer_list<std::initializer_list<dtype>> &inList) :
            shape_{static_cast<uint32>(inList.size()), 0}
        {
            for (const auto &list : inList)
            {
                if (shape_.m_Cols == 0)
                {
                    shape_.m_Cols = static_cast<uint32>(list.size());
                }
                else if (list.size() != shape_.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "All rows of the initializer list needs to have the same number of elements");
                }
            }

            size_ = shape_.size();
            newArray();
            uint32 row = 0;
            for (const auto &list : inList)
            {
                const auto ptr = begin() += row * shape_.m_Cols;
                algo::copy(list.begin(), list.end(), ptr);
                ++row;
            }
        }


        /// Constructor
        /// @param inArray
        /// @param policy: the policy to use the pointer, copy or non-owning shell. default copy
        template <size_t ArraySize, std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
        Matrix(std::array<dtype, ArraySize> &inArray, PointerPolicy policy = PointerPolicy::COPY) :
            shape_{1, static_cast<uint32>(ArraySize)},
            size_{shape_.size()}
        {
            switch (policy)
            {
            case PointerPolicy::COPY:
            {
                newArray();
                if (size_ > 0)
                {
                    algo::copy(inArray.begin(), inArray.end(), begin());
                }
                break;
            }
            case PointerPolicy::SHELL:
            {
                array_ = inArray.data();
                ownsPtr_ = false;
                break;
            }
            default:
            {
                THROW_RUNTIME_ERROR("Unimplemented PointerPolicy type");
            }
            }
        }


        /// Constructor
        ///
        /// @param in2dArray
        /// @param policy: the policy to use the pointer, copy or non-owning shell. default copy
        ///
        template <size_t Dim0Size, size_t Dim1Size>
        Matrix(std::array<std::array<dtype, Dim1Size>, Dim0Size> &in2dArray,
               PointerPolicy policy = PointerPolicy::COPY) :
            shape_{static_cast<uint32>(Dim0Size), static_cast<uint32>(Dim1Size)},
            size_{shape_.size()}
        {
            switch (policy)
            {
            case PointerPolicy::COPY:
            {
                newArray();
                if (size_ > 0)
                {
                    const auto start = in2dArray.front().begin();
                    algo::copy(start, start + size_, begin());
                }
                break;
            }
            case PointerPolicy::SHELL:
            {
                array_ = in2dArray.front().data();
                ownsPtr_ = false;
                break;
            }
            default:
            {
                THROW_RUNTIME_ERROR("Unimplemented PointerPolicy type");
            }
            }
        }


        /// Constructor
        ///
        /// @param inVector
        /// @param policy: the policy to use the pointer, copy or non-owning shell. default copy
        ///
        template <std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
        Matrix(std::vector<dtype> &inVector, PointerPolicy policy = PointerPolicy::COPY) :
            shape_{1, static_cast<uint32>(inVector.size())},
            size_{shape_.size()}
        {
            switch (policy)
            {
            case PointerPolicy::COPY:
            {
                newArray();
                if (size_ > 0)
                {
                    algo::copy(inVector.begin(), inVector.end(), begin());
                }
                break;
            }
            case PointerPolicy::SHELL:
            {
                array_ = inVector.data();
                ownsPtr_ = false;
                break;
            }
            default:
            {
                THROW_RUNTIME_ERROR("Unimplemented PointerPolicy type");
            }
            }
        }


        /// Constructor
        ///
        /// @param in2dVector
        ///
        explicit Matrix(const std::vector<std::vector<dtype>> &in2dVector) :
            shape_{static_cast<uint32>(in2dVector.size()), 0}
        {
            for (const auto &row : in2dVector)
            {
                if (shape_.m_Cols == 0)
                {
                    shape_.m_Cols = static_cast<uint32>(row.size());
                }
                else if (row.size() != shape_.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the 2d vector need to have the same number of elements");
                }
            }

            size_ = shape_.size();

            newArray();
            auto currentPosition = begin();
            for (const auto &row : in2dVector)
            {
                algo::copy(row.begin(), row.end(), currentPosition);
                currentPosition += shape_.m_Cols;
            }
        }


        /// Constructor
        ///
        /// @param in2dArray
        /// @param policy: the policy to use the pointer, copy or non-owning shell. default copy
        ///
        template <size_t Dim1Size>
        Matrix(std::vector<std::array<dtype, Dim1Size>> &in2dArray, PointerPolicy policy = PointerPolicy::COPY) :
            shape_{static_cast<uint32>(in2dArray.size()), static_cast<uint32>(Dim1Size)},
            size_{shape_.size()}
        {
            switch (policy)
            {
            case PointerPolicy::COPY:
            {
                newArray();
                if (size_ > 0)
                {
                    const auto start = in2dArray.front().begin();
                    algo::copy(start, start + size_, begin());
                }
                break;
            }
            case PointerPolicy::SHELL:
            {
                array_ = in2dArray.front().data();
                ownsPtr_ = false;
                break;
            }
            default:
            {
                THROW_RUNTIME_ERROR("Unimplemented PointerPolicy type");
            }
            }
        }


        /// Constructor
        ///
        /// @param inDeque
        ///
        template <std::enable_if_t<is_valid_dtype_v<dtype>, int>  = 0>
        explicit Matrix(const std::deque<dtype> &inDeque) :
            shape_{1, static_cast<uint32>(inDeque.size())},
            size_{shape_.size()}
        {
            newArray();
            if (size_ > 0)
            {
                algo::copy(inDeque.begin(), inDeque.end(), begin());
            }
        }


        /// Constructor
        ///
        /// @param in2dDeque
        ///
        explicit Matrix(const std::deque<std::deque<dtype>> &in2dDeque) :
            shape_{static_cast<uint32>(in2dDeque.size()), 0}
        {
            for (const auto &row : in2dDeque)
            {
                if (shape_.m_Cols == 0)
                {
                    shape_.m_Cols = static_cast<uint32>(row.size());
                }
                else if (row.size() != shape_.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the 2d vector need to have the same number of elements");
                }
            }

            size_ = shape_.size();

            newArray();
            auto currentPosition = begin();
            for (const auto &row : in2dDeque)
            {
                algo::copy(row.begin(), row.end(), currentPosition);
                currentPosition += shape_.m_Cols;
            }
        }


        /// Constructor
        ///
        /// @param inList
        ///
        explicit Matrix(const std::list<dtype> &inList) :
            shape_{1, static_cast<uint32>(inList.size())},
            size_{shape_.size()}
        {
            newArray();
            if (size_ > 0)
            {
                algo::copy(inList.begin(), inList.end(), begin());
            }
        }


        /// Constructor
        ///
        /// @param inFirst
        /// @param inLast
        ///
        template <typename Iterator,
                  std::enable_if_t<std::is_same_v<typename std::iterator_traits<Iterator>::value_type, dtype>, int>  =
                      0>
        Matrix(Iterator inFirst, Iterator inLast) :
            shape_{1, static_cast<uint32>(std::distance(inFirst, inLast))},
            size_{shape_.size()}
        {
            newArray();
            if (size_ > 0)
            {
                algo::copy(inFirst, inLast, begin());
            }
        }


        /// Constructor. Copies the contents of the buffer into the array
        ///
        /// @param inPtr: pointer to the beginning of buffer
        /// @param size: number of elements in buffer
        ///
        template <typename UIntType,
                  std::enable_if_t<std::is_integral_v<UIntType> && !std::is_same_v<UIntType, bool>, int>  = 0>
        Matrix(const_pointer inPtr, UIntType size) :
            Matrix(inPtr, 1, size)
        {
        }


        /// Constructor. Copies the contents of the buffer into the array
        ///
        /// @param inPtr: pointer to beginning of buffer
        /// @param numRows: number of rows of the buffer
        /// @param numCols: number of cols of the buffer
        ///
        template <typename UIntType1,
                  typename UIntType2,
                  std::enable_if_t<std::is_integral_v<UIntType1> && !std::is_same_v<UIntType1, bool>, int>  = 0,
                  std::enable_if_t<std::is_integral_v<UIntType2> && !std::is_same_v<UIntType2, bool>, int>  = 0>
        Matrix(const_pointer inPtr, UIntType1 numRows, UIntType2 numCols) :
            shape_(numRows, numCols),
            size_{shape_.size()}
        {
            newArray();
            if (inPtr != nullptr && size_ > 0)
            {
                algo::copy(inPtr, inPtr + size_, begin());
            }
        }


        /// Constructor. Copies the contents of the buffer into the array, or acts as a
        ///              non-owning shell
        ///
        /// @param inPtr: pointer to beginning of buffer
        /// @param size: number of elements in buffer
        /// @param policy: the policy to use the pointer, copy or non-owning shell. default copy
        ///
        template <typename UIntType,
                  std::enable_if_t<std::is_integral_v<UIntType> && !std::is_same_v<UIntType, bool>, int>  = 0>
        Matrix(pointer inPtr, UIntType size, PointerPolicy policy) :
            Matrix(inPtr, 1, size, policy)
        {
        }


        /// Constructor. Copies the contents of the buffer into the array, or acts as a
        ///              non-owning shell
        ///
        /// @param inPtr: pointer to beginning of buffer
        /// @param numRows: number of rows of the buffer
        /// @param numCols: number of cols of the buffer
        /// @param policy: the policy to use the pointer, copy or non-owning shell
        ///
        template <typename UIntType1,
                  typename UIntType2,
                  std::enable_if_t<std::is_integral_v<UIntType1> && !std::is_same_v<UIntType1, bool>, int>  = 0,
                  std::enable_if_t<std::is_integral_v<UIntType2> && !std::is_same_v<UIntType2, bool>, int>  = 0>
        Matrix(pointer inPtr, UIntType1 numRows, UIntType2 numCols, PointerPolicy policy) :
            shape_(numRows, numCols),
            size_{shape_.size()}
        {
            switch (policy)
            {
            case PointerPolicy::COPY:
            {
                newArray();
                if (inPtr != nullptr && size_ > 0)
                {
                    algo::copy(inPtr, inPtr + size_, begin());
                }
                break;
            }
            case PointerPolicy::SHELL:
            {
                array_ = inPtr;
                ownsPtr_ = false;
                break;
            }
            default:
            {
                THROW_RUNTIME_ERROR("Unimplemented PointerPolicy type");
            }
            }
        }


        /// Copy Constructor
        ///
        /// @param inOtherArray
        ///
        Matrix(const self_type &inOtherArray) :
            shape_{inOtherArray.shape_},
            size_{inOtherArray.size_}
        {
            newArray();
            if (size_ > 0)
            {
                algo::copy(inOtherArray.cbegin(), inOtherArray.cend(), begin());
            }
        }


        /// Move Constructor
        ///
        /// @param inOtherArray
        ///
        Matrix(self_type &&inOtherArray) noexcept :
            shape_{inOtherArray.shape_},
            size_{inOtherArray.size_},
            array_{inOtherArray.array_},
            ownsPtr_{inOtherArray.ownsPtr_}
        {
            inOtherArray.shape_.m_Rows = inOtherArray.shape_.m_Cols = 0;
            inOtherArray.size_ = 0;
            inOtherArray.ownsPtr_ = false;
            inOtherArray.array_ = nullptr;
        }


        /// Destructor
        ///
        ~Matrix() noexcept
        {
            deleteArray();
        }

        void append(const Matrix<dtype> &inAppendValues, Axis inAxis = Axis::NONE)
        {
            if (shape().isnull())
            {
                newArray(inAppendValues.shape());
                algo::copy(inAppendValues.begin(), inAppendValues.end(), begin());
                return;
            }
            if (inAppendValues.shape().isnull())
            {
                return;
            }

            switch (inAxis)
            {
            case Axis::NONE:
            {
                algo::copy(inAppendValues.cbegin(), inAppendValues.cend(), end());
                break;
            }
            case Axis::ROW:
            {
                const Shape inShape = shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.m_Cols != appendShape.m_Cols)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                Matrix<dtype> returnArray(inShape.m_Rows + appendShape.m_Rows, inShape.m_Cols);
                algo::copy(cbegin(), cend(), returnArray.begin());
                algo::copy(inAppendValues.cbegin(),
                           inAppendValues.cend(),
                           returnArray.begin() + size());

                newArray(returnArray.shape_);
                algo::copy(returnArray.begin(), returnArray.end(), begin());
                break;
            }
            case Axis::COL:
            {
                const Shape inShape = shape();
                const Shape appendShape = inAppendValues.shape();
                if (inShape.m_Rows != appendShape.m_Rows)
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "all the input array dimensions except for the concatenation axis must match exactly");
                }

                Matrix<dtype> returnArray(inShape.m_Rows, inShape.m_Cols + appendShape.m_Cols);
                for (uint32 row = 0; row < returnArray.shape().m_Rows; ++row)
                {
                    algo::copy(cbegin(row), cend(row), returnArray.begin(row));
                    algo::copy(inAppendValues.cbegin(row),
                               inAppendValues.cend(row),
                               returnArray.begin(row) + inShape.m_Cols);
                }

                newArray(returnArray.shape_);
                algo::copy(returnArray.begin(), returnArray.end(), begin());
                break;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                break; // get rid of compiler warning
            }
            }
        }

        void clear()
        {
            newArray(shape_);
        }

        void reverse()
        {
            algo::reverse(begin(), end());
        }

        /// Cast to bool operator
        /// @returns bool false if empty, otherwise true
        ///
        explicit operator bool() const noexcept
        {
            return isempty();
        }


        /// Assignment operator, performs a deep copy
        ///
        /// @param rhs
        /// @return Matrix<dtype>
        ///
        self_type &operator=(const self_type &rhs)
        {
            if (&rhs != this)
            {
                if (rhs.size_ > 0)
                {
                    newArray(rhs.shape_);

                    algo::copy(rhs.cbegin(), rhs.cend(), begin());
                }
            }

            return *this;
        }


        /// Assignment operator, sets the entire array to a single
        /// scalar value.
        ///
        /// @param inValue
        /// @return Matrix<dtype>
        ///
        self_type &operator=(value_type inValue) noexcept
        {
            if (array_ != nullptr)
            {
                algo::fill(begin(), end(), inValue);
            }

            return *this;
        }


        /// Move operator, performs a deep move
        ///
        /// @param rhs
        /// @return Matrix<dtype>
        ///
        self_type &operator=(self_type &&rhs) noexcept
        {
            if (&rhs != this)
            {
                deleteArray();
                shape_ = rhs.shape_;
                size_ = rhs.size_;
                array_ = rhs.array_;
                ownsPtr_ = rhs.ownsPtr_;

                rhs.shape_.m_Rows = rhs.shape_.m_Cols = rhs.size_ = 0;
                rhs.array_ = nullptr;
                rhs.ownsPtr_ = false;
            }

            return *this;
        }


        /// 1D access operator with no bounds checking
        ///
        /// @param inIndex
        /// @return value
        ///
        reference operator[](index_type inIndex) noexcept
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->operator[](inIndex));
        }


        /// const 1D access operator with no bounds checking
        ///
        /// @param inIndex
        /// @return value
        ///
        [[nodiscard]] const_reference operator[](index_type inIndex) const noexcept
        {
            if (inIndex < 0)
            {
                inIndex += size_;
            }

            return array_[inIndex];
        }


        /// 2D access operator with no bounds checking
        ///
        /// @param inRowIndex
        /// @param inColIndex
        /// @return value
        ///
        reference operator()(index_type inRowIndex, index_type inColIndex) noexcept
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->operator()(inRowIndex, inColIndex));
        }


        /// const 2D access operator with no bounds checking
        ///
        /// @param inRowIndex
        /// @param inColIndex
        /// @return value
        ///
        [[nodiscard]] const_reference operator()(index_type inRowIndex, index_type inColIndex) const noexcept
        {
            if (inRowIndex < 0)
            {
                inRowIndex += shape_.m_Rows;
            }

            if (inColIndex < 0)
            {
                inColIndex += shape_.m_Cols;
            }

            return array_[inRowIndex * shape_.m_Cols + inColIndex];
        }


        /// 1D Slicing access operator with no bounds checking
        /// a returned array is of the range [start, stop).
        ///
        /// @param inSlice
        /// @return Matrix
        ///
        [[nodiscard]] self_type operator[](Slice inSlice) const
        {
            return operator[](toIndices(inSlice, Axis::NONE));
        }


        /// Returns the values from the input mask with no bounds checking
        ///
        /// @param inMask
        /// @return Matrix
        ///
        [[nodiscard]] self_type operator[](const Matrix<bool> &inMask) const
        {
            return operator[](inMask.flatnonzero());
        }


        /// Returns the values from the input indices with no bounds checking
        ///
        /// @param inIndices
        /// @return Matrix
        ///
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type operator[](const Indices &inIndices) const
        {
            auto outArray = self_type(1, static_cast<size_type>(inIndices.size()));
            size_type i = 0;
            for (auto &index : inIndices)
            {
                outArray[i++] = operator[](static_cast<index_type>(index));
            }

            return outArray;
        }


        /// 2D Slicing access operator with no bounds checking
        /// returned array is of the range [start, stop).
        ///
        /// @param inRowSlice
        /// @param inColSlice
        /// @return Matrix
        ///
        [[nodiscard]] self_type operator()(Slice inRowSlice, Slice inColSlice) const
        {
            return operator()(toIndices(inRowSlice, Axis::ROW), toIndices(inColSlice, Axis::COL));
        }


        /// 2D Slicing access operator with no bounds checking
        /// returned array is of the range [start, stop).
        ///
        /// @param inRowSlice
        /// @param inColIndex
        /// @return Matrix
        ///
        [[nodiscard]] self_type operator()(Slice inRowSlice, index_type inColIndex) const
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return operator()(toIndices(inRowSlice, Axis::ROW), colIndices);
        }


        /// 2D Slicing access operator with no bounds checking
        /// returned array is of the range [start, stop).
        ///
        /// @param inRowIndex
        /// @param inColSlice
        /// @return Matrix
        ///
        [[nodiscard]] self_type operator()(index_type inRowIndex, Slice inColSlice) const
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return operator()(rowIndices, toIndices(inColSlice, Axis::COL));
        }


        /// 2D index access operator with no bounds checking
        /// returned array is of the range.
        ///
        /// @param rowIndices
        /// @param colIndex
        /// @return Matrix
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type operator()(const Indices &rowIndices, index_type colIndex) const
        {
            const Matrix<index_type> colIndices = {colIndex};
            return operator()(rowIndices, colIndices);
        }


        /// 2D index access operator with no bounds checking
        /// returned array is of the range.
        ///
        /// @param rowIndices
        /// @param colSlice
        /// @return Matrix
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type operator()(const Indices &rowIndices, Slice colSlice) const
        {
            return operator()(rowIndices, toIndices(colSlice, Axis::COL));
        }


        /// 2D index access operator with no bounds checking
        /// returned array is of the range.
        ///
        /// @param rowIndex
        /// @param colIndices
        /// @return Matrix
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type operator()(index_type rowIndex, const Indices &colIndices) const
        {
            const Matrix<index_type> rowIndices = {rowIndex};
            return operator()(rowIndices, colIndices);
        }


        /// 2D index access operator with no bounds checking
        /// returned array is of the range.
        ///
        /// @param rowSlice
        /// @param colIndices
        /// @return Matrix
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type operator()(Slice rowSlice, const Indices &colIndices) const
        {
            return operator()(toIndices(rowSlice, Axis::ROW), colIndices);
        }


        /// 2D index access operator with no bounds checking
        /// returned array is of the range.
        ///
        /// @param rowIndices
        /// @param colIndices
        /// @return Matrix
        ///
        template <typename RowIndices,
                  typename ColIndices,
                  type_traits::matrix_int_concept<RowIndices>  = 0,
                  type_traits::matrix_int_concept<ColIndices>  = 0>
        [[nodiscard]] self_type operator()(const RowIndices &rowIndices, const ColIndices &colIndices) const
        {
            self_type returnArray(rowIndices.size(), colIndices.size());

            size_type rowCounter = 0;
            for (auto rowIter = rowIndices.begin(); rowIter != rowIndices.end(); ++rowIter)
            {
                size_type colCounter = 0;
                for (auto colIter = colIndices.begin(); colIter != colIndices.end(); ++colIter)
                {
                    returnArray(rowCounter, colCounter++) = operator()(*rowIter, *colIter);
                }

                ++rowCounter;
            }

            return returnArray;
        }


        /// Returns a Slice object for slicing a row to the end of
        /// array.
        ///
        /// @param inStartIdx (default 0)
        /// @param inStepSize (default 1)
        /// @return Slice
        ///
        [[nodiscard]] Slice cSlice(index_type inStartIdx = 0, size_type inStepSize = 1) const
        {
            return Slice(inStartIdx, shape_.m_Cols, inStepSize); // NOLINT(modernize-return-braced-init-list)
        }


        /// Returns a Slice object for slicing a column to the end
        /// of the array.
        ///
        /// @param inStartIdx (default 0)
        /// @param inStepSize (default 1)
        /// @return Slice
        ///
        [[nodiscard]] Slice rSlice(index_type inStartIdx = 0, size_type inStepSize = 1) const
        {
            return Slice(inStartIdx, shape_.m_Rows, inStepSize); // NOLINT(modernize-return-braced-init-list)
        }


        /// 1D access method with bounds checking
        ///
        /// @param inIndex
        /// @return value
        ///
        reference at(index_type inIndex)
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->at(inIndex));
        }


        /// const 1D access method with bounds checking
        ///
        /// @param inIndex
        /// @return value
        ///
        [[nodiscard]] const_reference at(index_type inIndex) const
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inIndex) > static_cast<int64>(size_ - 1))
            {
                std::string errStr = "Input index " + utils::num2str(inIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(size_) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator[](inIndex); // cppcheck-suppress returnTempReference
        }


        /// 2D access method with bounds checking
        ///
        /// @param inRowIndex
        /// @param inColIndex
        /// @return value
        ///
        reference at(index_type inRowIndex, index_type inColIndex)
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->at(inRowIndex, inColIndex));
        }


        /// const 2D access method with bounds checking
        ///
        /// @param inRowIndex
        /// @param inColIndex
        /// @return value
        ///
        [[nodiscard]] const_reference at(index_type inRowIndex, index_type inColIndex) const
        {
            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inRowIndex) > static_cast<index_type>(shape_.m_Rows - 1))
            {
                std::string errStr = "Row index " + utils::num2str(inRowIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.m_Rows) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            // this doesn't allow for calling the first element as -size_...
            // but why would you really want to do that anyway?
            if (std::abs(inColIndex) > static_cast<index_type>(shape_.m_Cols - 1))
            {
                std::string errStr = "Column index " + utils::num2str(inColIndex);
                errStr += " is out of bounds for array of size " + utils::num2str(shape_.m_Cols) + ".";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            return operator()(inRowIndex, inColIndex); // cppcheck-suppress returnTempReference
        }


        /// const 1D access method with bounds checking
        ///
        /// @param inSlice
        /// @return Ndarray
        ///
        [[nodiscard]] self_type at(const Slice &inSlice) const
        {
            return at(toIndices(inSlice, Axis::NONE));
        }


        /// const 1D access method with bounds checking
        ///
        /// @param inMask
        /// @return Ndarray
        ///
        [[nodiscard]] self_type at(const Matrix<bool> &inMask) const
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("Input mask must have the same dimensions as array.");
            }

            return operator[](inMask);
        }


        /// const 1D access method with bounds checking
        ///
        /// @param inIndices
        /// @return Ndarray
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type at(const Indices &inIndices) const
        {
            algo::for_each(inIndices.begin(),
                           inIndices.end(),
                           [this](auto index)
                           {
                               auto indexSigned = static_cast<index_type>(index);
                               if (indexSigned < 0)
                               {
                                   indexSigned += size_;
                               }

                               if (indexSigned < 0 || indexSigned > static_cast<index_type>(size_ - 1))
                               {
                                   THROW_INVALID_ARGUMENT_ERROR("Index exceeds matrix dimensions");
                               }
                           });

            return operator[](inIndices);
        }


        /// const 2D access method with bounds checking
        ///
        /// @param inRowSlice
        /// @param inColSlice
        /// @return Ndarray
        ///
        [[nodiscard]] self_type at(const Slice &inRowSlice, const Slice &inColSlice) const
        {
            return at(toIndices(inRowSlice, Axis::ROW), toIndices(inColSlice, Axis::COL));
        }


        /// const 2D access method with bounds checking
        ///
        /// @param inRowSlice
        /// @param inColIndex
        /// @return Ndarray
        ///
        [[nodiscard]] self_type at(const Slice &inRowSlice, index_type inColIndex) const
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return at(toIndices(inRowSlice, Axis::ROW), colIndices);
        }


        /// const 2D access method with bounds checking
        ///
        /// @param inRowIndex
        /// @param inColSlice
        /// @return Ndarray
        ///
        [[nodiscard]] self_type at(index_type inRowIndex, const Slice &inColSlice) const
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return at(rowIndices, toIndices(inColSlice, Axis::COL));
        }


        /// const 2D access method with bounds checking
        ///
        /// @param rowIndices
        /// @param colIndex
        /// @return Ndarray
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type at(const Indices &rowIndices, index_type colIndex) const
        {
            const Matrix<index_type> colIndices = {colIndex};
            return at(rowIndices, colIndices);
        }


        /// const 2D access method with bounds checking
        ///
        /// @param rowIndices
        /// @param colSlice
        /// @return Ndarray
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type at(const Indices &rowIndices, Slice colSlice) const
        {
            return at(rowIndices, toIndices(colSlice, Axis::COL));
        }


        /// const 2D access method with bounds checking
        ///
        /// @param rowIndex
        /// @param colIndices
        /// @return Ndarray
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type at(index_type rowIndex, const Indices &colIndices) const
        {
            const Matrix<index_type> rowIndices = {rowIndex};
            return at(rowIndices, colIndices);
        }


        /// const 2D access method with bounds checking
        ///
        /// @param rowSlice
        /// @param colIndices
        /// @return Ndarray
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        [[nodiscard]] self_type at(Slice rowSlice, const Indices &colIndices) const
        {
            return at(toIndices(rowSlice, Axis::ROW), colIndices);
        }


        /// const 2D access method with bounds checking
        ///
        /// @param rowIndices
        /// @param colIndices
        /// @return Ndarray
        ///
        template <typename RowIndices,
                  typename ColIndices,
                  type_traits::matrix_int_concept<RowIndices>  = 0,
                  type_traits::matrix_int_concept<ColIndices>  = 0>
        [[nodiscard]] self_type at(const RowIndices &rowIndices, const ColIndices &colIndices) const
        {
            algo::for_each(rowIndices.begin(),
                           rowIndices.end(),
                           [this](auto row)
                           {
                               auto rowSigned = static_cast<index_type>(row);
                               if (rowSigned < 0)
                               {
                                   rowSigned += shape_.m_Rows;
                               }

                               if (rowSigned < 0 || rowSigned > static_cast<index_type>(shape_.m_Rows - 1))
                               {
                                   THROW_INVALID_ARGUMENT_ERROR("Row index exceeds matrix dimensions");
                               }
                           });

            algo::for_each(colIndices.begin(),
                           colIndices.end(),
                           [this](auto col)
                           {
                               auto colSigned = static_cast<index_type>(col);
                               if (colSigned < 0)
                               {
                                   colSigned += shape_.m_Cols;
                               }

                               if (colSigned < 0 || colSigned > static_cast<index_type>(shape_.m_Cols - 1))
                               {
                                   THROW_INVALID_ARGUMENT_ERROR("Column index exceeds matrix dimensions");
                               }
                           });

            return operator()(rowIndices, colIndices);
        }


        /// iterator to the beginning of the flattened array
        /// @return iterator
        ///
        [[nodiscard]] iterator begin() noexcept
        {
            return iterator(array_);
        }


        /// iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return iterator
        ///
        [[nodiscard]] iterator begin(size_type inRow)
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return begin() += (inRow * shape_.m_Cols);
        }


        /// const iterator to the beginning of the flattened array
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator begin() const noexcept
        {
            return cbegin();
        }


        /// const iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator begin(size_type inRow) const
        {
            return cbegin(inRow);
        }


        /// const iterator to the beginning of the flattened array
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cbegin() const noexcept
        {
            return const_iterator(array_);
        }


        /// const iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cbegin(size_type inRow) const
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return cbegin() += (inRow * shape_.m_Cols);
        }


        /// column_iterator to the beginning of the flattened array
        /// @return column_iterator
        ///
        [[nodiscard]] column_iterator colbegin() noexcept
        {
            return column_iterator(array_, shape_.m_Rows, shape_.m_Cols);
        }


        /// column_iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return column_iterator
        ///
        [[nodiscard]] column_iterator colbegin(size_type inCol)
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return colbegin() += (inCol * shape_.m_Rows);
        }


        /// const column_iterator to the beginning of the flattened array
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator colbegin() const noexcept
        {
            return ccolbegin();
        }


        /// const column_iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator colbegin(size_type inCol) const
        {
            return ccolbegin(inCol);
        }


        /// const_column_iterator to the beginning of the flattened array
        ///
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator ccolbegin() const noexcept
        {
            return const_column_iterator(array_, shape_.m_Rows, shape_.m_Cols);
        }


        /// const_column_iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator ccolbegin(size_type inCol) const
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return ccolbegin() += (inCol * shape_.m_Rows);
        }


        /// reverse_iterator to the beginning of the flattened array
        /// @return reverse_iterator
        ///
        [[nodiscard]] reverse_iterator rbegin() noexcept
        {
            return reverse_iterator(end());
        }


        /// reverse_iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return reverse_iterator
        ///
        [[nodiscard]] reverse_iterator rbegin(size_type inRow)
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return rbegin() += (shape_.m_Rows - inRow - 1) * shape_.m_Cols;
        }


        /// const iterator to the beginning of the flattened array
        /// @return const_iterator
        ///
        [[nodiscard]] const_reverse_iterator rbegin() const noexcept
        {
            return crbegin();
        }


        /// const iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return const_iterator
        ///
        [[nodiscard]] const_reverse_iterator rbegin(size_type inRow) const
        {
            return crbegin(inRow);
        }


        /// const_reverse_iterator to the beginning of the flattened array
        ///
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator crbegin() const noexcept
        {
            return const_reverse_iterator(cend());
        }


        /// const_reverse_iterator to the beginning of the input row
        ///
        /// @param inRow
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator crbegin(size_type inRow) const
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return crbegin() += (shape_.m_Rows - inRow - 1) * shape_.m_Cols;
        }


        /// reverse_column_iterator to the beginning of the flattened array
        /// @return reverse_column_iterator
        ///
        [[nodiscard]] reverse_column_iterator rcolbegin() noexcept
        {
            return reverse_column_iterator(colend());
        }


        /// reverse_column_iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return reverse_column_iterator
        ///
        [[nodiscard]] reverse_column_iterator rcolbegin(size_type inCol)
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return rcolbegin() += (shape_.m_Cols - inCol - 1) * shape_.m_Rows;
        }


        /// const iterator to the beginning of the flattened array
        /// @return const_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator rcolbegin() const noexcept
        {
            return crcolbegin();
        }


        /// const iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return const_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator rcolbegin(size_type inCol) const
        {
            return crcolbegin(inCol);
        }


        /// const_reverse_column_iterator to the beginning of the flattened array
        ///
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator crcolbegin() const noexcept
        {
            return const_reverse_column_iterator(ccolend());
        }


        /// const_reverse_column_iterator to the beginning of the input column
        ///
        /// @param inCol
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator crcolbegin(size_type inCol) const
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return crcolbegin() += (shape_.m_Cols - inCol - 1) * shape_.m_Rows;
        }


        /// iterator to 1 past the end of the flattened array
        /// @return iterator
        ///
        [[nodiscard]] iterator end() noexcept
        {
            return begin() += size_;
        }


        /// iterator to the 1 past end of the row
        ///
        /// @param inRow
        /// @return iterator
        ///
        [[nodiscard]] iterator end(size_type inRow)
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return begin(inRow) += shape_.m_Cols;
        }


        /// const iterator to 1 past the end of the flattened array
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator end() const noexcept
        {
            return cend();
        }


        /// const iterator to the 1 past end of the row
        ///
        /// @param inRow
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator end(size_type inRow) const
        {
            return cend(inRow);
        }


        /// const iterator to 1 past the end of the flattened array
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cend() const noexcept
        {
            return cbegin() += size_;
        }


        /// const iterator to 1 past the end of the input row
        ///
        /// @param inRow
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cend(size_type inRow) const
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return cbegin(inRow) += shape_.m_Cols;
        }


        /// reverse_iterator to 1 past the end of the flattened array
        /// @return reverse_iterator
        ///
        [[nodiscard]] reverse_iterator rend() noexcept
        {
            return rbegin() += size_;
        }


        /// reverse_iterator to the 1 past end of the row
        ///
        /// @param inRow
        /// @return reverse_iterator
        ///
        [[nodiscard]] reverse_iterator rend(size_type inRow)
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return rbegin(inRow) += shape_.m_Cols;
        }


        /// const_reverse_iterator to 1 past the end of the flattened array
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator rend() const noexcept
        {
            return crend();
        }


        /// const_reverse_iterator to the 1 past end of the row
        ///
        /// @param inRow
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator rend(size_type inRow) const
        {
            return crend(inRow);
        }


        /// const_reverse_iterator to 1 past the end of the flattened array
        ///
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator crend() const noexcept
        {
            return crbegin() += size_;
        }


        /// const_reverse_iterator to 1 past the end of the input row
        ///
        /// @param inRow
        /// @return const_reverse_iterator
        ///
        [[nodiscard]] const_reverse_iterator crend(size_type inRow) const
        {
            if (inRow >= shape_.m_Rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("input row is greater than the number of rows in the array.");
            }

            return crbegin(inRow) += shape_.m_Cols;
        }


        /// column_iterator to 1 past the end of the flattened array
        /// @return column_iterator
        ///
        [[nodiscard]] column_iterator colend() noexcept
        {
            return colbegin() += size_;
        }


        /// column_iterator to the 1 past end of the column
        ///
        /// @param inCol
        /// @return column_iterator
        ///
        [[nodiscard]] column_iterator colend(size_type inCol)
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return colbegin(inCol) += shape_.m_Rows;
        }


        /// const column_iterator to 1 past the end of the flattened array
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator colend() const noexcept
        {
            return ccolend();
        }


        /// const column_iterator to the 1 past end of the column
        ///
        /// @param inCol
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator colend(size_type inCol) const
        {
            return ccolend(inCol);
        }


        /// const_column_iterator to 1 past the end of the flattened array
        ///
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator ccolend() const noexcept
        {
            return ccolbegin() += size_;
        }


        /// const_column_iterator to 1 past the end of the input col
        ///
        /// @param inCol
        /// @return const_column_iterator
        ///
        [[nodiscard]] const_column_iterator ccolend(size_type inCol) const
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return ccolbegin(inCol) += shape_.m_Rows;
        }


        /// reverse_column_iterator to 1 past the end of the flattened array
        /// @return reverse_column_iterator
        ///
        [[nodiscard]] reverse_column_iterator rcolend() noexcept
        {
            return rcolbegin() += size_;
        }


        /// reverse_column_iterator to the 1 past end of the column
        ///
        /// @param inCol
        /// @return reverse_column_iterator
        ///
        [[nodiscard]] reverse_column_iterator rcolend(size_type inCol)
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return rcolbegin(inCol) += shape_.m_Rows;
        }


        /// const_reverse_column_iterator to 1 past the end of the flattened array
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator rcolend() const noexcept
        {
            return crcolend();
        }


        /// const_reverse_column_iterator to the 1 past end of the column
        ///
        /// @param inCol
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator rcolend(size_type inCol) const
        {
            return crcolend(inCol);
        }


        /// const_reverse_column_iterator to 1 past the end of the flattened array
        ///
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator crcolend() const noexcept
        {
            return crcolbegin() += size_;
        }


        /// const_reverse_column_iterator to 1 past the end of the input col
        ///
        /// @param inCol
        /// @return const_reverse_column_iterator
        ///
        [[nodiscard]] const_reverse_column_iterator crcolend(size_type inCol) const
        {
            if (inCol >= shape_.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input col is greater than the number of cols in the array.");
            }

            return crcolbegin(inCol) += shape_.m_Rows;
        }


        /// Returns True if all elements evaluate to True or non zero
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.all.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<bool> all(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i) -> bool { return !utils::essentiallyEqual(i, dtype{0}); };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<bool> returnArray = {algo::all_of(cbegin(), cend(), function)};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<bool> returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = algo::all_of(cbegin(row), cend(row), function);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().all(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Returns True if any elements evaluate to True or non zero
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.any.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<bool> any(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i) -> bool { return !utils::essentiallyEqual(i, dtype{0}); };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<bool> returnArray = {algo::any_of(cbegin(), cend(), function)};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<bool> returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = algo::any_of(cbegin(row), cend(row), function);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().any(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return indices of the maximum values along the given axis.
        /// Only the first index is returned.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.argmax.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<size_type> argmax(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<size_type> returnArray = {static_cast<size_type>(
                    algo::max_element(cbegin(), cend(), comparitor) - cbegin())};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<size_type> returnArray(1, shape_.m_Rows);
                for (size_type row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = static_cast<size_type>(
                        algo::max_element(cbegin(row), cend(row), comparitor) - cbegin(row));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().argmax(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return indices of the minimum values along the given axis.
        /// Only the first index is returned.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.argmin.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<size_type> argmin(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<size_type> returnArray = {static_cast<size_type>(
                    algo::min_element(cbegin(), cend(), comparitor) - cbegin())};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<size_type> returnArray(1, shape_.m_Rows);
                for (size_type row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = static_cast<size_type>(
                        algo::min_element(cbegin(row), cend(row), comparitor) - cbegin(row));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().argmin(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Returns the indices that would sort this array.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.argsort.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<size_type> argsort(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                std::vector<size_type> idx(size_);
                std::iota(idx.begin(), idx.end(), 0);

                const auto function = [this](size_type i1, size_type i2) noexcept -> bool
                {
                    return (*this)[i1] < (*this)[i2];
                };

                algo::stable_sort(idx.begin(), idx.end(), function);
                return Matrix<size_type>(idx); // NOLINT(modernize-return-braced-init-list)
            }
            case Axis::COL:
            {
                Matrix<size_type> returnArray(shape_);
                std::vector<size_type> idx(shape_.m_Cols);

                for (index_type row = 0; row < static_cast<index_type>(shape_.m_Rows); ++row)
                {
                    std::iota(idx.begin(), idx.end(), 0);

                    const auto function = [this, row](size_type i1, size_type i2) noexcept -> bool
                    {
                        return operator()(row, i1) < operator()(row, i2);
                    };

                    algo::stable_sort(idx.begin(), idx.end(), function);

                    for (index_type col = 0; col < static_cast<index_type>(shape_.m_Cols); ++col)
                    {
                        returnArray(row, col) = idx[static_cast<size_type>(col)];
                    }
                }
                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().argsort(Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Returns a copy of the array, cast to a specified type.
        /// Arithmetic to Arithmetic
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.astype.html
        ///
        /// @return Matrix
        ///
        template <typename dtypeOut,
                  typename dtype_ = dtype,
                  std::enable_if_t<std::is_same_v<dtype_, dtype>, int>  = 0,
                  std::enable_if_t<std::is_arithmetic_v<dtype_>, int>  = 0,
                  std::enable_if_t<std::is_arithmetic_v<dtypeOut>, int>  = 0>
        [[nodiscard]] Matrix<dtypeOut> astype() const
        {
            if constexpr (std::is_same_v<dtypeOut, dtype>)
            {
                return *this;
            }
            else
            {
                Matrix<dtypeOut> outArray(shape_);
                algo::transform(cbegin(),
                                cend(),
                                outArray.begin(),
                                [](dtype value) -> dtypeOut { return static_cast<dtypeOut>(value); });

                return outArray;
            }
        }


        /// Returns a copy of the array, cast to a specified type.
        /// Arithmetic to Complex
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.astype.html
        ///
        /// @return Matrix
        ///
        template <typename dtypeOut,
                  typename dtype_ = dtype,
                  std::enable_if_t<std::is_same_v<dtype_, dtype>, int>  = 0,
                  std::enable_if_t<std::is_arithmetic_v<dtype_>, int>  = 0,
                  std::enable_if_t<is_complex_v<dtypeOut>, int>  = 0>
        [[nodiscard]] Matrix<dtypeOut> astype() const
        {
            Matrix<dtypeOut> outArray(shape_);

            const auto function = [](const_reference value) -> dtypeOut
            {
                return std::complex<typename dtypeOut::value_type>(value);
            };

            algo::transform(cbegin(), cend(), outArray.begin(), function);

            return outArray;
        }


        /// Returns a copy of the array, cast to a specified type.
        /// Complex to Complex
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.astype.html
        ///
        /// @return Matrix
        ///
        template <typename dtypeOut,
                  typename dtype_ = dtype,
                  std::enable_if_t<std::is_same_v<dtype_, dtype>, int>  = 0,
                  std::enable_if_t<is_complex_v<dtype_>, int>  = 0,
                  std::enable_if_t<is_complex_v<dtypeOut>, int>  = 0>
        [[nodiscard]] Matrix<dtypeOut> astype() const
        {
            if constexpr (std::is_same_v<dtypeOut, dtype>)
            {
                return *this;
            }
            else
            {
                const auto function = [](const_reference value) noexcept -> dtypeOut
                {
                    return complex_cast<typename dtypeOut::value_type>(value);
                };

                Matrix<dtypeOut> outArray(shape_);
                algo::transform(cbegin(), cend(), outArray.begin(), function);
                return outArray;
            }
        }


        /// Returns a copy of the array, cast to a specified type.
        /// Complex to Arithmetic
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.astype.html
        ///
        /// @return Matrix
        ///
        template <typename dtypeOut,
                  typename dtype_ = dtype,
                  std::enable_if_t<std::is_same_v<dtype_, dtype>, int>  = 0,
                  std::enable_if_t<is_complex_v<dtype_>, int>  = 0,
                  std::enable_if_t<std::is_arithmetic_v<dtypeOut>, int>  = 0>
        [[nodiscard]] Matrix<dtypeOut> astype() const
        {
            Matrix<dtypeOut> outArray(shape_);

            const auto function = [](const_reference value) -> dtypeOut { return static_cast<dtypeOut>(value.real()); };

            algo::transform(cbegin(), cend(), outArray.begin(), function);

            return outArray;
        }


        /// Returns a copy of the last element of the flattened array.
        ///
        /// @return dtype
        ///
        [[nodiscard]] const_reference back() const noexcept
        {
            return *(cend() - 1);
        }


        /// Returns a reference the last element of the flattened array.
        ///
        /// @return dtype
        ///
        reference back() noexcept
        {
            return *(end() - 1);
        }


        /// Returns a copy of the last element of the input row.
        ///
        /// @return dtype
        ///
        [[nodiscard]] const_reference back(size_type row) const
        {
            return *(cend(row) - 1);
        }


        /// Returns a reference the last element of the input row.
        ///
        /// @return dtype
        ///
        reference back(size_type row)
        {
            return *(end(row) - 1);
        }


        /// Returns an array whose values are limited to [min, max].
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.clip.html
        ///
        /// @param inMin: min value to clip to
        /// @param inMax: max value to clip to
        /// @return clipped value
        ///
        [[nodiscard]] self_type clip(value_type inMin, value_type inMax) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            self_type outArray(shape_);
            algo::transform(cbegin(),
                            cend(),
                            outArray.begin(),
                            [inMin, inMax](dtype value) noexcept -> dtype
                            {
#ifdef __cpp_lib_clamp
                                const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

                                return std::clamp(value, inMin, inMax, comparitor);
#else
                    if (value < inMin)
                    {
                        return inMin;
                    }
                    else if (value > inMax)
                    {
                        return inMax;
                    }

                    return value;
#endif
                            });

            return outArray;
        }


        /// Returns the full column of the array
        ///
        /// @param inColumn: the column index
        /// @return self_type
        ///
        [[nodiscard]] self_type column(size_type inColumn) const
        {
            return operator()(rSlice(), inColumn);
        }


        /// Returns the full column of the array
        ///
        /// @param inCols: the column indices
        /// @return self_type
        ///
        [[nodiscard]] self_type columns(const Matrix<size_type> &inCols) const
        {
            auto returnArray = self_type(shape_.m_Rows, inCols.size());
            const auto rSlice = returnArray.rSlice();

            for (size_type i = 0; i < inCols.size(); ++i)
            {
                returnArray.put(rSlice, i, column(inCols[i]));
            }

            return returnArray;
        }


        /// returns whether or not a value is included the array
        ///
        /// @param inValue
        /// @param inAxis (Optional, default NONE)
        /// @return bool
        ///
        [[nodiscard]] Matrix<bool> contains(value_type inValue, Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<bool> returnArray = {algo::find(cbegin(), cend(), inValue) != cend()};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<bool> returnArray(1, shape_.m_Rows);
                for (size_type row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = algo::find(cbegin(row), cend(row), inValue) != cend(row);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().contains(inValue, Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return a copy of the array
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.copy.html
        ///
        /// @return Matrix
        ///
        [[nodiscard]] self_type copy() const
        {
            return self_type(*this);
        }


        /// Return the cumulative product of the elements along the given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.cumprod.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type cumprod(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type returnArray(1, size_);
                returnArray[0] = front();
                for (size_type i = 1; i < size_; ++i)
                {
                    returnArray[i] = returnArray[i - 1] * array_[i];
                }

                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(shape_);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(row, 0) = operator()(row, 0);
                    for (uint32 col = 1; col < shape_.m_Cols; ++col)
                    {
                        returnArray(row, col) = returnArray(row, col - 1) * operator()(row, col);
                    }
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().cumprod(Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return the cumulative sum of the elements along the given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.cumsum.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type cumsum(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type returnArray(1, size_);
                returnArray[0] = front();
                for (size_type i = 1; i < size_; ++i)
                {
                    returnArray[i] = returnArray[i - 1] + array_[i];
                }

                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(shape_);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(row, 0) = operator()(row, 0);
                    for (uint32 col = 1; col < shape_.m_Cols; ++col)
                    {
                        returnArray(row, col) = returnArray(row, col - 1) + operator()(row, col);
                    }
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().cumsum(Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Returns the raw pointer to the underlying data
        /// @return pointer
        ///
        [[nodiscard]] pointer data() noexcept
        {
            return array_;
        }


        /// Returns the raw pointer to the underlying data
        /// @return const_pointer
        ///
        [[nodiscard]] const_pointer data() const noexcept
        {
            return array_;
        }


        /// Releases the internal data pointer so that the destructor
        /// will not call delete on it, and returns the raw pointer
        /// to the underlying data.
        /// @return pointer
        ///
        [[nodiscard]] pointer dataRelease() noexcept
        {
            ownsPtr_ = false;
            return data();
        }


        /// Return specified diagonals.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.diagonal.html
        ///
        /// @param inOffset: Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults
        /// to 0.
        /// @param inAxis: (Optional, default ROW) axis the offset is applied to
        /// @return Matrix
        ///
        [[nodiscard]] self_type diagonal(index_type inOffset = 0, Axis inAxis = Axis::ROW) const
        {
            switch (inAxis)
            {
            case Axis::COL:
            {
                std::vector<dtype> diagnolValues;
                size_type col = 0;
                for (index_type row = inOffset; row < static_cast<index_type>(shape_.m_Rows); ++row)
                {
                    if (row < 0)
                    {
                        ++col;
                        continue;
                    }
                    if (col >= shape_.m_Cols)
                    {
                        break;
                    }

                    diagnolValues.push_back(operator()(static_cast<size_type>(row), col));
                    ++col;
                }

                return self_type(diagnolValues);
            }
            case Axis::ROW:
            {
                return transpose().diagonal(inOffset, Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return size of the axis dimension
        ///
        /// @param inAxis: the array axis
        /// @return size of the dimension
        ///
        [[nodiscard]] size_type dimSize(Axis inAxis) const noexcept
        {
            switch (inAxis)
            {
            case Axis::NONE:
            {
                return size();
            }
            case Axis::ROW:
            {
                return numRows();
            }
            case Axis::COL:
            {
                return numCols();
            }
            default:
            {
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Dot product of two arrays.
        ///
        /// For 2-D arrays it is equivalent to matrix multiplication,
        /// and for 1-D arrays to inner product of vectors.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.dot.html
        ///
        /// @param inOtherArray
        /// @return dot product
        ///
        [[nodiscard]] self_type dot(const self_type &inOtherArray) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            if (shape_ == inOtherArray.shape_ && (shape_.m_Rows == 1 || shape_.m_Cols == 1))
            {
                dtype dotProduct = std::inner_product(cbegin(), cend(), inOtherArray.cbegin(), dtype{0});
                self_type returnArray = {dotProduct};
                return returnArray;
            }
            if (shape_.m_Cols == inOtherArray.shape_.m_Rows)
            {
                // 2D array, use matrix multiplication
                self_type returnArray(shape_.m_Rows, inOtherArray.shape_.m_Cols);
                auto otherArrayT = inOtherArray.transpose();

                for (uint32 i = 0; i < shape_.m_Rows; ++i)
                {
                    for (uint32 j = 0; j < otherArrayT.shape_.m_Rows; ++j)
                    {
                        returnArray(i, j) =
                            std::inner_product(otherArrayT.cbegin(j), otherArrayT.cend(j), cbegin(i), dtype{0});
                    }
                }

                return returnArray;
            }

            std::string errStr = "shapes of [" + utils::num2str(shape_.m_Rows) + ", " + utils::num2str(shape_.m_Cols) +
                "]";
            errStr += " and [" + utils::num2str(inOtherArray.shape_.m_Rows) + ", " +
                utils::num2str(inOtherArray.shape_.m_Cols) + "]";
            errStr += " are not consistent.";
            THROW_INVALID_ARGUMENT_ERROR(errStr);

            return self_type(); // get rid of compiler warning
        }


        /// Dump a binary file of the array to the specified file.
        /// The array can be read back with mc::load.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.dump.html
        ///
        /// @param inFilename
        ///
        void dump(const std::string &inFilename) const
        {
            std::filesystem::path f(inFilename);
            if (!f.has_extension())
            {
                f.replace_extension("bin");
            }

            std::ofstream ofile(f.c_str(), std::ios::binary);
            if (!ofile.good())
            {
                THROW_RUNTIME_ERROR("Unable to open the input file:\n\t" + inFilename);
            }

            if (array_ != nullptr)
            {
                ofile.write(reinterpret_cast<const char *>(array_), size_ * sizeof(dtype));
            }
            ofile.close();
        }


        /// Fill the array with a scalar value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.fill.html
        ///
        /// @param inFillValue
        /// @return None
        ///
        self_type &fill(value_type inFillValue) noexcept
        {
            algo::fill(begin(), end(), inFillValue);
            return *this;
        }


        /// Return the indices of the flattened array of the
        /// elements that are non-zero.
        ///
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<size_type> flatnonzero() const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::vector<size_type> indices;
            size_type idx = 0;
            for (auto value : *this)
            {
                if (!utils::essentiallyEqual(value, dtype{0}))
                {
                    indices.push_back(idx);
                }
                ++idx;
            }

            return Matrix<size_type>(indices);
        }


        /// Return a copy of the array collapsed into one dimension.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.flatten.html
        ///
        /// @return Matrix
        ///
        [[nodiscard]] self_type flatten() const
        {
            self_type outArray(1, size_);
            algo::copy(cbegin(), cend(), outArray.begin());
            return outArray;
        }


        /// Returns a copy of the first element of the flattened array.
        ///
        /// @return dtype
        ///
        [[nodiscard]] const_reference front() const noexcept
        {
            return *cbegin();
        }


        /// Returns a reference to the first element of the flattened array.
        ///
        /// @return dtype
        ///
        reference front() noexcept
        {
            return *begin();
        }


        /// Returns a copy of the first element of the input row.
        ///
        /// @return dtype
        ///
        [[nodiscard]] const_reference front(size_type row) const
        {
            return *cbegin(row);
        }


        /// Returns a reference to the first element of the input row.
        ///
        /// @return dtype
        ///
        reference front(size_type row)
        {
            return *begin(row);
        }


        /// Returns a new flat array with the givin flat input indices.
        ///
        /// @param inIndices
        /// @return values
        ///
        [[nodiscard]] self_type getByIndices(const Matrix<size_type> &inIndices) const
        {
            return operator[](inIndices);
        }


        /// Takes in a boolean mask the same size as the array
        /// and returns a flattened array with the values cooresponding
        /// to the input mask.
        ///
        /// @param inMask
        /// @return values
        ///
        [[nodiscard]] self_type getByMask(const Matrix<bool> &inMask) const
        {
            return operator[](inMask);
        }


        /// Return if the Matrix is empty. ie the default constructor
        /// was used.
        ///
        /// @return boolean
        ///
        // NOLINTNEXTLINE(modernize-use-nodiscard)
        bool isempty() const noexcept
        {
            return size_ == 0;
        }


        /// Return if the Matrix is flat. ie the number of columns or
        /// rows is equal to one.
        ///
        /// @return boolean
        ///
        // NOLINTNEXTLINE(modernize-use-nodiscard)
        bool isflat() const noexcept
        {
            return !isscalar() && (shape_.m_Rows == 1 || shape_.m_Cols == 1);
        }


        /// Return if the Matrix is scalar
        ///
        /// @return boolean
        // NOLINTNEXTLINE(modernize-use-nodiscard)
        bool isscalar() const noexcept
        {
            return size_ == 1;
        }


        /// Return if the Matrix is sorted.
        ///
        /// @param inAxis
        /// @return boolean
        ///
        [[nodiscard]] Matrix<bool> issorted(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                return {algo::is_sorted(cbegin(), cend(), comparitor)};
            }
            case Axis::COL:
            {
                Matrix<bool> returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = algo::is_sorted(cbegin(row), cend(row), comparitor);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().issorted(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return if the Matrix is square.
        ///
        /// @return boolean
        ///
        // NOLINTNEXTLINE(modernize-use-nodiscard)
        bool issquare() const noexcept
        {
            return shape_.issquare();
        }


        /// Copy an element of an array to a standard C++ scalar and return it.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.item.html
        ///
        /// @return array element
        ///
        [[nodiscard]] value_type item() const
        {
            if (!isscalar())
            {
                THROW_INVALID_ARGUMENT_ERROR("Can only convert an array of size 1 to a C++ scalar");
            }

            return front();
        }


        /// Return the maximum along a given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.max.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type max(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type returnArray = {*algo::max_element(cbegin(), cend(), comparitor)};
                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = *algo::max_element(cbegin(row), cend(row), comparitor);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().max(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return the minimum along a given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.min.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type min(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type returnArray = {*algo::min_element(cbegin(), cend(), comparitor)};
                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = *algo::min_element(cbegin(row), cend(row), comparitor);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().min(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return the median along a given axis.
        /// If the dtype is floating point then the middle elements will be
        /// averaged for arrays of even number of elements.
        /// If the dtype is integral then the middle elements will be intager
        /// averaged (rounded down to integer) for arrays of even number of elements.
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type median(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            if (size_ == 0)
            {
                THROW_RUNTIME_ERROR("Median is undefined for an array of size = 0.");
            }

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type copyArray(*this);

                const size_type middleIdx = size_ / 2; // integer division
                algo::nth_element(copyArray.begin(),
                                  copyArray.begin() + middleIdx,
                                  copyArray.end(),
                                  comparitor);

                dtype medianValue = copyArray.array_[middleIdx];
                if (size_ % 2 == 0)
                {
                    const size_type lhsIndex = middleIdx - 1;
                    algo::nth_element(copyArray.begin(),
                                      copyArray.begin() + lhsIndex,
                                      copyArray.end(),
                                      comparitor);
                    medianValue =
                        (medianValue + copyArray.array_[lhsIndex]) / dtype{2}; // potentially integer division, ok
                }

                return {medianValue};
            }
            case Axis::COL:
            {
                self_type copyArray(*this);
                self_type returnArray(1, shape_.m_Rows);

                const bool isEven = shape_.m_Cols % 2 == 0;
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    const uint32 middleIdx = shape_.m_Cols / 2; // integer division
                    algo::nth_element(copyArray.begin(row),
                                      copyArray.begin(row) + middleIdx,
                                      copyArray.end(row),
                                      comparitor);

                    dtype medianValue = copyArray(row, middleIdx);
                    if (isEven)
                    {
                        const size_type lhsIndex = middleIdx - 1;
                        algo::nth_element(copyArray.begin(row),
                                          copyArray.begin(row) + lhsIndex,
                                          copyArray.end(row),
                                          comparitor);
                        medianValue = (medianValue + copyArray(row, lhsIndex)) /
                            dtype{2}; // potentially integer division, ok
                    }

                    returnArray(0, row) = medianValue;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().median(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Fills the array with nans.
        ///
        ///
        self_type &nans() noexcept

        {
            STATIC_ASSERT_FLOAT(dtype);

            fill(consts::nan);
            return *this;
        }


        /// Returns the number of bytes held by the array
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.nbytes.html
        ///
        /// @return number of bytes
        ///
        [[nodiscard]] uint64 nbytes() const noexcept
        {
            return sizeof(dtype) * size_;
        }

        /// Returns True if none elements evaluate to True or non zero
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.any.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] Matrix<bool> none(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto function = [](dtype i) -> bool { return !utils::essentiallyEqual(i, dtype{0}); };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                Matrix<bool> returnArray = {algo::none_of(cbegin(), cend(), function)};
                return returnArray;
            }
            case Axis::COL:
            {
                Matrix<bool> returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = algo::none_of(cbegin(row), cend(row), function);
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().none(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Return the row/col indices of the array of the
        /// elements that are non-zero.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.nonzero.html
        ///
        /// @return std::pair<Matrix, Matrix> where first is the row indices and second is the
        /// column indices
        ///
        [[nodiscard]] std::pair<Matrix<size_type>, Matrix<size_type>> nonzero() const;


        /// Returns the number of columns in the array
        ///
        ///
        /// @return size_type
        ///
        [[nodiscard]] size_type numCols() const noexcept
        {
            return shape_.m_Cols;
        }


        /// Returns the number of rows in the array
        ///
        ///
        /// @return size_type
        ///
        [[nodiscard]] size_type numRows() const noexcept
        {
            return shape_.m_Rows;
        }


        /// Fills the array with ones
        ///
        ///
        self_type &ones() noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            fill(dtype{1});
            return *this;
        }


        /// Returns whether or not the array object owns the underlying data
        ///
        /// @return bool
        ///
        bool ownsInternalData() noexcept
        {
            return ownsPtr_;
        }


        /// Rearranges the elements in the array in such a way that
        /// value of the element in kth position is in the position it
        /// would be in a sorted array. All elements smaller than the kth
        /// element are moved before this element and all equal or greater
        /// are moved behind it. The ordering of the elements in the two
        /// partitions is undefined.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.partition.html
        ///
        /// @param inKth: kth element
        /// @param inAxis (Optional, default NONE)
        /// @return None
        ///
        self_type &partition(size_type inKth, Axis inAxis = Axis::NONE)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };
            // cppcheck-suppress returnTempReference

            switch (inAxis)
            {
            case Axis::NONE:
            {
                if (inKth >= size_)
                {
                    std::string errStr = "kth(=" + utils::num2str(inKth);
                    errStr += ") out of bounds (" + utils::num2str(size_) + ")";
                    THROW_INVALID_ARGUMENT_ERROR(errStr);
                }

                algo::nth_element(begin(), begin() + inKth, end(), comparitor);
                break;
            }
            case Axis::COL:
            {
                if (inKth >= shape_.m_Cols)
                {
                    std::string errStr = "kth(=" + utils::num2str(inKth);
                    errStr += ") out of bounds (" + utils::num2str(shape_.m_Cols) + ")";
                    THROW_INVALID_ARGUMENT_ERROR(errStr);
                }

                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    algo::nth_element(begin(row), begin(row) + inKth, end(row), comparitor);
                }
                break;
            }
            case Axis::ROW:
            {
                if (inKth >= shape_.m_Rows)
                {
                    std::string errStr = "kth(=" + utils::num2str(inKth);
                    errStr += ") out of bounds (" + utils::num2str(shape_.m_Rows) + ")";
                    THROW_INVALID_ARGUMENT_ERROR(errStr);
                }

                self_type transposedArray = transpose();
                for (uint32 row = 0; row < transposedArray.shape_.m_Rows; ++row)
                {
                    algo::nth_element(transposedArray.begin(row),
                                      transposedArray.begin(row) + inKth,
                                      transposedArray.end(row),
                                      comparitor);
                }
                *this = transposedArray.transpose();
                break;
            }
            }

            return *this;
        }


        /// Prints the array to the console.
        ///
        ///
        void print() const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::cout << *this;
        }


        /// Return the product of the array elements over the given axis
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.prod.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type prod(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                dtype product = std::accumulate(cbegin(), cend(), dtype{1}, std::multiplies<dtype>());
                self_type returnArray = {product};
                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) =
                        std::accumulate(cbegin(row), cend(row), dtype{1}, std::multiplies<dtype>());
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().prod(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Peak to peak (maximum - minimum) value along a given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.ptp.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type ptp(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };

            switch (inAxis)
            {
            case Axis::NONE:
            {
                const auto result = algo::minmax_element(cbegin(), cend(), comparitor);
                self_type returnArray = {*result.second - *result.first};
                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    const auto result = algo::minmax_element(cbegin(row), cend(row), comparitor);
                    returnArray(0, row) = *result.second - *result.first;
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().ptp(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// set the flat index element to the value
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inIndex
        /// @param inValue
        ///
        self_type &put(index_type inIndex, const value_type &inValue)
        {
            at(inIndex) = inValue;

            return *this;
        }


        /// set the 2D row/col index element to the value
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRow
        /// @param inCol
        /// @param inValue
        ///
        self_type &put(index_type inRow, index_type inCol, const value_type &inValue)
        {
            at(inRow, inCol) = inValue;

            return *this;
        }


        /// Set a.flat[n] = values for all n in indices.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inIndices
        /// @param inValue
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(const Indices &inIndices, const value_type &inValue)
        {
            for (auto index : inIndices)
            {
                put(index, inValue);
            }

            return *this;
        }


        /// Set a.flat[n] = values[n] for all n in indices.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inIndices
        /// @param inValues
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(const Indices &inIndices, const self_type &inValues)
        {
            if (inValues.isscalar())
            {
                return put(inIndices, inValues.item());
            }
            else if (inIndices.size() != inValues.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("Input indices do not match values dimensions.");
            }

            size_type counter = 0;
            for (auto index : inIndices)
            {
                put(index, inValues[counter++]);
            }

            return *this;
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inSlice
        /// @param inValue
        /// @return reference to self
        ///
        self_type &put(const Slice &inSlice, const value_type &inValue)
        {
            return put(toIndices(inSlice, Axis::NONE), inValue);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inSlice
        /// @param inValues
        /// @return reference to self
        ///
        self_type &put(const Slice &inSlice, const self_type &inValues)
        {
            return put(toIndices(inSlice, Axis::NONE), inValues);
        }


        /// Set the slice indices to the input value.
        /// @param inRowIndices
        /// @param inColIndices
        /// @param inValue
        /// @return reference to self
        ///
        template <typename RowIndices,
                  typename ColIndices,
                  type_traits::matrix_int_concept<RowIndices>  = 0,
                  type_traits::matrix_int_concept<ColIndices>  = 0>
        self_type &put(const RowIndices &inRowIndices, const ColIndices &inColIndices, const value_type &inValue)
        {
            algo::for_each(inRowIndices.begin(),
                           inRowIndices.end(),
                           [this, &inColIndices, &inValue](const auto row)
                           {
                               algo::for_each(inColIndices.begin(),
                                              inColIndices.end(),
                                              [this, row, &inValue](const auto col)
                                              {
                                                  this->put(row, col, inValue);
                                              });
                           });

            return *this;
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndices
        /// @param inColSlice
        /// @param inValue
        /// @return reference to self
        ///
        template <typename RowIndices, type_traits::matrix_int_concept<RowIndices>  = 0>
        self_type &put(const RowIndices &inRowIndices, const Slice &inColSlice, const value_type &inValue)
        {
            return put(inRowIndices, toIndices(inColSlice, Axis::COL), inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColIndices
        /// @param inValue
        /// @return reference to self
        ///
        template <typename ColIndices, type_traits::matrix_int_concept<ColIndices>  = 0>
        self_type &put(const Slice &inRowSlice, const ColIndices &inColIndices, const value_type &inValue)
        {
            return put(toIndices(inRowSlice, Axis::ROW), inColIndices, inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColSlice
        /// @param inValue
        /// @return reference to self
        ///
        self_type &put(const Slice &inRowSlice, const Slice &inColSlice, const value_type &inValue)
        {
            return put(toIndices(inRowSlice, Axis::ROW), toIndices(inColSlice, Axis::COL), inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndices
        /// @param inColIndex
        /// @param inValue
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(const Indices &inRowIndices, index_type inColIndex, const value_type &inValue)
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return put(inRowIndices, colIndices, inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColIndex
        /// @param inValue
        /// @return reference to self
        ///
        self_type &put(const Slice &inRowSlice, index_type inColIndex, const value_type &inValue)
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return put(toIndices(inRowSlice, Axis::ROW), colIndices, inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndex
        /// @param inColIndices
        /// @param inValue
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(index_type inRowIndex, const Indices &inColIndices, const value_type &inValue)
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return put(rowIndices, inColIndices, inValue);
        }


        /// Set the slice indices to the input value.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndex
        /// @param inColSlice
        /// @param inValue
        /// @return reference to self
        ///
        self_type &put(index_type inRowIndex, const Slice &inColSlice, const value_type &inValue)
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return put(rowIndices, toIndices(inColSlice, Axis::COL), inValue);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndices
        /// @param inColIndices
        /// @param inValues
        /// @return reference to self
        ///
        template <typename RowIndices,
                  typename ColIndices,
                  type_traits::matrix_int_concept<RowIndices>  = 0,
                  type_traits::matrix_int_concept<ColIndices>  = 0>
        self_type &put(const RowIndices &inRowIndices, const ColIndices &inColIndices, const self_type &inValues)
        {
            std::vector<size_type> indices;
            indices.reserve(inRowIndices.size() * inColIndices.size());
            std::for_each(inRowIndices.begin(),
                          inRowIndices.end(),
                          [this, &inColIndices, &indices](auto row)
                          {
                              if constexpr (std::is_signed_v<decltype(row)>)
                              {
                                  if (row < 0)
                                  {
                                      row += shape_.m_Rows;
                                  }
                                  // still
                                  if (row < 0)
                                  {
                                      THROW_INVALID_ARGUMENT_ERROR("row index exceeds matrix dimensions");
                                  }
                              }
                              std::for_each(inColIndices.begin(),
                                            inColIndices.end(),
                                            [this, row, &indices](auto col)
                                            {
                                                if constexpr (std::is_signed_v<decltype(col)>)
                                                {
                                                    if (col < 0)
                                                    {
                                                        col += shape_.m_Cols;
                                                    }
                                                    // still
                                                    if (col < 0)
                                                    {
                                                        THROW_INVALID_ARGUMENT_ERROR(
                                                            "col index exceeds matrix dimensions");
                                                    }
                                                }
                                                indices.push_back(row * shape_.m_Cols + col);
                                            });
                          });

            return put(Matrix<size_type>(indices.data(), indices.size(), PointerPolicy::SHELL), inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndices
        /// @param inColSlice
        /// @param inValues
        /// @return reference to self
        ///
        template <typename RowIndices, type_traits::matrix_int_concept<RowIndices>  = 0>
        self_type &put(const RowIndices &inRowIndices, Slice inColSlice, const self_type &inValues)
        {
            return put(inRowIndices, toIndices(inColSlice, Axis::COL), inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColIndices
        /// @param inValues
        /// @return reference to self
        ///
        template <typename ColIndices, type_traits::matrix_int_concept<ColIndices>  = 0>
        self_type &put(Slice inRowSlice, const ColIndices &inColIndices, const self_type &inValues)
        {
            return put(toIndices(inRowSlice, Axis::ROW), inColIndices, inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColSlice
        /// @param inValues
        /// @return reference to self
        ///
        self_type &put(Slice inRowSlice, Slice inColSlice, const self_type &inValues)
        {
            return put(toIndices(inRowSlice, Axis::ROW), toIndices(inColSlice, Axis::COL), inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndices
        /// @param inColIndex
        /// @param inValues
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(const Indices &inRowIndices, index_type inColIndex, const self_type &inValues)
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return put(inRowIndices, colIndices, inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowSlice
        /// @param inColIndex
        /// @param inValues
        /// @return reference to self
        ///
        self_type &put(const Slice &inRowSlice, index_type inColIndex, const self_type &inValues)
        {
            const Matrix<index_type> colIndices = {inColIndex};
            return put(toIndices(inRowSlice, Axis::ROW), colIndices, inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndex
        /// @param inColIndices
        /// @param inValues
        /// @return reference to self
        ///
        template <typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
        self_type &put(index_type inRowIndex, const Indices &inColIndices, const self_type &inValues)
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return put(rowIndices, inColIndices, inValues);
        }


        /// Set the slice indices to the input values.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.put.html
        ///
        /// @param inRowIndex
        /// @param inColSlice
        /// @param inValues
        /// @return reference to self
        ///
        self_type &put(index_type inRowIndex, const Slice &inColSlice, const self_type &inValues)
        {
            const Matrix<index_type> rowIndices = {inRowIndex};
            return put(rowIndices, toIndices(inColSlice, Axis::COL), inValues);
        }


        /// Set the mask indices to the input value.
        ///
        /// @param inMask
        /// @param inValue
        ///
        self_type &putMask(const Matrix<bool> &inMask, const value_type &inValue)
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the array it is masking.");
            }

            return put(inMask.flatnonzero(), inValue);
        }


        /// Set the mask indices to the input values.
        ///
        /// @param inMask
        /// @param inValues
        ///
        self_type &putMask(const Matrix<bool> &inMask, const self_type &inValues)
        {
            if (inMask.shape() != shape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the array it is masking.");
            }

            if (inValues.isscalar())
            {
                put(inMask.flatnonzero(), inValues.item());
            }
            else
            {
                put(inMask.flatnonzero(), inValues);
            }

            return *this;
        }


        /// Flattens the array but does not make a copy.
        ///
        /// Numpy Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
        ///
        /// @return Matrix
        ///
        self_type &ravel()
        {
            reshape(size_);
            return *this;
        }


        /// Repeat elements of an array.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.repeat.html
        ///
        /// @param inNumRows
        /// @param inNumCols
        /// @return Matrix
        ///
        [[nodiscard]] self_type repeat(size_type inNumRows, size_type inNumCols) const
        {
            self_type returnArray(shape_.m_Rows * inNumRows, shape_.m_Cols * inNumCols);

            for (size_type row = 0; row < inNumRows; ++row)
            {
                for (size_type col = 0; col < inNumCols; ++col)
                {
                    std::vector<size_type> indices(shape_.size());

                    const size_type rowStart = row * shape_.m_Rows;
                    const size_type colStart = col * shape_.m_Cols;

                    const size_type rowEnd = (row + 1) * shape_.m_Rows;
                    const size_type colEnd = (col + 1) * shape_.m_Cols;

                    size_type counter = 0;
                    for (size_type rowIdx = rowStart; rowIdx < rowEnd; ++rowIdx)
                    {
                        for (size_type colIdx = colStart; colIdx < colEnd; ++colIdx)
                        {
                            indices[counter++] = rowIdx * returnArray.shape_.m_Cols + colIdx;
                        }
                    }

                    returnArray.put(Matrix(indices), *this);
                }
            }

            return returnArray;
        }


        /// Repeat elements of an array.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.repeat.html
        ///
        /// @param inRepeatShape
        /// @return Matrix
        ///
        [[nodiscard]] self_type repeat(const Shape &inRepeatShape) const
        {
            return repeat(inRepeatShape.m_Rows, inRepeatShape.m_Cols);
        }


        /// Replaces a value of the array with another value
        ///
        /// @param oldValue: the value to replace
        /// @param newValue: the value to replace with
        ///
        self_type &replace(value_type oldValue, value_type newValue)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            algo::replace(begin(), end(), oldValue, newValue);
            return *this;
        }


        /// The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension
        /// can be -1. In this case, the value is inferred from the length of the
        /// array and remaining dimensions.
        ///
        /// NOTE: array shape is modified. If a copy is desired then the user should call the copy method:
        /// auto aNew = a.reshape(newSize).copy()
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.reshape.html
        ///
        /// @param inSize
        /// @returns reference to self
        ///
        self_type &reshape(size_type inSize)
        {
            if (inSize != size_)
            {
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into shape ";
                errStr += "[" + utils::num2str(1) + ", " + utils::num2str(inSize) + "]";
                THROW_RUNTIME_ERROR(errStr);
            }

            shape_.m_Rows = 1;
            shape_.m_Cols = inSize;

            return *this;
        }


        /// The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension
        /// can be -1. In this case, the value is inferred from the length of the
        /// array and remaining dimensions.
        ///
        /// NOTE: array shape is modified. If a copy is desired then the user should call the copy method:
        /// auto aNew = a.reshape(newRows, newCols).copy()
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.reshape.html
        ///
        /// @param inNumRows
        /// @param inNumCols
        /// @returns reference to self
        ///
        self_type &reshape(index_type inNumRows, index_type inNumCols)
        {
            if (inNumRows < 0)
            {
                if (size_ % inNumCols == 0)
                {
                    return reshape(size_ / inNumCols, inNumCols);
                }

                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                errStr += "with " + utils::num2str(inNumCols) + " columns";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            if (inNumCols < 0)
            {
                if (size_ % inNumRows == 0)
                {
                    return reshape(inNumRows, size_ / inNumRows);
                }

                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into a shape ";
                errStr += "with " + utils::num2str(inNumRows) + " rows";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            if (static_cast<size_type>(inNumRows * inNumCols) != size_)
            {
                std::string errStr = "Cannot reshape array of size " + utils::num2str(size_) + " into shape ";
                errStr += "[" + utils::num2str(inNumRows) + ", " + utils::num2str(inNumCols) + "]";
                THROW_INVALID_ARGUMENT_ERROR(errStr);
            }

            shape_.m_Rows = static_cast<size_type>(inNumRows);
            shape_.m_Cols = static_cast<size_type>(inNumCols);

            return *this;
        }


        /// The new shape should be compatible with the original shape. If an single integer,
        /// then the result will be a 1-D array of that length. One shape dimension
        /// can be -1. In this case, the value is inferred from the length of the
        /// array and remaining dimensions.
        ///
        /// NOTE: array shape is modified. If a copy is desired then the user should call the copy method:
        /// auto aNew = a.reshape(newShape).copy()
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.reshape.html
        ///
        /// @param inShape
        /// @returns reference to self
        ///
        self_type &reshape(const Shape &inShape)
        {
            return reshape(inShape.m_Rows, inShape.m_Cols);
        }


        /// Change shape and size of array in-place. All previous
        /// data of the array is lost.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.resize.html
        ///
        /// @param inNumRows
        /// @param inNumCols
        ///
        self_type &resizeFast(size_type inNumRows, size_type inNumCols)
        {
            newArray(Shape(inNumRows, inNumCols));
            return *this;
        }


        /// Change shape and size of an array in-place. All previous
        /// data of the array is lost.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.resize.html
        ///
        /// @param inShape
        ///
        self_type &resizeFast(const Shape &inShape)
        {
            return resizeFast(inShape.m_Rows, inShape.m_Cols);
        }


        /// Return a new array with the specified shape. If new shape
        /// is larger than old shape, then an array will be padded with zeros.
        /// If the new shape is smaller than the old shape, then the data will
        /// be discarded.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.resize.html
        ///
        /// @param inNumRows
        /// @param inNumCols
        ///
        self_type &resizeSlow(size_type inNumRows, size_type inNumCols)
        {
            std::vector<dtype> oldData(size_);
            algo::copy(begin(), end(), oldData.begin());

            const Shape inShape(inNumRows, inNumCols);
            const Shape oldShape = shape_;

            newArray(inShape);

            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                for (uint32 col = 0; col < inShape.m_Cols; ++col)
                {
                    if (row >= oldShape.m_Rows || col >= oldShape.m_Cols)
                    {
                        operator()(row, col) = dtype{0}; // zero fill
                    }
                    else
                    {
                        operator()(row, col) = oldData[row * oldShape.m_Cols + col];
                    }
                }
            }

            return *this;
        }


        /// Return a new array with the specified shape. If new shape
        /// is larger than old shape then array will be padded with zeros.
        /// If new shape is smaller than the old shape then the data will
        /// be discarded.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.resize.html
        ///
        /// @param inShape
        ///
        self_type &resizeSlow(const Shape &inShape)
        {
            return resizeSlow(inShape.m_Rows, inShape.m_Cols);
        }


        /// Return a new array with the specified shape.
        self_type &resize(size_type inNumRows, size_type inNumCols, bool keepData = true)
        {
            if (keepData)
                return resizeSlow(inNumRows, inNumCols);
            return resizeFast(inNumRows, inNumCols);
        }


        /// Return a new array with the specified shape.
        self_type &resize(const Shape &inShape, bool keepData = true)
        {
            return resize(inShape.m_Rows, inShape.m_Cols, keepData);
        }


        /// Return a with each element rounded to the given number
        /// of decimals.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.round.html
        ///
        /// @param inNumDecimals (default 0)
        /// @return Matrix
        ///
        [[nodiscard]] self_type round(uint8 inNumDecimals = 0) const
        {
            STATIC_ASSERT_FLOAT(dtype);

            self_type returnArray(shape_);
            const double multFactor = utils::power(10., inNumDecimals);
            const auto function = [multFactor](dtype value) noexcept -> dtype
            {
                return static_cast<dtype>(std::nearbyint(static_cast<double>(value) * multFactor) / multFactor);
            };

            algo::transform(cbegin(), cend(), returnArray.begin(), function);

            return returnArray;
        }


        /// Returns the full row of the array
        ///
        /// @param inRow: the row index
        /// @return self_type
        ///
        [[nodiscard]] self_type row(size_type inRow) const
        {
            return self_type(cbegin(inRow), cend(inRow));
        }


        /// Returns the full row of the array
        ///
        /// @param inRows: the row indices
        /// @return self_type
        ///
        [[nodiscard]] self_type rows(const Matrix<size_type> &inRows) const
        {
            auto returnArray = self_type(inRows.size(), shape_.m_Cols);
            const auto cSlice = returnArray.cSlice();

            for (size_type i = 0; i < inRows.size(); ++i)
            {
                returnArray.put(i, cSlice, row(inRows[i]));
            }

            return returnArray;
        }


        /// Return the shape of the array
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.shape.html
        ///
        /// @return Shape
        ///
        [[nodiscard]] const Shape &shape() const noexcept
        {
            return shape_;
        }


        /// Return the size of the array
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.size.html
        ///
        /// @return size
        ///
        [[nodiscard]] size_type size() const noexcept
        {
            return size_;
        }


        /// Sort an array, in-place.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.sort.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return size
        ///
        self_type &sort(Axis inAxis = Axis::NONE)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const auto comparitor = [](dtype lhs, dtype rhs) noexcept -> bool { return lhs < rhs; };
            // cppcheck-suppress returnTempReference

            switch (inAxis)
            {
            case Axis::NONE:
            {
                algo::sort(begin(), end(), comparitor);
                break;
            }
            case Axis::COL:
            {
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    algo::sort(begin(row), end(row), comparitor);
                }
                break;
            }
            case Axis::ROW:
            {
                self_type transposedArray = transpose();
                for (uint32 row = 0; row < transposedArray.shape_.m_Rows; ++row)
                {
                    algo::sort(transposedArray.begin(row), transposedArray.end(row), comparitor);
                }

                *this = transposedArray.transpose();
                break;
            }
            }

            return *this;
        }


        /// returns the Matrix as a string representation
        ///
        /// @return string
        ///
        [[nodiscard]] std::string str() const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::string out;
            out += "[";
            for (uint32 row = 0; row < shape_.m_Rows; ++row)
            {
                out += "[";
                for (uint32 col = 0; col < shape_.m_Cols; ++col)
                {
                    out += utils::value2str(operator()(row, col)) + ", ";
                }

                if (row == shape_.m_Rows - 1)
                {
                    out += "]";
                }
                else
                {
                    out += "]\n";
                }
            }
            out += "]\n";
            return out;
        }


        /// Return the sum of the array elements over the given axis.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.sum.html
        ///
        /// @param inAxis (Optional, default NONE)
        /// @return Matrix
        ///
        [[nodiscard]] self_type sum(Axis inAxis = Axis::NONE) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            switch (inAxis)
            {
            case Axis::NONE:
            {
                self_type returnArray = {std::accumulate(cbegin(), cend(), dtype{0})};
                return returnArray;
            }
            case Axis::COL:
            {
                self_type returnArray(1, shape_.m_Rows);
                for (uint32 row = 0; row < shape_.m_Rows; ++row)
                {
                    returnArray(0, row) = std::accumulate(cbegin(row), cend(row), dtype{0});
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return transpose().sum(Axis::COL);
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
            }
        }


        /// Interchange two axes of an array. Equivalent to transpose...
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.swapaxes.html
        ///
        /// @return Matrix
        ///
        [[nodiscard]] self_type swapaxes() const
        {
            return transpose();
        }


        /// Swaps rows of the array
        ///
        /// @param colIdx1
        /// @param colIdx2
        /// @return reference to self
        ///
        self_type &swapCols(index_type colIdx1, index_type colIdx2) noexcept
        {
            for (index_type row = 0; row < static_cast<index_type>(shape_.m_Rows); ++row)
            {
                std::swap(operator()(row, colIdx1), operator()(row, colIdx2));
            }

            return *this;
        }


        /// Swaps rows of the array
        ///
        /// @param rowIdx1
        /// @param rowIdx2
        ///
        /// @return reference to self
        self_type &swapRows(index_type rowIdx1, index_type rowIdx2) noexcept
        {
            for (index_type col = 0; col < static_cast<index_type>(shape_.m_Cols); ++col)
            {
                std::swap(operator()(rowIdx1, col), operator()(rowIdx2, col));
            }

            return *this;
        }


        /// Write array to a file as binary.
        /// The data produced by this method can be recovered
        /// using the function fromfile().
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.tofile.html
        ///
        /// @param inFilename
        ///
        void tofile(const std::string &inFilename) const
        {
            dump(inFilename);
        }


        /// Write array to a file as text.
        /// The data produced by this method can be recovered
        /// using the function fromfile().
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.tofile.html
        ///
        /// @param inFilename
        /// @param inSep: Separator between array items for text output.
        ///
        void tofile(const std::string &inFilename, const char inSep) const
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            std::filesystem::path f(inFilename);
            if (!f.has_extension())
            {
                f.replace_extension("txt");
            }

            std::ofstream ofile(f.c_str());
            if (!ofile.good())
            {
                THROW_RUNTIME_ERROR("Input file could not be opened:\n\t" + inFilename);
            }

            size_type counter = 0;
            for (auto value : *this)
            {
                ofile << value;
                if (counter++ != size_ - 1)
                {
                    ofile << inSep;
                }
            }
            ofile.close();
        }


        /// Converts the slice object to an Matrix of indices for this array
        ///
        /// @param inSlice: the slice object
        /// @param inAxis: the array axis
        ///
        /// @return Matrix<index_type>
        ///
        [[nodiscard]] Matrix<size_type> toIndices(Slice inSlice, Axis inAxis = Axis::NONE) const
        {
            size_type numElements = 0;
            switch (inAxis)
            {
            case Axis::NONE:
            {
                numElements = inSlice.numElements(size_);
                break;
            }
            case Axis::ROW:
            {
                numElements = inSlice.numElements(shape_.m_Rows);
                break;
            }
            case Axis::COL:
            {
                numElements = inSlice.numElements(shape_.m_Cols);
                break;
            }
            default:
            {
                // not actually possible, getting rid of compiler warning
                THROW_INVALID_ARGUMENT_ERROR("Invalid 'inAxis' option");
            }
            }

            if (numElements == 0)
            {
                return {};
            }

            Matrix<size_type> indices(1, numElements);
            indices[0] = static_cast<size_type>(inSlice.m_Start);
            for (size_type i = 1; i < indices.size(); ++i)
            {
                indices[static_cast<index_type>(i)] = indices[static_cast<index_type>(i - size_type{1})] + static_cast<
                    size_type>(inSlice.m_Step);
            }

            return indices;
        }

        /// Write a flattened array to an STL vector
        /// @return std::vector
        [[nodiscard]] std::vector<dtype> toFlattenVector() const
        {
            return std::vector<dtype>(cbegin(), cend());
        }

        /// Return the sum along diagonals of the array.
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.trace.html
        ///
        /// @param inOffset: Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults
        /// to 0.
        /// @param inAxis: (Optional, default ROW) Axis to offset from
        ///
        /// @return value
        ///
        [[nodiscard]] value_type trace(size_type inOffset = 0, Axis inAxis = Axis::ROW) const noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            size_type rowStart = 0;
            size_type colStart = 0;
            switch (inAxis)
            {
            case Axis::ROW:
            {
                rowStart += inOffset;
                break;
            }
            case Axis::COL:
            {
                colStart += inOffset;
                break;
            }
            default:
            {
                // if the user input NONE, override back to ROW
                inAxis = Axis::ROW;
                break;
            }
            }

            if (rowStart >= shape_.m_Rows || colStart >= shape_.m_Cols)
            {
                return dtype{0};
            }

            size_type col = colStart;
            dtype sum = 0;
            for (size_type row = rowStart; row < shape_.m_Rows; ++row)
            {
                if (col >= shape_.m_Cols)
                {
                    break;
                }
                sum += operator()(row, col++);
            }

            return sum;
        }


        /// Tranpose the rows and columns of an array
        ///
        /// Numpy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.matrix.transpose.html
        ///
        /// @return Matrix
        ///
        [[nodiscard]] self_type transpose() const
        {
            self_type transArray(shape_.m_Cols, shape_.m_Rows);
            for (uint32 row = 0; row < shape_.m_Rows; ++row)
            {
                for (uint32 col = 0; col < shape_.m_Cols; ++col)
                {
                    transArray(col, row) = operator()(row, col);
                }
            }
            return transArray;
        }


        /// Fills the array with zeros
        ///
        ///
        self_type &zeros() noexcept
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            fill(dtype{0});
            return *this;
        }

    private:
        //====================================Attributes==============================
        allocator_type allocator_{};
        Shape shape_{0, 0};
        size_type size_{0};
        pointer array_{nullptr};
        bool ownsPtr_{false};


        /// Deletes the internal array
        ///
        void deleteArray() noexcept
        {
            if (ownsPtr_ && array_ != nullptr)
            {
                allocator_.deallocate(array_, size_);
            }

            array_ = nullptr;
            shape_.m_Rows = shape_.m_Cols = 0;
            size_ = 0;
            ownsPtr_ = false;
        }


        /// Creates a new internal array
        void newArray()
        {
            if (size_ > 0)
            {
                array_ = allocator_.allocate(size_);
                ownsPtr_ = true;
            }
        }


        /// Creates a new internal array
        void newArray(const Shape &inShape)
        {
            deleteArray();

            shape_ = inShape;
            size_ = inShape.size();
            newArray();
        }
    };

    // NOTE: this needs to be defined outside the class to get rid of a compiler
    // error in Visual Studio
    template <typename dtype, class Alloc_>
    [[nodiscard]] std::pair<Matrix<uint32>, Matrix<uint32>> Matrix<dtype, Alloc_>::nonzero() const
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        std::vector<size_type> rowIndices;
        std::vector<size_type> colIndices;

        for (uint32 row = 0; row < shape_.m_Rows; ++row)
        {
            for (uint32 col = 0; col < shape_.m_Cols; ++col)
            {
                if (!utils::essentiallyEqual(operator()(row, col), dtype{0}))
                {
                    rowIndices.push_back(row);
                    colIndices.push_back(col);
                }
            }
        }

        return std::make_pair(Matrix(rowIndices), Matrix(colIndices));
    }
}

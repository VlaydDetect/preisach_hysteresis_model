// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <iterator>

namespace mc
{
    //================================================================================
    // Class Description:
    /// Custom const_iterator for Matrix
    template <typename dtype, typename PointerType, typename DifferenceType>
    class MatrixConstIterator
    {
    private:
        using self_type = MatrixConstIterator<dtype, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = const value_type &;
        using difference_type = DifferenceType;
        
        MatrixConstIterator() = default;
        
        /// Constructor
        /// @param ptr: the iterator pointer
        explicit MatrixConstIterator(pointer ptr) noexcept :
            m_Pointer(ptr)
        {
        }
        
        /// Iterator dereference
        /// @return reference
        reference operator*() const noexcept
        {
            return *m_Pointer;
        }

        /// Iterator pointer operator
        /// @return pointer
        pointer operator->() const noexcept
        {
            return m_Pointer;
        }
        
        /// Iterator prefix increment operator
        /// @return MatrixConstIterator&
        self_type &operator++() noexcept
        {
            ++m_Pointer;
            return *this;
        }
        
        /// Iterator postfix increment operator
        /// @return MatrixConstIterator
        self_type operator++(int) noexcept
        {
            self_type tmp = *this;
            ++*this;
            return tmp;
        }
        
        /// Iterator prefix decrement operator
        /// @return MatrixConstIterator&
        self_type &operator--() noexcept
        {
            --m_Pointer;
            return *this;
        }
        
        /// Iterator postfix decrement operator
        /// @return MatrixConstIterator
        self_type operator--(int) noexcept
        {
            self_type tmp = *this;
            --*this;
            return tmp;
        }

        
        /// Iterator addition assignment operator
        /// @return MatrixConstIterator&
        self_type &operator+=(const difference_type offset) noexcept
        {
            m_Pointer += offset;
            return *this;
        }

        
        /// Iterator addition operator
        /// @return MatrixConstIterator
        self_type operator+(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        
        /// Iterator subtraction assignment operator
        /// @return MatrixConstIterator&
        self_type &operator-=(const difference_type offset) noexcept
        {
            return *this += -offset;
        }

        
        /// Iterator subtraction operator
        /// @return MatrixConstIterator
        self_type operator-(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        
        /// Iterator difference operator
        /// @return difference_type
        difference_type operator-(const self_type &rhs) const noexcept
        {
            return m_Pointer - rhs.m_Pointer;
        }

        
        /// Iterator access operator
        /// @return reference
        reference operator[](const difference_type offset) const noexcept
        {
            return *(*this + offset);
        }

        bool operator==(const self_type &rhs) const noexcept
        {
            return m_Pointer == rhs.m_Pointer;
        }

        bool operator!=(const self_type &rhs) const noexcept
        {
            return !(*this == rhs);
        }
        
        bool operator<(const self_type &rhs) const noexcept
        {
            return m_Pointer < rhs.m_Pointer;
        }
        
        bool operator>(const self_type &rhs) const noexcept
        {
            return rhs < *this;
        }
        
        bool operator<=(const self_type &rhs) const noexcept
        {
            return !(rhs < *this);
        }
        
        bool operator>=(const self_type &rhs) const noexcept
        {
            return !(*this < rhs);
        }

    private:
        pointer m_Pointer{nullptr};
    };
    
    /// Iterator addition operator
    template <class dtype, typename PointerType, typename DifferenceType>
    MatrixConstIterator<dtype, PointerType, DifferenceType>
    operator+(typename MatrixConstIterator<dtype, PointerType, DifferenceType>::difference_type offset,
              MatrixConstIterator<dtype, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
    
    /// Custom iterator for Matrix
    template <typename dtype, typename PointerType, typename DifferenceType>
    class MatrixIterator : public MatrixConstIterator<dtype, PointerType, DifferenceType>
    {
    private:
        using BaseIt = MatrixConstIterator<dtype, PointerType, DifferenceType>;
        using self_type = MatrixIterator<dtype, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = value_type &;
        using difference_type = DifferenceType;

        using BaseIt::BaseIt;

        
        /// Iterator dereference
        reference operator*() const noexcept
        {
            return const_cast<reference>(BaseIt::operator*());
        }

        
        /// Iterator pointer operator
        pointer operator->() const noexcept
        {
            return const_cast<pointer>(BaseIt::operator->());
        }

        
        /// Iterator prefix increment operator
        self_type &operator++() noexcept
        {
            BaseIt::operator++();
            return *this;
        }

        
        /// Iterator postfix increment operator
        self_type operator++(int) noexcept
        {
            self_type tmp = *this;
            BaseIt::operator++();
            return tmp;
        }

        
        /// Iterator prefix decrement operator
        self_type &operator--() noexcept
        {
            BaseIt::operator--();
            return *this;
        }

        
        /// Iterator postfix decrement operator
        self_type operator--(int) noexcept
        {
            self_type tmp = *this;
            BaseIt::operator--();
            return tmp;
        }

        
        /// Iterator addition assignment operator
        self_type &operator+=(const difference_type offset) noexcept
        {
            BaseIt::operator+=(offset);
            return *this;
        }

        
        /// Iterator addition operator
        self_type operator+(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        
        /// Iterator subtraction assignment operator
        self_type &operator-=(const difference_type offset) noexcept
        {
            BaseIt::operator-=(offset);
            return *this;
        }

        using BaseIt::operator-;

        
        /// Iterator difference operator
        self_type operator-(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        
        /// Iterator access operator
        reference operator[](const difference_type offset) const noexcept
        {
            return const_cast<reference>(BaseIt::operator[](offset));
        }
    };
    
    /// Iterator addition operator
    template <class dtype, typename PointerType, typename DifferenceType>
    MatrixIterator<dtype, PointerType, DifferenceType>
    operator+(typename MatrixIterator<dtype, PointerType, DifferenceType>::difference_type offset,
              MatrixIterator<dtype, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
    
    /// Custom column const_iterator for Matrix
    template <typename dtype, typename SizeType, typename PointerType, typename DifferenceType>
    class MatrixConstColumnIterator
    {
    private:
        using self_type = MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using size_type = SizeType;
        using pointer = PointerType;
        using reference = const value_type &;
        using difference_type = DifferenceType;

        
        MatrixConstColumnIterator() = default;

        
        /// Constructor
        ///
        /// @param ptr: the iterator pointer
        /// @param numRows: the number of rows in the array
        /// @param numCols: the number of cols in the array
        MatrixConstColumnIterator(pointer ptr, SizeType numRows, SizeType numCols) noexcept :
            m_Pointer(ptr),
            m_CurrentPointer(ptr),
            m_NumRows(static_cast<difference_type>(numRows)),
            m_NumCols(static_cast<difference_type>(numCols)),
            m_Size(m_NumRows * m_NumCols)
        {
        }

        
        /// Iterator dereference
        reference operator*() const noexcept
        {
            return *m_CurrentPointer;
        }

        
        /// Iterator pointer operator
        pointer operator->() const noexcept
        {
            return m_CurrentPointer;
        }

        
        /// Iterator prefix increment operator
        self_type &operator++() noexcept
        {
            return *this += 1;
        }

        
        /// Iterator postfix increment operator
        self_type operator++(int) noexcept
        {
            self_type tmp = *this;
            ++*this;
            return tmp;
        }

        
        /// Iterator prefix decrement operator
        self_type &operator--() noexcept
        {
            return *this -= 1;
        }

        
        /// Iterator postfix decrement operator
        self_type operator--(int) noexcept
        {
            self_type tmp = *this;
            --*this;
            return tmp;
        }

        
        /// Iterator addition assignment operator
        self_type &operator+=(const difference_type offset) noexcept
        {
            m_CurrentPointer = colIdx2Ptr(ptr2ColIdx(m_CurrentPointer) + offset);
            return *this;
        }

        
        /// Iterator addition operator
        self_type operator+(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        
        /// Iterator subtraction assignment operator
        self_type &operator-=(const difference_type offset) noexcept
        {
            return *this += -offset;
        }

        
        /// Iterator subtraction operator
        self_type operator-(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        
        /// Iterator difference operator
        difference_type operator-(const self_type &rhs) const noexcept
        {
            return ptr2ColIdx(m_CurrentPointer) - ptr2ColIdx(rhs.m_CurrentPointer);
        }

        
        /// Iterator access operator
        reference operator[](const difference_type offset) const noexcept
        {
            return *(*this + offset);
        }

        
        /// Iterator equality operator
        bool operator==(const self_type &rhs) const noexcept
        {
            return m_CurrentPointer == rhs.m_CurrentPointer;
        }

        
        /// Iterator not-equality operator
        bool operator!=(const self_type &rhs) const noexcept
        {
            return !(*this == rhs);
        }

        
        /// Iterator less than operator
        bool operator<(const self_type &rhs) const noexcept
        {
            return *this - rhs < 0;
        }

        
        /// Iterator greater than operator
        bool operator>(const self_type &rhs) const noexcept
        {
            return *this - rhs > 0;
        }

        
        /// Iterator less than equal operator
        bool operator<=(const self_type &rhs) const noexcept
        {
            return !(rhs < *this);
        }

        
        /// Iterator greater than equal operator
        bool operator>=(const self_type &rhs) const noexcept
        {
            return !(*this < rhs);
        }

    private:
        pointer m_Pointer{};
        pointer m_CurrentPointer{};
        difference_type m_NumRows{0};
        difference_type m_NumCols{0};
        difference_type m_Size{0};

        
        /// Converts a pointer to column order index
        /// @return difference_type
        difference_type ptr2ColIdx(pointer ptr) const noexcept
        {
            if (ptr == nullptr)
            {
                return m_Size;
            }

            const auto rowIdx = ptr - m_Pointer;
            if (rowIdx >= m_Size)
            {
                return m_Size;
            }

            const auto row = rowIdx / m_NumCols;
            const auto col = rowIdx % m_NumCols;
            return row + col * m_NumRows;
        }

        
        /// Converts column order index to a pointer
        /// @return pointer
        pointer colIdx2Ptr(difference_type colIdx) const noexcept
        {
            if (colIdx >= m_Size)
            {
                return nullptr;
            }

            const auto row = colIdx % m_NumRows;
            const auto col = colIdx / m_NumRows;
            const auto rowIdx = col + row * m_NumCols;
            return m_Pointer + rowIdx;
        }
    };
    
    /// Iterator addition operator
    template <class dtype, typename SizeType, typename PointerType, typename DifferenceType>
    MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType> operator+(
        typename MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>::difference_type offset,
        MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
    
    /// Custom column iterator for Matrix
    template <typename dtype, typename SizeType, typename PointerType, typename DifferenceType>
    class MatrixColumnIterator : public MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>
    {
    private:
        using BaseIt = MatrixConstColumnIterator<dtype, SizeType, PointerType, DifferenceType>;
        using self_type = MatrixColumnIterator<dtype, SizeType, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using size_type = SizeType;
        using pointer = PointerType;
        using reference = value_type &;
        using difference_type = DifferenceType;

        using BaseIt::BaseIt;

        
        /// Iterator dereference
        reference operator*() const noexcept
        {
            return const_cast<reference>(BaseIt::operator*());
        }

        
        /// Iterator pointer operator
        pointer operator->() const noexcept
        {
            return const_cast<pointer>(BaseIt::operator->());
        }

        
        /// Iterator prefix increment operator
        self_type &operator++() noexcept
        {
            BaseIt::operator++();
            return *this;
        }

        
        /// Iterator postfix increment operator

        self_type operator++(int) noexcept
        {
            self_type tmp = *this;
            BaseIt::operator++();
            return tmp;
        }

        
        /// Iterator prefix decrement operator
        self_type &operator--() noexcept
        {
            BaseIt::operator--();
            return *this;
        }

        
        /// Iterator postfix decrement operator
        self_type operator--(int) noexcept
        {
            self_type tmp = *this;
            BaseIt::operator--();
            return tmp;
        }

        
        /// Iterator addition assignment operator
        self_type &operator+=(const difference_type offset) noexcept
        {
            BaseIt::operator+=(offset);
            return *this;
        }

        
        /// Iterator addition operator
        self_type operator+(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp += offset;
        }

        
        /// Iterator subtraction assignment operator
        self_type &operator-=(const difference_type offset) noexcept
        {
            BaseIt::operator-=(offset);
            return *this;
        }

        using BaseIt::operator-;

        
        /// Iterator difference operator
        self_type operator-(const difference_type offset) const noexcept
        {
            self_type tmp = *this;
            return tmp -= offset;
        }

        
        /// Iterator access operator
        reference operator[](const difference_type offset) const noexcept
        {
            return const_cast<reference>(BaseIt::operator[](offset));
        }
    };
    
    /// Iterator addition operator
    template <class dtype, typename SizeType, typename PointerType, typename DifferenceType>
    MatrixColumnIterator<dtype, SizeType, PointerType, DifferenceType>
    operator+(typename MatrixColumnIterator<dtype, SizeType, PointerType, DifferenceType>::difference_type offset,
              MatrixColumnIterator<dtype, SizeType, PointerType, DifferenceType> next) noexcept
    {
        return next += offset;
    }
}

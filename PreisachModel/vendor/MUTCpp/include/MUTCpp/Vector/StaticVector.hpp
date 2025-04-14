// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>
#include <initializer_list>
#include <array>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"

namespace mc
{
    template <typename T, uint32 MaxSize>
    class StaticVector
    {
    public:
        using self_type = StaticVector<T, MaxSize>;
        using reference = T&;
        using const_reference = const T&;
        
    public:
        StaticVector() :
            m_Size(0)
        {
        }

        StaticVector(const std::initializer_list<T>& inValue)
        {
            if (inValue.size() > MaxSize)
            {
                THROW_INVALID_ARGUMENT_ERROR("The inValue size exceeds the MaxSize of StaticVector!");
            }
            m_Size = inValue.size();
            algo::copy(inValue.begin(), inValue.end(), std::begin(m_Data));
        }

        StaticVector(const std::vector<T>& inValue)
        {
            if (inValue.size() > MaxSize)
            {
                THROW_INVALID_ARGUMENT_ERROR("The inValue size exceeds the MaxSize of StaticVector!");
            }
            m_Size = inValue.size();
            algo::copy(inValue.begin(), inValue.end(), std::begin(m_Data));
        }

        template<size_t ArraySize>
        StaticVector(const std::array<T, ArraySize>& inValue)
        {
            if (ArraySize > MaxSize)
            {
                THROW_INVALID_ARGUMENT_ERROR("The inValue size exceeds the MaxSize of StaticVector!");
            }
            m_Size = ArraySize;
            algo::copy(inValue.begin(), inValue.end(), std::begin(m_Data));
        }

        StaticVector(T inValue)
        {
            m_Size = 1;
            m_Data[0] = inValue;
        }
        
        void push(const T& value)
        {
            if (m_Size >= MaxSize)
            {
                THROW_OVERFLOW_ERROR("StaticVector is full!");
            }

            m_Data[m_Size++] = value;
        }

        void pop() noexcept
        {
            if (m_Size > 0)
            {
                --m_Size;
            }
            else
            {
                THROW_UNDERFLOW_ERROR("StaticVector is empty!");
            }
        }

        reference operator[](int32 index) noexcept
        {
            return const_cast<reference>(const_cast<const self_type*>(this)->operator[](index));
        }

        [[nodiscard]] const_reference operator[](int32 index) const noexcept
        {
            if (index >= m_Size)
            {
                THROW_OUT_OF_RANGE_ERROR("Index out of StaticVector size!");
            }

            if (index < 0)
            {
                index += m_Size;
            }
            
            return m_Data[index];
        }

        uint32 size() const { return m_Size; }
        constexpr uint32 max_size() const { return MaxSize; }

        bool empty() const { return m_Size == 0; }

    private:
        T m_Data[MaxSize];
        uint32 m_Size;
    };
}

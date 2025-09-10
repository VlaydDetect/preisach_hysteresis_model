// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <ranges>
#include "Core/Error.hpp"

namespace mc
{
    template <typename T, uint32 MaxSize>
    class StaticVector
    {
    public:
        using self_type = StaticVector<T, MaxSize>;
        using reference = T &;
        using const_reference = const T &;

    public:
        StaticVector() :
            m_Size(0)
        {
        }

        template <std::ranges::input_range R>
            requires std::convertible_to<std::ranges::range_value_t<R>, T>
        StaticVector(R &&range)
        {
            auto n = std::ranges::distance(range);
            if (n > MaxSize)
            {
                THROW_INVALID_ARGUMENT_ERROR("The inValue size exceeds the MaxSize of StaticVector!");
            }
            m_Size = static_cast<uint32>(n);
            std::ranges::copy(range, m_Data.begin());
        }

        explicit StaticVector(const T& inValue)
        {
            m_Size = 1;
            m_Data[0] = inValue;
        }

        void push(const T &value)
        {
            if (m_Size >= MaxSize)
            {
                THROW_OVERFLOW_ERROR("StaticVector is full!");
            }

            m_Data[m_Size++] = value;
        }

        void pop() noexcept
        {
            if (m_Size == 0) {
                THROW_UNDERFLOW_ERROR("Empty StaticVector!");
            }
            std::destroy_at(std::addressof(m_Data[m_Size--]));
        }

        reference front() noexcept
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->front());
        }

        [[nodiscard]] const_reference front() const noexcept
        {
            return m_Data[0];
        }

        reference back() noexcept
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->back());
        }

        [[nodiscard]] const_reference back() const noexcept
        {
            return m_Data[m_Size - 1];
        }

        auto begin() noexcept { return std::begin(m_Data); }
        
        auto end() noexcept { return std::begin(m_Data) + m_Size; }
        
        [[nodiscard]] auto begin() const noexcept { return std::begin(m_Data); }
        
        [[nodiscard]] auto end() const noexcept { return std::begin(m_Data) + m_Size; }

        reference operator[](uint32 index) noexcept
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->operator[](index));
        }

        [[nodiscard]] const_reference operator[](uint32 index) const noexcept
        {
            return m_Data[index];
        }

        reference at(int32 index)
        {
            return const_cast<reference>(const_cast<const self_type *>(this)->at(index));
        }

        [[nodiscard]] const_reference at(int32 index) const
        {
            if (index < 0)
                index += m_Size;
            if (index < 0 || index >= m_Size)
            {
                THROW_OUT_OF_RANGE_ERROR("Index out of StaticVector size!");
            }
            return m_Data[index];
        }

        constexpr uint32 size() const noexcept { return m_Size; }
        static constexpr uint32 max_size() noexcept { return MaxSize; }

        constexpr bool empty() const noexcept { return m_Size == 0; }

    private:
        T m_Data[MaxSize];
        uint32 m_Size;
    };
}

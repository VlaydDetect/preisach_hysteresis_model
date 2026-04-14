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
    template <typename T, uint32_t MaxSize>
    class StaticVector
    {
    public:
        using value_type = T;
        using self_type = StaticVector<T, MaxSize>;
        using reference = T &;
        using const_reference = const T &;

    public:
        StaticVector() noexcept : m_Size(0) {}

        template <std::ranges::input_range R>
            requires std::convertible_to<std::ranges::range_value_t<R>, T>
        StaticVector(R&& range) : m_Size(0)
        {
            // Исправлено: избегаем std::ranges::distance, который "потребляет" input_range
            for (auto&& val : range)
            {
                push(std::forward<decltype(val)>(val));
            }
        }

        explicit StaticVector(const T& inValue) : m_Size(0)
        {
            push(inValue);
        }
        
        // Правильное управление памятью (Rule of 5)
        ~StaticVector()
        {
            clear();
        }
        
        StaticVector(const StaticVector& other) : m_Size(0)
        {
            for (uint32_t i = 0; i < other.m_Size; ++i) {
                push(other[i]);
            }
        }
        
        StaticVector& operator=(const StaticVector& other)
        {
            if (this != &other) {
                clear();
                for (uint32_t i = 0; i < other.m_Size; ++i) {
                    push(other[i]);
                }
            }
            return *this;
        }
        
        StaticVector(StaticVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) : m_Size(0)
        {
            for (uint32_t i = 0; i < other.m_Size; ++i) {
                push(std::move(other[i]));
            }
            other.clear();
        }
        
        StaticVector& operator=(StaticVector&& other) noexcept(std::is_nothrow_move_assignable_v<T>)
        {
            if (this != &other) {
                clear();
                for (uint32_t i = 0; i < other.m_Size; ++i) {
                    push(std::move(other[i]));
                }
                other.clear();
            }
            return *this;
        }

        void push(const T &value)
        {
            if (m_Size >= MaxSize)
            {
                THROW_OVERFLOW_ERROR("StaticVector is full!");
            }

            std::construct_at(data() + m_Size, value);
            ++m_Size;
        }
        
        void push(T&& value)
        {
            if (m_Size >= MaxSize) THROW_OVERFLOW_ERROR("StaticVector is full!");
            std::construct_at(data() + m_Size, std::move(value));
            ++m_Size;
        }

        void pop() noexcept
        {
            if (m_Size == 0) THROW_UNDERFLOW_ERROR("Empty StaticVector!");
            --m_Size;
            std::destroy_at(data() + m_Size);
        }
        
        void clear() noexcept
        {
            while (m_Size > 0) {
                pop();
            }
        }

        // ======================================================================
        // C++23 Deducing 'this': Убирает дублирование const/non-const геттеров
        // ======================================================================
        
        template <typename Self>
        constexpr auto&& operator[](this Self&& self, uint32_t index) noexcept
        {
            return std::forward<Self>(self).data()[index];
        }

        template <typename Self>
        constexpr auto&& front(this Self&& self) noexcept
        {
            return std::forward<Self>(self).data()[0];
        }

        template <typename Self>
        constexpr auto&& back(this Self&& self) noexcept
        {
            return std::forward<Self>(self).data()[self.m_Size - 1];
        }

        template <typename Self>
        constexpr auto&& at(this Self&& self, int32_t index)
        {
            if (index < 0) index += self.m_Size;
            
            // Защита от implicit cast при сравнении знаковых и беззнаковых чисел
            if (index < 0 || static_cast<uint32_t>(index) >= self.m_Size)
            {
                THROW_OUT_OF_RANGE_ERROR("Index out of StaticVector size!");
            }
            return std::forward<Self>(self).data()[index];
        }

        // ======================================================================
        // Итераторы и данные
        // ======================================================================

        T* data() noexcept { return reinterpret_cast<T*>(&m_Storage); }
        const T* data() const noexcept { return reinterpret_cast<const T*>(&m_Storage); }

        auto begin() noexcept { return data(); }
        auto end() noexcept { return data() + m_Size; }
        auto begin() const noexcept { return data(); }
        auto end() const noexcept { return data() + m_Size; }

        constexpr uint32_t size() const noexcept { return m_Size; }
        static constexpr uint32_t max_size() noexcept { return MaxSize; }
        constexpr bool empty() const noexcept { return m_Size == 0; }

    private:
        // Используем неинициализированную память для соблюдения строгих правил RAII (одно создание = одно удаление)
        alignas(T) std::byte m_Storage[MaxSize * sizeof(T)];
        uint32_t m_Size;
    };
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "Core/Types.hpp"
#include "Utils/num2str.hpp"

namespace mc
{
    class Shape
    {
    public:
        constexpr Shape() = default;

        constexpr explicit Shape(uint32 inSquareSize) noexcept :
            m_Rows(inSquareSize),
            m_Cols(inSquareSize)
        {
        }

        constexpr Shape(uint32 inRows, uint32 inCols) noexcept :
            m_Rows(inRows),
            m_Cols(inCols)
        {
        }

        bool operator==(const Shape& inOtherShape) const noexcept
        {
            return m_Rows == inOtherShape.m_Rows && m_Cols == inOtherShape.m_Cols;
        }

        bool operator!=(const Shape& inOtherShape) const noexcept
        {
            return !(*this == inOtherShape);
        }

        /// IO operator for the Shape class
        friend std::ostream& operator<<(std::ostream& inOStream, const Shape& inShape)
        {
            inOStream << inShape.str();
            return inOStream;
        }

        /// Returns the size of the shape
        [[nodiscard]] uint32 size() const noexcept
        {
            return m_Rows * m_Cols;
        }

        /// Returns whether the shape is null (constructed with the
        /// default constructor).
        [[nodiscard]] bool isnull() const noexcept
        {
            return m_Rows == 0 && m_Cols == 0;
        }

        /// Returns whether the shape is square or not.
        [[nodiscard]] bool issquare() const noexcept
        {
            return m_Rows == m_Cols;
        }

        /// Returns the shape as a string representation
        [[nodiscard]] std::string str() const
        {
            std::string out = "[" + utils::num2str(m_Rows) + ", " + utils::num2str(m_Cols) + "]\n";
            return out;
        }

        /// Prints the shape to the console
        void print() const
        {
            std::cout << *this;
        }
        
    public:
        uint32 m_Rows{ 0 };
        uint32 m_Cols{ 0 };
    };
}

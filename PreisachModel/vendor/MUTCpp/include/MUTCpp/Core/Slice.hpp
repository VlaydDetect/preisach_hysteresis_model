// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Utils/num2str.hpp"

namespace mc
{
    /// A Class for slicing into NdArrays
    class Slice
    {
    public:
        constexpr Slice() = default;

        /// @param inStop (index isn't included)
        constexpr explicit Slice(int32 inStop) noexcept :
            m_Stop(inStop)
        {
        }

        /// @param inStart
        /// @param inStop (index isn't included)
        constexpr Slice(int32 inStart, int32 inStop) noexcept :
            m_Start(inStart),
            m_Stop(inStop)
        {
        }

        /// @param inStart
        /// @param inStop (not included)
        /// @param inStep
        constexpr Slice(int32 inStart, int32 inStop, int32 inStep) noexcept :
            m_Start(inStart),
            m_Stop(inStop),
            m_Step(inStep)
        {
        }

        bool operator==(const Slice& inOtherSlice) const noexcept
        {
            return m_Start == inOtherSlice.m_Start && m_Stop == inOtherSlice.m_Stop && m_Step == inOtherSlice.m_Step;
        }

        bool operator!=(const Slice& inOtherSlice) const noexcept
        {
            return !(*this == inOtherSlice);
        }

        /// IO operator for the Slice class
        friend std::ostream& operator<<(std::ostream& inOStream, const Slice& inSlice)
        {
            inOStream << inSlice.str();
            return inOStream;
        }

        /// Converts Slice to string
        /// @return std::string
        [[nodiscard]] std::string str() const
        {
            std::string out =
                "[" + utils::num2str(m_Start) + ":" + utils::num2str(m_Stop) + ":" + utils::num2str(m_Step) + "]\n";
            return out;
        }

        /// Prints the shape to the console
        ///
        void print() const
        {
            std::cout << *this;
        }

        /// Make the slice all positive and does some error checking
        void makePositiveAndValidate(uint32 inArraySize)
        {
            /// convert the start value
            if (m_Start < 0)
            {
                m_Start += static_cast<int32>(inArraySize);
            }
            if (m_Start > static_cast<int32>(inArraySize - 1))
            {
                THROW_INVALID_ARGUMENT_ERROR("Invalid start value for array of size " + utils::num2str(inArraySize));
            }

            /// convert the stop value
            if (m_Stop < 0)
            {
                m_Stop += static_cast<int32>(inArraySize);
            }
            if (m_Stop > static_cast<int32>(inArraySize))
            {
                THROW_INVALID_ARGUMENT_ERROR("Invalid stop value for array of size " + utils::num2str(inArraySize));
            }

            /// do some error checking
            if (m_Start < m_Stop)
            {
                if (m_Step < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("Invalid slice values [" + utils::num2str(m_Start) + ", " +
                                                 utils::num2str(m_Stop) + ", " + utils::num2str(m_Step) + ']');
                }
            }

            if (m_Stop < m_Start)
            {
                if (m_Step > 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("Invalid slice values [" + utils::num2str(m_Start) + ", " +
                                                 utils::num2str(m_Stop) + ", " + utils::num2str(m_Step) + ']');
                }

                /// otherwise flip things around for my own sanity
                std::swap(m_Start, m_Stop);
                m_Step *= -1;
            }
        }

        /// Returns the number of elements that the slice contains.
        /// Be aware that this method will also make the slice all
        /// positive!
        uint32 numElements(uint32 inArraySize)
        {
            makePositiveAndValidate(inArraySize);

            uint32 num = 0;
            for (int32 i = m_Start; i < m_Stop; i += m_Step)
            {
                ++num;
            }
            return num;
        }

        /// Returns the indices that correspond to the slice
        /// be aware that this method will also make the slice all
        /// positive!
        ///
        /// @param inArrayDimSize: the size of the dimension that is being sliced
        std::vector<uint32> toIndices(uint32 inArrayDimSize)
        {
            std::vector<uint32> indices;
            indices.reserve(numElements(inArrayDimSize));
            for (int32 i = m_Start; i < m_Stop; i += m_Step)
            {
                indices.push_back(static_cast<uint32>(i));
            }
            return indices;
        }
        
    public:
        int32 m_Start{ 0 };
        int32 m_Stop{ 1 };
        int32 m_Step{ 1 };
    };
}

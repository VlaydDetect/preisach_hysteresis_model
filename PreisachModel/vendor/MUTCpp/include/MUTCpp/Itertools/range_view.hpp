// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

namespace mc
{
    namespace itertools
    {
        template <typename Iterator>
        class range_view
        {
        public:
            range_view(Iterator first, Iterator last) :
                m_First(first), m_Last(last)
            {
            }

            Iterator begin() const { return m_First; }

            Iterator end() const { return m_Last; }

        private:
            Iterator m_First;
            Iterator m_Last;
        };
    }
}

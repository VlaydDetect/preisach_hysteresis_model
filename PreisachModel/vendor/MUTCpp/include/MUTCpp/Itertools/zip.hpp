// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <tuple>
#include "Itertools/range_view.hpp"

namespace mc
{
    namespace itertools
    {
        namespace detail
        {
            template <typename... Iterator>
            class zip_iterator;

            template <typename Iterator>
            class zip_iterator<Iterator>
            {
            public:
                zip_iterator(Iterator first) :
                    m_It(first)
                {
                }

                decltype(auto) operator*() const
                {
                    return std::make_tuple(*m_It);
                }

                zip_iterator &operator++()
                {
                    ++m_It;
                    return *this;
                }

                bool operator==(const zip_iterator &other) const
                {
                    return m_It == other.m_It;
                }

                bool operator!=(const zip_iterator &other) const
                {
                    return !(*this == other);
                }

            private:
                Iterator m_It;
            };

            template <typename Iterator, typename... Iterators>
            class zip_iterator<Iterator, Iterators...>
            {
            public:
                zip_iterator(Iterator first, Iterators... rest) :
                    m_It(first), m_SubIt(rest...)
                {
                }

                decltype(auto) operator*() const
                {
                    return std::tuple_cat(std::make_tuple(*m_It), *m_SubIt);
                }

                zip_iterator &operator++()
                {
                    ++m_It;
                    ++m_SubIt;
                    return *this;
                }

                bool operator==(const zip_iterator &other) const
                {
                    return m_It == other.m_It || m_SubIt == other.m_SubIt;
                }

                bool operator!=(const zip_iterator &other) const
                {
                    return !(*this == other);
                }

            private:
                Iterator m_It;
                zip_iterator<Iterators...> m_SubIt;
            };
        }

        template <typename... Iterables>
        auto zip(Iterables &&... iterables)
        {
            using it_t = detail::zip_iterator<decltype(iterables.begin())...>;
            return range_view<it_t>(it_t(iterables.begin()...), it_t(iterables.end()...));
        }
    }
}

// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>
#include <tuple>
#include <utility>

namespace mc
{
    namespace itertools
    {
        namespace detail
        {
            // product_iterator is a tuple of regular iterators
            template <typename... Iterators>
            class product_iterator;

            // product_iterator specialization for just one iterator
            template <typename Iterator>
            class product_iterator<Iterator>
            {
            public:
                product_iterator(Iterator it, Iterator last) :
                    m_It(it), m_ItLast(last)
                {
                }

                decltype(auto) operator*() const
                {
                    return std::make_tuple(*m_It);
                }

                product_iterator<Iterator> &operator++()
                {
                    ++m_It;
                    return *this;
                }

                bool operator==(const product_iterator &other) const
                {
                    return m_It == other.m_It;
                }

                bool operator!=(const product_iterator &other) const
                {
                    return !(*this == other);
                }

            private:
                Iterator m_It;
                Iterator m_ItLast;
            };

            // product_iterator general case
            template <typename Iterator, typename... Iterators>
            class product_iterator<Iterator, Iterators...>
            {
            public:
                product_iterator(Iterator it, Iterator it_last, product_iterator<Iterators...> its,
                                 product_iterator<Iterators...> its_first, product_iterator<Iterators...> its_last) :
                    m_It(it), m_ItLast(it_last), m_Its(its), m_ItsFirst(its_first), m_ItsLast(its_last)
                {
                }

                decltype(auto) operator*() const
                {
                    return std::tuple_cat(std::make_tuple(*m_It), *m_Its);
                }

                product_iterator &operator++()
                {
                    if (++m_Its == m_ItsLast && ++m_It != m_ItLast)
                    {
                        m_Its = m_ItsFirst;
                    }
                    return *this;
                }

                bool operator==(const product_iterator &other) const
                {
                    return m_It == other.m_It && m_Its == other.m_Its;
                }

                bool operator!=(const product_iterator &other) const
                {
                    return !(*this == other);
                }

            private:
                Iterator m_It;
                Iterator m_ItLast;
                product_iterator<Iterators...> m_Its;
                product_iterator<Iterators...> m_ItsFirst;
                product_iterator<Iterators...> m_ItsLast;
            };

            // similar strategy for product_impl
            template <typename... Iterators>
            class product_impl;

            // product_impl specialization for just one iterator
            template <typename Iterator>
            class product_impl<Iterator>
            {
            public:
                product_impl(Iterator first, Iterator last) :
                    m_ItFirst(first), m_ItLast(last)
                {
                }

                auto begin() const
                {
                    return product_iterator<Iterator>(m_ItFirst, m_ItLast);
                }

                auto end() const
                {
                    return product_iterator<Iterator>(m_ItLast, m_ItLast);
                }

            private:
                Iterator m_ItFirst;
                Iterator m_ItLast;
            };

            // product_impl general case
            template <typename Iterator, typename... Iterators>
            class product_impl<Iterator, Iterators...>
            {
            public:
                template <typename... ArgsT>
                product_impl(Iterator first, Iterator last, ArgsT... rest) :
                    m_ItFirst(first), m_ItLast(last), m_SubProduct(rest...)
                {
                }

                auto begin() const
                {
                    return product_iterator<Iterator, Iterators...>(m_ItFirst, m_ItLast, m_SubProduct.begin(),
                                                                    m_SubProduct.begin(), m_SubProduct.end());
                }

                auto end() const
                {
                    return product_iterator<Iterator, Iterators...>(m_ItLast, m_ItLast, m_SubProduct.end(),
                                                                    m_SubProduct.begin(), m_SubProduct.end());
                }

            private:
                Iterator m_ItFirst;
                Iterator m_ItLast;
                product_impl<Iterators...> m_SubProduct;
            };
        }

        /// Cartesian product of input iterables. Equivalent to nested for-loops.
        template <typename... Iterables>
        auto product(Iterables &&... iterables)
        {
            return std::apply(
                [&iterables]<typename... T0>(T0 &&... args)
                {
                    return detail::product_impl<decltype(iterables.begin())...>(std::forward<T0>(args)...);
                },
                std::tuple_cat(std::make_tuple(iterables.begin(), iterables.end())...));
        }
    }
}

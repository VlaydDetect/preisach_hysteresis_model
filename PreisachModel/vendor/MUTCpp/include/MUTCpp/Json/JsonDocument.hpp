// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cassert>
#include <set>
#include <map>

#include <Eigen/Dense>
#include "Eigen/utils.hpp"

#include "Matrix/MatrixCore.hpp"
#include "Vector/Vec2.hpp"
#include "Vector/Vec3.hpp"
#include "vendor/json/nlohmann_json.hpp"

#define ASSIGN_FIELDS(container, data)  \
    [&]() {                             \
        auto it = (container).begin();  \
        auto end = (container).end();   \
        if (it == end) return;          \
        auto* target = &m_Doc[*it++];   \
        while (it != std::prev(end)) {  \
            target = &(*target)[*it++]; \
        }                               \
        (*target)[*it] = data;          \
    }()

namespace mc
{
    namespace json
    {
        struct JsonDocument
        {
        public:
            JsonDocument(std::initializer_list<std::string> header) :
                m_Header(header)
            {
                m_Doc["header"] = header;
            }

            JsonDocument(std::set<std::string> header) :
                m_Header(header)
            {
                m_Doc["header"] = header;
            }

            JsonDocument(std::vector<std::string> header) :
                m_Header(std::set(header.begin(), header.end()))
            {
                m_Doc["header"] = header;
            }

            void ExtendHeader(std::string field)
            {
                m_Header.emplace(field);
                m_Doc["header"] = m_Header;
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            void AddField(const std::string &name, T data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data;
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            void AddField(const std::string &name, std::vector<T> data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data;
            }

            template <typename Scalar, int R, int C>
            void AddField(const std::string &name, const Eigen::Matrix<Scalar, R, C> &data)
            {
                assert(m_Header.contains(name));
                if (Eigen::isFlat(data))
                {
                    std::vector<Scalar> vec(data.size());
                    m_Doc[name] = Eigen::to_std(data);
                }
                else
                {
                    m_Doc[name] = Eigen::toVector2D(data);
                }
            }

            void AddField(const std::string &name, const std::vector<std::unordered_map<std::string, double>> &data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data;
            }

            template <typename Scalar, int R, int C>
            void AddField(const std::string &name,
                          const std::vector<Eigen::Matrix<Scalar, R, C>> &data)
            {
                assert(m_Header.contains(name));

                // Все плоские
                if (std::all_of(data.begin(), data.end(),
                                [](const auto &elem) { return Eigen::isFlat(elem); }))
                {
                    // Все плоские → vector<vector<Scalar>>
                    std::vector<std::vector<Scalar>> vecs(data.size());
                    std::transform(data.begin(), data.end(), vecs.begin(),
                                   [](const Eigen::Vector<Scalar, R> &elem) { return Eigen::to_std(elem); });
                    m_Doc[name] = vecs;
                }
                // Все матрицы
                else if (std::all_of(data.begin(), data.end(),
                                     [](const auto &elem) { return !Eigen::isFlat(elem); }))
                {
                    // Все матрицы → vector<vector<vector<Scalar>>>
                    std::vector<std::vector<std::vector<Scalar>>> vecs(data.size());
                    std::transform(data.begin(), data.end(), vecs.begin(),
                                   [](const Eigen::Matrix<Scalar, R, C> &elem) { return Eigen::toVector2D(elem); });
                    m_Doc[name] = vecs;
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Vector with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            template <typename T, typename Scalar, int R, int C>
            void AddField(const std::string &name,
                          const std::unordered_map<T, std::vector<Eigen::Matrix<Scalar, R, C>>> &data)
            {
                assert(m_Header.contains(name));

                const auto allFlat = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &vec)
                {
                    return std::all_of(vec.begin(), vec.end(),
                                       [](const auto &elem) { return Eigen::isFlat(elem); });
                };

                const auto allNotFlat = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &vec)
                {
                    return std::all_of(vec.begin(), vec.end(),
                                       [](const auto &elem) { return !Eigen::isFlat(elem); });
                };

                // Все векторы
                if (std::all_of(data.begin(), data.end(),
                                [&](const auto &kv) { return allFlat(kv.second); }))
                {
                    // Все векторы
                    std::unordered_map<T, std::vector<std::vector<Scalar>>> vecs;
                    for (const auto &[key, mats] : data)
                    {
                        std::vector<std::vector<Scalar>> inner(mats.size());
                        std::transform(mats.begin(), mats.end(), inner.begin(),
                                       [](const Eigen::Vector<Scalar, R> &elem) { return Eigen::to_std(elem); });
                        vecs.insert({key, std::move(inner)});
                    }
                    m_Doc[name] = vecs;
                }
                // Все матрицы
                else if (std::all_of(data.begin(), data.end(),
                                     [&](const auto &kv) { return allNotFlat(kv.second); }))
                {
                    // Все матрицы
                    std::unordered_map<T, std::vector<std::vector<std::vector<Scalar>>>> vecs;
                    for (const auto &[key, mats] : data)
                    {
                        std::vector<std::vector<std::vector<Scalar>>> inner(mats.size());
                        std::transform(mats.begin(), mats.end(), inner.begin(),
                                       [](const Eigen::Matrix<Scalar, R, C> &elem) { return Eigen::toVector2D(elem); });
                        vecs.insert({key, std::move(inner)});
                    }
                    m_Doc[name] = vecs;
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Map with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            void AddSubField(const std::initializer_list<std::string> &fields, T data)
            {
                ASSIGN_FIELDS(fields, data);
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            void AddSubField(const std::initializer_list<std::string> &fields, std::vector<T> data)
            {
                ASSIGN_FIELDS(fields, data);
            }

            template <typename Scalar, int R, int C>
            void AddSubField(const std::initializer_list<std::string> &fields, const Eigen::Matrix<Scalar, R, C> &data)
            {
                if (Eigen::isFlat(data))
                {
                    ASSIGN_FIELDS(fields, Eigen::to_std(data));
                }
                else
                {
                    ASSIGN_FIELDS(fields, Eigen::toVector2D(data));
                }
            }

            template <typename Scalar, int R, int C>
            void AddSubField(const std::initializer_list<std::string> &fields,
                             std::vector<Eigen::Matrix<Scalar, R, C>> data)
            {
                // Все векторы
                if (algo::all_of(data.begin(), data.end(), [](const Eigen::Matrix<Scalar, R, C> &elem)
                {
                    return Eigen::isFlat(elem);
                }))
                {
                    // Все векторы
                    std::vector<std::vector<Scalar>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Eigen::Vector<Scalar, R> &elem)
                    {
                        return Eigen::to_std(elem);
                    });
                    ASSIGN_FIELDS(fields, vecs);
                }
                // Все матрицы
                else if (algo::all_of(data.begin(), data.end(),
                                      [](const Eigen::Matrix<Scalar, R, C> &elem) { return !Eigen::isFlat(elem); }))
                {
                    // Все матрицы
                    std::vector<std::vector<std::vector<Scalar>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Eigen::Matrix<Scalar, R, C> &elem)
                    {
                        return Eigen::toVector2D(elem);
                    });
                    ASSIGN_FIELDS(fields, vecs);
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Vector with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            template <typename T, typename Scalar, int R, int C>
            void AddSubField(const std::initializer_list<std::string> &fields,
                             std::unordered_map<T, std::vector<Eigen::Matrix<Scalar, R, C>>> data)
            {
                using iterType = std::pair<T, std::vector<Eigen::Matrix<Scalar, R, C>>>;

                const auto allFlat = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &data)
                {
                    return algo::all_of(data.begin(), data.end(), [](const Eigen::Matrix<Scalar, R, C> &elem)
                    {
                        return Eigen::isFlat(elem);
                    });
                };

                const auto allNotFlat = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &data)
                {
                    return algo::none_of(data.begin(), data.end(), [](const Eigen::Matrix<Scalar, R, C> &elem)
                    {
                        return Eigen::isFlat(elem);
                    });
                };

                const auto flat = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &data)
                {
                    std::vector<std::vector<Scalar>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Eigen::Vector<Scalar, R> &elem)
                    {
                        return Eigen::to_std(elem);
                    });
                    return vecs;
                };

                const auto toVec = [](const std::vector<Eigen::Matrix<Scalar, R, C>> &data)
                {
                    std::vector<std::vector<std::vector<Scalar>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Eigen::Matrix<Scalar, R, C> &elem)
                    {
                        return Eigen::toVector2D(elem);
                    });
                    return vecs;
                };

                // Все векторы
                if (algo::all_of(data.begin(), data.end(), [allFlat](const iterType &elem)
                {
                    return allFlat(elem.second);
                }))
                {
                    // Все векторы
                    std::unordered_map<T, std::vector<std::vector<Scalar>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, flat](const iterType &elem)
                    {
                        vecs.insert({elem.first, flat(elem.second)});
                    });
                    ASSIGN_FIELDS(fields, vecs);
                }
                // Все матрицы
                else if (algo::all_of(data.begin(), data.end(), [allNotFlat](const iterType &elem)
                {
                    return allNotFlat(elem.second);
                }))
                {
                    // Все матрицы
                    std::unordered_map<T, std::vector<std::vector<std::vector<Scalar>>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, toVec](const iterType &elem)
                    {
                        vecs.insert({elem.first, toVec(elem.second)});
                    });
                    ASSIGN_FIELDS(fields, vecs);
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Map with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            void Check() const
            {
                for (auto &val : m_Header)
                {
                    assert(
                        m_Doc.contains(val) && std::format("Object '{}' wasn't writed to the json document", val).c_str(
                        ));
                }
            }

            std::string ToString() const
            {
                Check();
                return m_Doc.dump();
            }

            nlohmann::json &GetDoc()
            {
                return m_Doc;
            }

        private:
            std::set<std::string> m_Header;
            nlohmann::json m_Doc;
        };
    }
}

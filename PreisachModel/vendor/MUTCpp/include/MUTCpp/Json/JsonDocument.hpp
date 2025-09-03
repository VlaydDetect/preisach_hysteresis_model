// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cassert>
#include <set>
#include <map>

#include "../Functions/toVector.hpp"
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

            template <typename dtype>
            void AddField(const std::string &name, Matrix<dtype> data)
            {
                assert(m_Header.contains(name));
                if (data.isflat())
                {
                    m_Doc[name] = data.toFlattenVector();
                }
                else
                {
                    m_Doc[name] = mc::toVector(data);
                }
            }

            void AddField(const std::string &name, const std::vector<std::unordered_map<std::string, double>> &data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data;
            }

            template <typename dtype>
            void AddField(const std::string &name, std::vector<Matrix<dtype>> data)
            {
                assert(m_Header.contains(name));

                if (algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem) { return elem.isflat(); }))
                {
                    std::vector<std::vector<dtype>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return elem.toFlattenVector();
                    });
                    m_Doc[name] = vecs;
                }
                else if (algo::all_of(data.begin(), data.end(),
                                      [](const Matrix<dtype> &elem) { return !elem.isflat(); }))
                {
                    std::vector<std::vector<std::vector<dtype>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        auto vec = mc::toVector(elem);
                        return vec;
                    });
                    m_Doc[name] = vecs;
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Vector with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            template <typename T, typename dtype>
            void AddField(const std::string &name, std::unordered_map<T, std::vector<Matrix<dtype>>> data)
            {
                assert(m_Header.contains(name));

                using iterType = std::pair<T, std::vector<Matrix<dtype>>>;

                const auto allFlat = [](const std::vector<Matrix<dtype>> &data)
                {
                    return algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem)
                    {
                        return elem.isflat();
                    });
                };

                const auto allNotFlat = [](const std::vector<Matrix<dtype>> &data)
                {
                    return algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem)
                    {
                        return elem.isflat();
                    });
                };

                const auto flat = [](const std::vector<Matrix<dtype>> &data)
                {
                    std::vector<std::vector<dtype>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return elem.toFlattenVector();
                    });
                    return vecs;
                };

                const auto toVec = [](const std::vector<Matrix<dtype>> &data)
                {
                    std::vector<std::vector<std::vector<dtype>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return mc::toVector(elem);
                    });
                    return vecs;
                };


                if (algo::all_of(data.begin(), data.end(), [allFlat](const iterType &elem)
                {
                    return allFlat(elem.second);
                }))
                {
                    std::unordered_map<T, std::vector<std::vector<dtype>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, flat](const iterType &elem)
                    {
                        vecs.insert({elem.first, flat(elem.second)});
                    });
                    m_Doc[name] = vecs;
                }
                else if (algo::all_of(data.begin(), data.end(), [allNotFlat](const iterType &elem)
                {
                    return allNotFlat(elem.second);
                }))
                {
                    std::unordered_map<T, std::vector<std::vector<std::vector<dtype>>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, toVec](const iterType &elem)
                    {
                        vecs.insert({elem.first, toVec(elem.second)});
                    });
                    m_Doc[name] = vecs;
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Map with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            void AddField(const std::string &name, Vec2 data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data.toArray();
            }

            void AddField(const std::string &name, Vec3 data)
            {
                assert(m_Header.contains(name));
                m_Doc[name] = data.toArray();
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            // void AddSubField(const std::string &field, const std::string &name, T data)
            void AddSubField(const std::initializer_list<std::string> &fields, T data)
            {
                // assert(m_Header.contains(field));
                // m_Doc[field][name] = data;
                ASSIGN_FIELDS(fields, data);
            }

            template <typename T>
                requires std::is_arithmetic_v<T> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, const char *>
            // void AddSubField(const std::string &field, const std::string &name, std::vector<T> data)
            void AddSubField(const std::initializer_list<std::string> &fields, std::vector<T> data)
            {
                // assert(m_Header.contains(field));
                // m_Doc[field][name] = data;
                ASSIGN_FIELDS(fields, data);
            }

            template <typename dtype>
            // void AddSubField(const std::string &field, const std::string &name, Matrix<dtype> data)
            void AddSubField(const std::initializer_list<std::string> &fields, Matrix<dtype> data)
            {
                // assert(m_Header.contains(field));
                if (data.isflat())
                {
                    // m_Doc[field][name] = data.toFlattenVector();
                    ASSIGN_FIELDS(fields, data.toFlattenVector());
                }
                else
                {
                    // m_Doc[field][name] = mc::toVector(data);
                    ASSIGN_FIELDS(fields, mc::toVector(data));
                }
            }

            template <typename dtype>
            // void AddSubField(const std::string &field, const std::string &name, std::vector<Matrix<dtype>> data)
            void AddSubField(const std::initializer_list<std::string> &fields, std::vector<Matrix<dtype>> data)
            {
                // assert(m_Header.contains(field));

                if (algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem) { return elem.isflat(); }))
                {
                    std::vector<std::vector<dtype>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return elem.toFlattenVector();
                    });
                    // m_Doc[field][name] = vecs;
                    ASSIGN_FIELDS(fields, vecs);
                }
                else if (algo::all_of(data.begin(), data.end(),
                                      [](const Matrix<dtype> &elem) { return !elem.isflat(); }))
                {
                    std::vector<std::vector<std::vector<dtype>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return mc::toVector(elem);
                    });
                    // m_Doc[field][name] = vecs;
                    ASSIGN_FIELDS(fields, vecs);
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Vector with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            template <typename T, typename dtype>
            // void AddSubField(const std::string &field, const std::string &name, std::unordered_map<T, std::vector<Matrix<dtype>>> data)
            void AddSubField(const std::initializer_list<std::string> &fields,
                             std::unordered_map<T, std::vector<Matrix<dtype>>> data)
            {
                // assert(m_Header.contains(name));

                using iterType = std::pair<T, std::vector<Matrix<dtype>>>;

                const auto allFlat = [](const std::vector<Matrix<dtype>> &data)
                {
                    return algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem)
                    {
                        return elem.isflat();
                    });
                };

                const auto allNotFlat = [](const std::vector<Matrix<dtype>> &data)
                {
                    return algo::all_of(data.begin(), data.end(), [](const Matrix<dtype> &elem)
                    {
                        return elem.isflat();
                    });
                };

                const auto flat = [](const std::vector<Matrix<dtype>> &data)
                {
                    std::vector<std::vector<dtype>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return elem.toFlattenVector();
                    });
                    return vecs;
                };

                const auto toVec = [](const std::vector<Matrix<dtype>> &data)
                {
                    std::vector<std::vector<std::vector<dtype>>> vecs(data.size());
                    algo::transform(data.begin(), data.end(), vecs.begin(), [](const Matrix<dtype> &elem)
                    {
                        return mc::toVector(elem);
                    });
                    return vecs;
                };


                if (algo::all_of(data.begin(), data.end(), [allFlat](const iterType &elem)
                {
                    return allFlat(elem.second);
                }))
                {
                    std::unordered_map<T, std::vector<std::vector<dtype>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, flat](const iterType &elem)
                    {
                        vecs.insert({elem.first, flat(elem.second)});
                    });
                    // m_Doc[field][name] = vecs;
                    ASSIGN_FIELDS(fields, vecs);
                }
                else if (algo::all_of(data.begin(), data.end(), [allNotFlat](const iterType &elem)
                {
                    return allNotFlat(elem.second);
                }))
                {
                    std::unordered_map<T, std::vector<std::vector<std::vector<dtype>>>> vecs;
                    algo::for_each(data.begin(), data.end(), [&vecs, toVec](const iterType &elem)
                    {
                        vecs.insert({elem.first, toVec(elem.second)});
                    });
                    // m_Doc[field][name] = vecs;
                    ASSIGN_FIELDS(fields, vecs);
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR(
                        "Map with matrices with different dimensions is not supported to json serialization now.");
                }
            }

            // void AddSubField(const std::string &field, const std::string &name, Vec2 data)
            void AddSubField(const std::initializer_list<std::string> &fields, Vec2 data)
            {
                // assert(m_Header.contains(name));
                // m_Doc[field][name] = data.toArray();
                ASSIGN_FIELDS(fields, data.toArray());
            }

            // void AddSubField(const std::string &field, const std::string &name, Vec3 data)
            void AddSubField(const std::initializer_list<std::string> &fields, Vec3 data)
            {
                // assert(m_Header.contains(name));
                // m_Doc[field][name] = data.toArray();
                ASSIGN_FIELDS(fields, data.toArray());
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

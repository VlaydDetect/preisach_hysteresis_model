// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <variant>
#include <string>
#include <unordered_map>
#include <vector>
#include <ranges>

#include <Eigen/Dense>
#include "Eigen/utils.hpp"

#include "Json/JsonDocument.hpp"
#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Hysteresis/ArealModel.hpp"
#include "Hysteresis/DiscreteModel.hpp"
#include "Hysteresis/ModelBase.hpp"

namespace mc
{
    namespace ode
    {
        enum class VoteDataType : uint8
        {
            Null = 0,
            Bool,
            Int,
            Double,
            PreisachModel,
            Vector,
            Matrix,
        };

        struct Vote
        {
        public:
            using ValueVariant = std::variant<std::monostate, bool, int, double, Eigen::MatrixXd, Ref<
                                                  PreisachModelBase>>;

            Vote() :
                m_Value(std::monostate{})
            {
            }

            Vote(bool v) :
                m_Value(v)
            {
            }

            Vote(int v) :
                m_Value(v)
            {
            }

            Vote(double v) :
                m_Value(v)
            {
            }

            template <typename Derived,
                      typename = std::enable_if_t<
                          std::is_base_of_v<Eigen::DenseBase<Derived>, Derived>
                      >>
            Vote(const Eigen::DenseBase<Derived> &m) :
                m_Value(m.derived())
            {
            }

            Vote(const Ref<PreisachModelBase> &m) :
                m_Value(static_cast<Ref<PreisachModelBase>>(m))
            {
            }

            Vote(const Ref<DiscretePreisachModel> &m) :
                m_Value(static_cast<Ref<PreisachModelBase>>(m))
            {
            }

            Vote(const Ref<ArealPreisachModel> &m) :
                m_Value(static_cast<Ref<PreisachModelBase>>(m))
            {
            }

            Vote(const Ref<DoubleArealPreisachModel> &m) :
                m_Value(static_cast<Ref<PreisachModelBase>>(m))
            {
            }

        public:
            VoteDataType Type() const
            {
                return std::visit([]<typename T>(T &&arg) -> VoteDataType
                {
                    using U = std::decay_t<T>;
                    if constexpr (std::is_same_v<U, std::monostate>)
                        return VoteDataType::Null;
                    else if constexpr (std::is_same_v<U, bool>)
                        return VoteDataType::Bool;
                    else if constexpr (std::is_same_v<U, int>)
                        return VoteDataType::Int;
                    else if constexpr (std::is_same_v<U, double>)
                        return VoteDataType::Double;
                    else if constexpr (std::is_same_v<U, Eigen::MatrixXd>)
                        return VoteDataType::Matrix;
                    else if constexpr (std::is_same_v<U, Ref<PreisachModelBase>>)
                        return VoteDataType::PreisachModel;
                    else
                        return VoteDataType::Null;
                }, m_Value);
            }

            bool toBool() const
            {
                return std::visit([]<typename T>(T &&arg) -> bool
                {
                    using U = std::decay_t<T>;
                    if constexpr (std::is_same_v<U, bool>)
                        return arg;
                    else if constexpr (std::is_same_v<U, int>)
                        return static_cast<bool>(arg);
                    else if constexpr (std::is_same_v<U, double>)
                        return static_cast<bool>(arg);
                    else
                    {
                        THROW_RUNTIME_ERROR("Vote: cannot convert to bool");
                        return false;
                    }
                }, m_Value);
            }

            int toInt() const
            {
                return std::visit([]<typename T>(T &&arg) -> int
                {
                    using U = std::decay_t<T>;
                    if constexpr (std::is_same_v<U, bool>)
                        return static_cast<int>(arg);
                    else if constexpr (std::is_same_v<U, int>)
                        return arg;
                    else if constexpr (std::is_same_v<U, double>)
                        return static_cast<int>(arg);
                    else
                    {
                        THROW_RUNTIME_ERROR("Vote: cannot convert to int");
                        return INT32_MAX;
                    }
                }, m_Value);
            }

            double toDouble() const
            {
                AL_PROFILE_SCOPE("Vote::toDouble");
                return std::visit([]<typename T>(T &&arg) -> double
                {
                    using U = std::decay_t<T>;
                    if constexpr (std::is_same_v<U, bool>)
                        return static_cast<double>(arg);
                    else if constexpr (std::is_same_v<U, int>)
                        return static_cast<double>(arg);
                    else if constexpr (std::is_same_v<U, double>)
                        return arg;
                    else
                    {
                        THROW_RUNTIME_ERROR("Vote: cannot convert to double");
                        return consts::nan;
                    }
                }, m_Value);
            }

            Eigen::VectorXd toVector() const
            {
                if (auto pval = std::get_if<Eigen::MatrixXd>(&m_Value))
                {
                    if (pval->cols() == 1)
                        return *pval;
                    if (pval->rows() == 1)
                        return pval->transpose();
                    THROW_RUNTIME_ERROR("Vote: not a Vector");
                }
                THROW_RUNTIME_ERROR("Vote: not a Vector");
            }

            Eigen::MatrixXd toMatrix() const
            {
                if (auto pval = std::get_if<Eigen::MatrixXd>(&m_Value))
                    return *pval;
                THROW_RUNTIME_ERROR("Vote: not a Matrix");
            }

            const Ref<PreisachModelBase> &toPreisachModel() const
            {
                if (auto pval = std::get_if<Ref<PreisachModelBase>>(&m_Value))
                    return *pval;
                THROW_RUNTIME_ERROR("Vote: not a PreisachModel");
            }

            Ref<PreisachModelBase> &toPreisachModel()
            {
                if (auto pval = std::get_if<Ref<PreisachModelBase>>(&m_Value))
                    return *pval;
                THROW_RUNTIME_ERROR("Vote: not a PreisachModel");
            }

            bool isWritableType() const
            {
                return Type() != VoteDataType::PreisachModel || Type() != VoteDataType::Null;
            }

            void WriteFieldToDoc(mc::json::JsonDocument &doc, const std::string &name) const
            {
                if (!isWritableType())
                {
                    THROW_RUNTIME_ERROR("Vote type is not writable to json");
                }

                switch (Type())
                {
                case VoteDataType::Bool:
                {
                    doc.AddField(name, toBool());
                    break;
                }
                case VoteDataType::Int:
                {
                    doc.AddField(name, toInt());
                    break;
                }
                case VoteDataType::Double:
                {
                    doc.AddField(name, toDouble());
                    break;
                }
                case VoteDataType::Vector:
                {
                    doc.AddField(name, toVector());
                    break;
                }
                case VoteDataType::Matrix:
                {
                    doc.AddField(name, toMatrix());
                    break;
                }

                default:
                    THROW_RUNTIME_ERROR("Vote type is not writable to json");
                }
            }

        private:
            ValueVariant m_Value;
        };

        using DSArgs = std::unordered_map<std::string, Vote>;

        inline void WriteVotesToDoc(mc::json::JsonDocument &doc, const std::string &name,
                                    const std::vector<Vote> &votes,
                                    VoteDataType type)
        {
            switch (type)
            {
            case VoteDataType::Bool:
            {
                std::vector<bool> data(votes.size());
                data.reserve(votes.size());
                for (auto &v : votes)
                    data.push_back(v.toBool());
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Int:
            {
                std::vector<int> data(votes.size());
                data.reserve(votes.size());
                for (auto &v : votes)
                    data.push_back(v.toInt());
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Double:
            {
                std::vector<double> data;
                data.reserve(votes.size());
                for (auto &v : votes)
                    data.push_back(v.toDouble());
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Vector:
            {
                std::vector<Eigen::VectorXd> data(votes.size());
                data.reserve(votes.size());
                for (auto &v : votes)
                    data.push_back(v.toVector());
                Eigen::isFlat(data[0]);
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Matrix:
            {
                std::vector<Eigen::MatrixXd> data(votes.size());
                data.reserve(votes.size());
                for (auto &v : votes)
                    data.push_back(v.toMatrix());
                doc.AddField(name, data);
                break;
            }

            default: THROW_RUNTIME_ERROR("Vote type is not writable to json");
            }
        }

        namespace utils
        {
            inline std::vector<Vote> VoteRange(double start, double stop, double step = 1.0)
            {
                const auto values = Eigen::arange(start, stop, step);
                std::vector<Vote> votes;
                votes.reserve(values.size());
                for (auto v : values)
                    votes.emplace_back(v);
                return votes;
            }

            inline std::vector<Vote> VoteRange(int32 start, int32 stop, int32 step = 1.0)
            {
                const auto values = Eigen::arange(start, stop, step);
                std::vector<Vote> votes;
                votes.reserve(values.size());
                for (auto v : values)
                    votes.emplace_back(v);
                return votes;
            }
        }
    }
}

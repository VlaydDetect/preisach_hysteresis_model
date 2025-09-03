// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Json/JsonDocument.hpp"

#include <string>
#include <unordered_map>

#include "Core/Constants.hpp"
#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Functions/arange.hpp"
#include "Hysteresis/ArealModel.hpp"
#include "Hysteresis/DiscreteModel.hpp"
#include "Hysteresis/ModelBase.hpp"

namespace mc
{
    namespace ode
    {
        enum class VoteDataType : uint8
        {
            NUll = 0,
            Bool,
            Int,
            Double,
            PreisachModel,
            Matrix,
        };

        struct Vote
        {
        public:
            Vote() :
                m_Type(VoteDataType::NUll)
            {
            }

            Vote(bool value) :
                m_Type(VoteDataType::Bool), m_BoolValue(value)
            {
            }

            Vote(int value) :
                m_Type(VoteDataType::Int), m_IntValue(value)
            {
            }

            Vote(double value) :
                m_Type(VoteDataType::Double), m_DoubleValue(value)
            {
            }

            Vote(const Matrix<double> &matrix) :
                m_Type(VoteDataType::Matrix), m_Matrix(matrix)
            {
            }

            // explicit Vote(const Ref<PreisachModelBase> &value) :
            //     m_Type(DataType::PreisachModel), m_ModelBaseValue(value)
            // {
            // }

            explicit Vote(const Ref<DiscretePreisachModel> &value) :
                m_Type(VoteDataType::PreisachModel), m_PreisachModelBase(value)
            {
            }

            explicit Vote(const Ref<ArealPreisachModel> &value) :
                m_Type(VoteDataType::PreisachModel), m_PreisachModelBase(value)
            {
            }

            explicit Vote(const Ref<DoubleArealPreisachModel> &value) :
                m_Type(VoteDataType::PreisachModel), m_PreisachModelBase(value)
            {
            }

            // Vote(const Vote&) = delete;
            // Vote& operator=(const Vote&) = delete;

            bool toBool() const
            {
                switch (m_Type)
                {
                case VoteDataType::Bool:
                {
                    return m_BoolValue;
                }
                case VoteDataType::Int:
                {
                    return static_cast<bool>(m_IntValue);
                }
                case VoteDataType::Double:
                {
                    return static_cast<bool>(m_DoubleValue);
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    return false;
                }
                }
            }

            int toInt() const
            {
                switch (m_Type)
                {
                case VoteDataType::Bool:
                {
                    return static_cast<int>(m_BoolValue);
                }
                case VoteDataType::Int:
                {
                    return m_IntValue;
                }
                case VoteDataType::Double:
                {
                    return static_cast<int>(m_DoubleValue);
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    return 0;
                }
                }
            }

            double toDouble() const
            {
                switch (m_Type)
                {
                case VoteDataType::Bool:
                {
                    return static_cast<double>(m_BoolValue);
                }
                case VoteDataType::Int:
                {
                    return static_cast<double>(m_IntValue);
                }
                case VoteDataType::Double:
                {
                    return m_DoubleValue;
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    return 0.0;
                }
                }
            }

            Matrix<double> toMatrix() const
            {
                switch (m_Type)
                {
                case VoteDataType::Matrix:
                {
                    return m_Matrix;
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    return {};
                }
                }
            }

            const Ref<PreisachModelBase> &toPreisachModel() const
            {
                switch (m_Type)
                {
                case VoteDataType::PreisachModel:
                {
                    return m_PreisachModelBase;
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    // return nullptr;
                }
                }
            }

            Ref<PreisachModelBase> &toPreisachModel()
            {
                switch (m_Type)
                {
                case VoteDataType::PreisachModel:
                {
                    return m_PreisachModelBase;
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    // return nullptr;
                }
                }
            }

            bool isWritableType() const
            {
                return m_Type != VoteDataType::PreisachModel || m_Type != VoteDataType::NUll;
            }

            void WriteFieldToDoc(mc::json::JsonDocument &doc, const std::string &name) const
            {
                if (!isWritableType())
                {
                    THROW_RUNTIME_ERROR("Vote type is not writable to json");
                }

                switch (m_Type)
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
                case VoteDataType::Matrix:
                {
                    doc.AddField(name, toMatrix());
                    break;
                }

                default:
                    THROW_RUNTIME_ERROR("Vote type is not writable to json");
                }
            }

            VoteDataType Type() const { return m_Type; }

        private:
            VoteDataType m_Type;
            int m_BoolValue{false};
            int m_IntValue{0};
            double m_DoubleValue{consts::nan};
            Matrix<double> m_Matrix;
            Ref<PreisachModelBase> m_PreisachModelBase;
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
                std::ranges::transform(votes, data.begin(), [](const Vote &vote) { return vote.toBool(); });
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Int:
            {
                std::vector<int> data(votes.size());
                std::ranges::transform(votes, data.begin(), [](const Vote &vote) { return vote.toInt(); });
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Double:
            {
                std::vector<double> data(votes.size());
                std::ranges::transform(votes, data.begin(), [](const Vote &vote) { return vote.toDouble(); });
                doc.AddField(name, data);
                break;
            }
            case VoteDataType::Matrix:
            {
                std::vector<Matrix<double>> data(votes.size());
                std::ranges::transform(votes, data.begin(), [](const Vote &vote) { return vote.toMatrix(); });
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
                std::vector<double> values = mc::arange(start, stop, step).toFlattenVector();
                std::vector<Vote> votes;
                std::ranges::transform(values, std::back_inserter(votes), [](const double v) { return Vote(v); });
                return votes;
            }

            inline std::vector<Vote> VoteRange(int32 start, int32 stop, int32 step = 1.0)
            {
                std::vector<int32> values = mc::arange(start, stop, step).toFlattenVector();
                std::vector<Vote> votes;
                std::ranges::transform(values, std::back_inserter(votes), [](const double v) { return Vote(v); });
                return votes;
            }
        }
    }
}

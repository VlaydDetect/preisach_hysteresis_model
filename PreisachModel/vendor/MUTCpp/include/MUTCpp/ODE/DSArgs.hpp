// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>
#include <unordered_map>

#include "Core/Constants.hpp"
#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Hysteresis/ModelBase.hpp"

namespace mc
{
    namespace ode
    {
        struct Vote
        {
        public:
            Vote() :
                m_Type(DataType::NUll)
            {
            }

            Vote(bool value) :
                m_Type(DataType::Bool), m_BoolValue(value)
            {
            }

            Vote(int value) :
                m_Type(DataType::Int), m_IntValue(value)
            {
            }

            Vote(double value) :
                m_Type(DataType::Double), m_DoubleValue(value)
            {
            }

            Vote(const Matrix<double> &matrix) :
                m_Type(DataType::Matrix), m_Matrix(matrix)
            {
            }

            explicit Vote(const Ref<PreisachModelBase> &value) :
                m_Type(DataType::PreisachModel), m_ModelBaseValue(value)
            {
            }

            explicit Vote(const Ref<ArealPreisachModel> &value) :
                m_Type(DataType::PreisachModel), m_ModelBaseValue(value)
            {
            }

            // Vote(const Vote&) = delete;
            // Vote& operator=(const Vote&) = delete;

            bool toBool() const
            {
                switch (m_Type)
                {
                case DataType::Bool:
                {
                    return m_BoolValue;
                }
                case DataType::Int:
                {
                    return static_cast<bool>(m_IntValue);
                }
                case DataType::Double:
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
                case DataType::Bool:
                {
                    return static_cast<int>(m_BoolValue);
                }
                case DataType::Int:
                {
                    return m_IntValue;
                }
                case DataType::Double:
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
                case DataType::Bool:
                {
                    return static_cast<double>(m_BoolValue);
                }
                case DataType::Int:
                {
                    return static_cast<double>(m_IntValue);
                }
                case DataType::Double:
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
                case DataType::Matrix:
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
                case DataType::PreisachModel:
                {
                    return m_ModelBaseValue;
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
                case DataType::PreisachModel:
                {
                    return m_ModelBaseValue;
                }
                default:
                {
                    THROW_RUNTIME_ERROR("");
                    // return nullptr;
                }
                }
            }

        private:
            enum class DataType : uint8
            {
                NUll = 0,
                Bool,
                Int,
                Double,
                PreisachModel,
                Matrix,
            };

            DataType m_Type;
            int m_BoolValue{false};
            int m_IntValue{0};
            double m_DoubleValue{consts::nan};
            Matrix<double> m_Matrix;
            Ref<PreisachModelBase> m_ModelBaseValue;
        };

        using DSArgs = std::unordered_map<std::string, Vote>;
    }
}

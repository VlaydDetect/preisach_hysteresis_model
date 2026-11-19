// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include "Core/Constants.hpp"
#include "MUTCpp/Utils/Profile.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/mathOperations.hpp"

#include <Eigen/Dense>

namespace mc
{
    struct PreisachThreshold
    {
        double level;
        std::optional<Eigen::Index> extremum;
        bool operator==(const PreisachThreshold &) const = default;
    };

    class PreisachModelBase
    {
    public:
        explicit PreisachModelBase(double L) : m_L(L) {}

        void ResetState()
        {
            m_History = {};
            m_CurrentInput = consts::nan;
            m_PreviousOutput = consts::nan;
            m_Increasing = true;
            ResetImpl();
        }

        virtual ~PreisachModelBase() = default;
        PreisachModelBase(const PreisachModelBase &) = delete;
        PreisachModelBase &operator=(const PreisachModelBase &) = delete;

        std::tuple<std::vector<double>, std::vector<double>> HysteresisLoop() const
        {
            return {m_History.inputs, m_History.outputs};
        }

        std::array<std::vector<double>, 2> DerivativeHistory() const
        {
            return {m_History.inputs, m_History.slopes};
        }

        double GetL() const
        {
            return m_L;
        }

        virtual double GetMaxArea() const = 0;

        virtual double P(double u, double v, int = -1)
        {
            const double output = AdvanceInput(u, v);
            RecordCurrentSample();
            return output;
        }

        virtual double AdvanceInput(double u, double v) = 0;

        virtual double DerivativeOperator(double, double) { return CurrentSlope(); }
        double DerivativeOperator(int32_t index) const { return m_History.slopes.at(index); }
        virtual double CurrentSignedSlope() const = 0;
        virtual double CurrentSlope() const = 0;
        virtual bool SupportsContinuousEvents() const { return false; }
        virtual void AppendInputSurfaces(std::vector<PreisachThreshold> &, bool, Eigen::Index = 0) const {}
        virtual Eigen::Index ExtremaCount() const { return 0; }
        virtual void AppendExtrema(std::vector<double> &) const {}
        virtual Eigen::MatrixXd TransferExtrema(const PreisachModelBase &, const Eigen::MatrixXd &,
                                                 const Eigen::RowVectorXd &) const
        {
            throw std::logic_error("Preisach model does not support extrema sensitivities");
        }
        virtual Eigen::RowVectorXd OutputVariation(const Eigen::MatrixXd &, const Eigen::RowVectorXd &) const
        {
            throw std::logic_error("Preisach model does not support output sensitivities");
        }

        virtual std::unique_ptr<PreisachModelBase> CloneWorkingState() const
        {
            throw std::logic_error("Preisach model does not support working-state snapshots");
        }

        void InitializeMemory(std::span<const double> inputs)
        {
            for (double input : inputs)
            {
                if (!std::isfinite(input))
                {
                    throw std::invalid_argument("Initial Preisach inputs must be finite");
                }
            }
            ResetState();
            for (double input : inputs)
            {
                AdvanceInput(input, 0.0);
            }
        }

        double GetCurrentInput() const { return m_CurrentInput; }
        double GetCurrentOutput() const { return m_PreviousOutput; }
        std::uint64_t ConfigurationRevision() const { return m_ConfigurationRevision; }
        void SetRecordHistory(bool record) { m_RecordHistory = record; }

        void RecordCurrentSample()
        {
            if (m_RecordHistory)
            {
                m_History.Append(m_CurrentInput, m_PreviousOutput, CurrentSlope());
            }
        }

        void AcceptWorkingState(PreisachModelBase &prepared, bool record = true)
        {
            ValidateWorkingState(prepared);
            if (record && m_RecordHistory)
            {
                m_History.Append(prepared.m_CurrentInput, prepared.m_PreviousOutput, prepared.CurrentSlope());
            }
            SwapAcceptedState(prepared);
        }

        void AcceptWorkingSequence(std::span<const std::shared_ptr<PreisachModelBase>> prepared)
        {
            if (prepared.empty())
            {
                return;
            }
            const auto size = m_History.inputs.size();
            try
            {
                for (const auto &sample : prepared)
                {
                    if (!sample)
                    {
                        throw std::invalid_argument("Missing prepared Preisach state");
                    }
                    ValidateWorkingState(*sample);
                    if (m_RecordHistory)
                    {
                        m_History.Append(sample->m_CurrentInput, sample->m_PreviousOutput, sample->CurrentSlope());
                    }
                }
            }
            catch (...)
            {
                m_History.Truncate(size);
                throw;
            }
            SwapAcceptedState(*prepared.back());
        }

    protected:
        virtual void ResetImpl() = 0;
        virtual void SwapWorkingMemory(PreisachModelBase &other) noexcept = 0;

        void CopyBaseWorkingStateTo(PreisachModelBase &copy) const
        {
            copy.m_CurrentInput = m_CurrentInput;
            copy.m_PreviousOutput = m_PreviousOutput;
            copy.m_Increasing = m_Increasing;
            copy.m_RecordHistory = m_RecordHistory;
            copy.m_ConfigurationRevision = m_ConfigurationRevision;
        }

        void SetCurrentInput(double u)
        {
            if (std::isfinite(m_CurrentInput) && u != m_CurrentInput)
            {
                m_Increasing = u > m_CurrentInput;
            }
            m_CurrentInput = u;
        }

        double m_L;
        double m_CurrentInput = consts::nan;
        double m_PreviousOutput = consts::nan;
        bool m_Increasing = true;
        std::uint64_t m_ConfigurationRevision = 0;

    private:
        void ValidateWorkingState(const PreisachModelBase &prepared) const
        {
            if (typeid(prepared) != typeid(*this) || prepared.m_ConfigurationRevision != m_ConfigurationRevision)
            {
                throw std::invalid_argument("Preisach snapshot has a different model type or configuration revision");
            }
        }

        void SwapAcceptedState(PreisachModelBase &prepared) noexcept
        {
            std::swap(m_CurrentInput, prepared.m_CurrentInput);
            std::swap(m_PreviousOutput, prepared.m_PreviousOutput);
            std::swap(m_Increasing, prepared.m_Increasing);
            SwapWorkingMemory(prepared);
        }

        struct History
        {
            std::vector<double> inputs, outputs, slopes;

            void Append(double input, double output, double slope)
            {
                const auto size = inputs.size();
                try
                {
                    inputs.push_back(input);
                    outputs.push_back(output);
                    slopes.push_back(slope);
                }
                catch (...)
                {
                    Truncate(size);
                    throw;
                }
            }

            void Truncate(std::size_t size)
            {
                inputs.resize(size);
                outputs.resize(size);
                slopes.resize(size);
            }
        };

        History m_History;
        bool m_RecordHistory = true;
    };

    class SinglePreisachModel : public PreisachModelBase
    {
    public:
        SinglePreisachModel(double L, bool = false, bool = false) : PreisachModelBase(L) {}

        void SetBounds(std::array<double, 2> bounds)
        {
            m_Bounds = bounds;
            ++m_ConfigurationRevision;
        }

        std::array<double, 2> GetBounds() const
        {
            return m_Bounds;
        }

        virtual double GetMaxArea() const override
        {
            return 2 * utils::power(m_L, 2);
        }

        std::unique_ptr<PreisachModelBase> CloneWorkingState() const override
        {
            return CloneSingleWorkingState();
        }

        virtual std::unique_ptr<SinglePreisachModel> CloneSingleWorkingState() const = 0;

        double AdvanceInput(const double u, const double) override
        {
            AL_PROFILE_FUNC("PreisachModelBase::P");
            if (std::isnan(m_PreviousInput))
            {
                m_InterfaceMax.push_back(u);
                m_PreviousInput = u;
                m_FirstElemType = ElementType::Max;
                m_LastElemType = ElementType::Max;
            }

            SetCurrentInput(u);

            if (u < m_Bounds[0] || utils::essentiallyEqual(u, m_Bounds[0]))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::LowerBound");
                KeepMinOnly(m_Bounds[0]);

                m_PreviousInput = m_Bounds[0];
                return m_PreviousOutput = -GetMaxArea();
            }
            if (u > m_Bounds[1] || utils::essentiallyEqual(u, m_Bounds[1]))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::UpperBound");
                KeepMaxOnly(m_Bounds[1]);

                m_PreviousInput = m_Bounds[1];
                return m_PreviousOutput = GetMaxArea();
            }

            // ----------- Simplest cleanup -----------
            if (!m_InterfaceMax.empty() && m_FirstElemType == ElementType::Max && u > m_InterfaceMax[0])
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::MaxOnly");
                KeepMaxOnly(u);
            }
            else if (!m_InterfaceMin.empty() && m_FirstElemType == ElementType::Min && u < m_InterfaceMin[0])
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::MinOnly");
                KeepMinOnly(u);
            }
            // ----------------------------------------

            // ----------- Check extrema -----------
            else if (u > m_PreviousInput)
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::Max");
                CleanupInterfacesByMax(u);

                m_LastElemType = ElementType::Max;
            }
            else if (u < m_PreviousInput)
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::Min");
                CleanupInterfacesByMin(u);

                m_LastElemType = ElementType::Min;
            }
            // -------------------------------------

            m_PreviousInput = u;
            return m_PreviousOutput = P_Impl(u);
        }

        double CurrentSignedSlope() const override
        {
            if (!std::isfinite(m_CurrentInput) || m_CurrentInput < m_Bounds[0] || m_CurrentInput > m_Bounds[1])
                return 0.0;

            const double extremum = m_LastElemType == ElementType::Max
                ? (m_InterfaceMin.empty() ? m_Bounds[0] : m_InterfaceMin.back())
                : (m_InterfaceMax.empty() ? m_Bounds[1] : m_InterfaceMax.back());
            return m_CurrentInput - extremum;
        }

        double CurrentSlope() const override { return 0.0; }

        void AppendInputSurfaces(std::vector<PreisachThreshold> &surfaces, bool increasing,
                                 Eigen::Index offset) const override
        {
            surfaces.push_back({m_Bounds[0], {}});
            surfaces.push_back({m_Bounds[1], {}});
            const auto &extrema = increasing ? m_InterfaceMax : m_InterfaceMin;
            const bool running = (m_LastElemType == ElementType::Max) == increasing;
            if (extrema.size() > static_cast<std::size_t>(running))
            {
                const auto index = extrema.size() - 1 - running;
                const double threshold = extrema[index];
                if (m_Bounds[0] < threshold && threshold < m_Bounds[1])
                {
                    const auto first = increasing ? 0 : static_cast<Eigen::Index>(m_InterfaceMax.size());
                    surfaces.push_back({threshold, offset + first + static_cast<Eigen::Index>(index)});
                }
            }
        }

        Eigen::Index ExtremaCount() const override
        {
            return static_cast<Eigen::Index>(m_InterfaceMax.size() + m_InterfaceMin.size());
        }

        void AppendExtrema(std::vector<double> &values) const override
        {
            values.insert(values.end(), m_InterfaceMax.begin(), m_InterfaceMax.end());
            values.insert(values.end(), m_InterfaceMin.begin(), m_InterfaceMin.end());
        }

        Eigen::MatrixXd TransferExtrema(const PreisachModelBase &previous, const Eigen::MatrixXd &directions,
                                       const Eigen::RowVectorXd &input) const override
        {
            const auto &old = dynamic_cast<const SinglePreisachModel &>(previous);
            Eigen::MatrixXd moved = Eigen::MatrixXd::Zero(ExtremaCount(), input.size());
            const auto transfer = [&](const std::vector<double> &current, const std::vector<double> &prior,
                                      Eigen::Index first, Eigen::Index old_first, ElementType type)
            {
                for (std::size_t i = 0; i < current.size(); ++i)
                {
                    const auto row = first + static_cast<Eigen::Index>(i);
                    const double value = current[i];
                    if (value <= m_Bounds[0] || value >= m_Bounds[1])
                    {
                        continue;
                    }
                    if (type == m_LastElemType && i + 1 == current.size() && value == m_CurrentInput)
                    {
                        moved.row(row) = input;
                        continue;
                    }
                    const auto old_value = std::find(prior.begin(), prior.end(), value);
                    if (old_value == prior.end())
                    {
                        throw std::runtime_error("Cannot associate a surviving Preisach extremum");
                    }
                    moved.row(row) = directions.row(old_first + (old_value - prior.begin()));
                }
            };
            transfer(m_InterfaceMax, old.m_InterfaceMax, 0, 0, ElementType::Max);
            transfer(m_InterfaceMin, old.m_InterfaceMin, static_cast<Eigen::Index>(m_InterfaceMax.size()),
                     static_cast<Eigen::Index>(old.m_InterfaceMax.size()), ElementType::Min);
            return moved;
        }

        void CleanupState()
        {
            m_InterfaceMax = {};
            m_InterfaceMin = {};
        }

    protected:
        void CopyWorkingStateTo(SinglePreisachModel &copy) const
        {
            CopyBaseWorkingStateTo(copy);
            copy.m_Bounds = m_Bounds;
            copy.m_PreviousInput = m_PreviousInput;
            copy.m_InterfaceMax = m_InterfaceMax;
            copy.m_InterfaceMin = m_InterfaceMin;
            copy.m_FirstElemType = m_FirstElemType;
            copy.m_LastElemType = m_LastElemType;
        }

        void SwapWorkingMemory(PreisachModelBase &other) noexcept override
        {
            auto &single = static_cast<SinglePreisachModel &>(other);
            std::swap(m_PreviousInput, single.m_PreviousInput);
            m_InterfaceMax.swap(single.m_InterfaceMax);
            m_InterfaceMin.swap(single.m_InterfaceMin);
            std::swap(m_FirstElemType, single.m_FirstElemType);
            std::swap(m_LastElemType, single.m_LastElemType);
        }

        virtual double P_Impl(double u) = 0;

        virtual void ResetImpl() override
        {
            m_PreviousInput = consts::nan;
            m_InterfaceMax = {};
            m_InterfaceMin = {};
            m_FirstElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }

        void CleanupInterfacesByMax(const double u)
        {
            AL_PROFILE_FUNC("PreisachModelBase::CleanupInterfacesByMax");
            switch (m_FirstElemType)
            {
            case ElementType::Max:
            {
                const auto pos = std::distance(m_InterfaceMax.begin(),
                                               std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
                m_InterfaceMax.insert(m_InterfaceMax.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos + 1, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos, m_InterfaceMin.end());

                break;
            }
            case ElementType::Min:
            {
                const auto pos = std::distance(m_InterfaceMax.begin(),
                                               std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
                m_InterfaceMax.insert(m_InterfaceMax.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos + 1, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos + 1, m_InterfaceMin.end());

                break;
            }
            }
        }

        void CleanupInterfacesByMin(const double u)
        {
            AL_PROFILE_FUNC("PreisachModelBase::CleanupInterfacesByMin");
            switch (m_FirstElemType)
            {
            case ElementType::Max:
            {
                const auto pos = std::distance(m_InterfaceMin.begin(), std::ranges::lower_bound(m_InterfaceMin, u));
                m_InterfaceMin.insert(m_InterfaceMin.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos + 1, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos + 1, m_InterfaceMin.end());

                break;
            }
            case ElementType::Min:
            {
                const auto pos = std::distance(m_InterfaceMin.begin(), std::ranges::lower_bound(m_InterfaceMin, u));
                m_InterfaceMin.insert(m_InterfaceMin.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos + 1, m_InterfaceMin.end());

                break;
            }
            }
        }

        void KeepMaxOnly(double u)
        {
            m_InterfaceMin = {};
            m_InterfaceMax = {u};

            m_FirstElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }

        void KeepMinOnly(double u)
        {
            m_InterfaceMin = {u};
            m_InterfaceMax = {};

            m_FirstElemType = ElementType::Min;
            m_LastElemType = ElementType::Min;
        }

    protected:
        std::array<double, 2> m_Bounds = {-m_L, m_L};
        double m_PreviousInput = consts::nan;
        std::vector<double> m_InterfaceMax{}, m_InterfaceMin{};

        enum class ElementType : bool
        {
            Min = false,
            Max = true,
        };

        ElementType m_FirstElemType = ElementType::Max;
        ElementType m_LastElemType = ElementType::Max;
    };

    class DoublePreisachModel : public PreisachModelBase
    {
    public:
        DoublePreisachModel(double L, double d, double k, bool = false) :
            PreisachModelBase(L), m_D({d, -d}), m_K(k)
        {
            ValidateGap(d);
        }

        double GetD() const
        {
            return m_D[0];
        }

        void SetD(double d)
        {
            ValidateGap(d);
            m_D = {d, -d};
            m_UpperModel->SetBounds({d, 2.0 * m_L + d});
            m_LowerModel->SetBounds({-2.0 * m_L - d, -d});
            ++m_ConfigurationRevision;
            ResetState();
        }

        double AdvanceInput(double u, double v) override
        {
            SetCurrentInput(u);
            const double p1 = m_UpperModel->AdvanceInput(u, v);
            const double p2 = m_LowerModel->AdvanceInput(u, v);
            if (m_D[1] < u && u < m_D[0])
            {
                return m_PreviousOutput = m_K * u;
            }
            const double offset = m_LowerModel->GetMaxArea() - m_UpperModel->GetMaxArea();
            return m_PreviousOutput = p1 + p2 - offset + m_K * std::clamp(u, m_D[1], m_D[0]);
        }

        double CurrentSignedSlope() const override
        {
            return m_UpperModel->CurrentSignedSlope() + m_LowerModel->CurrentSignedSlope() + m_K;
        }

        double CurrentSlope() const override
        {
            return IsLinearBranch() ? m_K : m_UpperModel->CurrentSlope() + m_LowerModel->CurrentSlope();
        }

        bool SupportsContinuousEvents() const override
        {
            return m_UpperModel->SupportsContinuousEvents() && m_LowerModel->SupportsContinuousEvents();
        }

        void AppendInputSurfaces(std::vector<PreisachThreshold> &surfaces, bool increasing,
                                 Eigen::Index offset) const override
        {
            m_UpperModel->AppendInputSurfaces(surfaces, increasing, offset);
            m_LowerModel->AppendInputSurfaces(surfaces, increasing, offset + m_UpperModel->ExtremaCount());
        }

        Eigen::Index ExtremaCount() const override
        {
            return m_UpperModel->ExtremaCount() + m_LowerModel->ExtremaCount();
        }

        void AppendExtrema(std::vector<double> &values) const override
        {
            m_UpperModel->AppendExtrema(values);
            m_LowerModel->AppendExtrema(values);
        }

        Eigen::MatrixXd TransferExtrema(const PreisachModelBase &previous, const Eigen::MatrixXd &directions,
                                       const Eigen::RowVectorXd &input) const override
        {
            const auto &old = dynamic_cast<const DoublePreisachModel &>(previous);
            Eigen::MatrixXd result(ExtremaCount(), input.size());
            result.topRows(m_UpperModel->ExtremaCount()) = m_UpperModel->TransferExtrema(
                *old.m_UpperModel, directions.topRows(old.m_UpperModel->ExtremaCount()), input);
            result.bottomRows(m_LowerModel->ExtremaCount()) = m_LowerModel->TransferExtrema(
                *old.m_LowerModel, directions.bottomRows(old.m_LowerModel->ExtremaCount()), input);
            return result;
        }

        Eigen::RowVectorXd OutputVariation(const Eigen::MatrixXd &directions,
                                           const Eigen::RowVectorXd &input) const override
        {
            if (IsLinearBranch())
            {
                return m_K * input;
            }
            return m_UpperModel->OutputVariation(directions.topRows(m_UpperModel->ExtremaCount()), input) +
                   m_LowerModel->OutputVariation(directions.bottomRows(m_LowerModel->ExtremaCount()), input);
        }

    protected:
        void SwapWorkingMemory(PreisachModelBase &other) noexcept override
        {
            auto &model = static_cast<DoublePreisachModel &>(other);
            m_UpperModel.swap(model.m_UpperModel);
            m_LowerModel.swap(model.m_LowerModel);
        }

        virtual void ResetImpl() override
        {
            m_UpperModel->ResetState();
            m_LowerModel->ResetState();
        }

    protected:
        std::unique_ptr<SinglePreisachModel> m_UpperModel;
        std::unique_ptr<SinglePreisachModel> m_LowerModel;

        std::array<double, 2> m_D;
        double m_K;

    private:
        bool IsLinearBranch() const
        {
            const double u = m_CurrentInput;
            return (m_D[1] < u && u < m_D[0]) || (m_D[0] > 0 &&
                ((u == m_D[1] && m_Increasing) || (u == m_D[0] && !m_Increasing)));
        }

        void ValidateGap(double d) const
        {
            if (!std::isfinite(d) || d < 0.0 || !std::isfinite(2.0 * m_L + d))
            {
                throw std::invalid_argument("Invalid double Preisach gap");
            }
        }
    };
}

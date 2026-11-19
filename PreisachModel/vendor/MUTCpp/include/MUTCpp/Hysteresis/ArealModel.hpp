// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "ModelBase.hpp"
#include "PreisachMeasure.hpp"

namespace mc
{
    class ArealPreisachModel : public SinglePreisachModel
    {
    public:
        ArealPreisachModel(const double L, bool keepDerivative = false, bool keepAnimation = false,
                           bool isDownZero = false) :
            SinglePreisachModel(L, keepDerivative, keepAnimation), m_IsDownZero(isDownZero)
        {
        }

        ArealPreisachModel(const ArealPreisachModel &) = delete;
        ArealPreisachModel &operator=(const ArealPreisachModel &) = delete;

        std::unique_ptr<SinglePreisachModel> CloneSingleWorkingState() const override
        {
            auto copy = std::make_unique<ArealPreisachModel>(m_L, false, false, m_IsDownZero);
            CopyWorkingStateTo(*copy);
            copy->m_PreviousArea = m_PreviousArea;
            copy->m_Measure = m_Measure;
            return copy;
        }

        void SetMeasure(std::shared_ptr<const PreisachMeasure> measure)
        {
            m_Measure = std::move(measure);
            ++m_ConfigurationRevision;
            ResetState();
        }

        virtual double GetMaxArea() const override
        {
            return Measure().TriangleMass(m_Bounds[0], m_Bounds[1]);
        }

        double CurrentSlope() const override
        {
            const double u = m_CurrentInput;
            if (!(m_Bounds[0] < u && u < m_Bounds[1]))
            {
                return 0.0;
            }
            const double scale = m_IsDownZero && !m_Measure ? 1.0 : 2.0;
            if (m_LastElemType == ElementType::Max)
            {
                const double minimum = m_InterfaceMin.empty() ? m_Bounds[0] : m_InterfaceMin.back();
                return scale * Measure().BoundaryDerivatives(minimum, u)[1];
            }
            const double maximum = m_InterfaceMax.empty() ? m_Bounds[1] : m_InterfaceMax.back();
            return -scale * Measure().BoundaryDerivatives(u, maximum)[0];
        }

        bool SupportsContinuousEvents() const override { return !m_IsDownZero; }

        Eigen::RowVectorXd OutputVariation(const Eigen::MatrixXd &directions,
                                           const Eigen::RowVectorXd &input) const override
        {
            Eigen::RowVectorXd variation = Eigen::RowVectorXd::Zero(input.size());
            if (m_Bounds[0] < m_CurrentInput && m_CurrentInput < m_Bounds[1])
            {
                VisitTriangles([&](double sign, double lo, double hi, Eigen::Index lower, Eigen::Index upper)
                {
                    const auto partial = Measure().BoundaryDerivatives(lo, hi);
                    variation += (2.0 * sign * partial[1]) * directions.row(upper);
                    if (lower >= 0)
                    {
                        variation += (2.0 * sign * partial[0]) * directions.row(lower);
                    }
                });
            }
            return variation;
        }

    protected:
        void SwapWorkingMemory(PreisachModelBase &other) noexcept override
        {
            SinglePreisachModel::SwapWorkingMemory(other);
            std::swap(m_PreviousArea, static_cast<ArealPreisachModel &>(other).m_PreviousArea);
        }

        virtual double P_Impl(double) override
        {
            AL_PROFILE_FUNC("ArealPreisachModel::P_Impl");
            const double positive = EverettPositiveMass();
            const double result = m_IsDownZero && !m_Measure ? positive - m_PreviousArea
                                                           : 2.0 * positive - GetMaxArea();
            m_PreviousArea = positive;
            return result;
        }

        virtual void ResetImpl() override
        {
            SinglePreisachModel::ResetImpl();
            m_PreviousArea = 0.0;
        }

    private:
        const PreisachMeasure &Measure() const
        {
            static const UniformPreisachMeasure uniform;
            return m_Measure ? *m_Measure : uniform;
        }

        template <typename Visit>
        void VisitTriangles(const Visit &visit) const
        {
            const auto min_begin = static_cast<Eigen::Index>(m_InterfaceMax.size());
            const bool starts_at_min = m_FirstElemType == ElementType::Min;
            if (starts_at_min && m_InterfaceMin.empty())
            {
                return;
            }
            double left = starts_at_min ? m_InterfaceMin.front() : m_Bounds[0];
            Eigen::Index left_row = starts_at_min ? min_begin : -1;
            std::size_t minimum = starts_at_min ? 1 : 0;
            for (Eigen::Index upper = 0; upper < min_begin; ++upper)
            {
                const double right = m_InterfaceMax[upper];
                visit(1.0, left, right, left_row, upper);
                if (minimum < m_InterfaceMin.size())
                {
                    left = m_InterfaceMin[minimum];
                    left_row = min_begin + static_cast<Eigen::Index>(minimum++);
                    visit(-1.0, left, right, left_row, upper);
                }
            }
        }

        double EverettPositiveMass() const
        {
            ddouble positive = 0.0L;
            VisitTriangles([&](double sign, double lo, double hi, Eigen::Index, Eigen::Index)
            {
                if (lo < hi)
                {
                    const double mass = Measure().TriangleMass(lo, hi);
                    assert(std::isfinite(mass));
                    positive += sign * std::max(0.0, mass);
                }
            });
            // Alternating sums can be slightly negative near lower saturation.
            const ddouble tolerance = 128.0L * std::numeric_limits<double>::epsilon() *
                                      std::max(1.0L, std::abs(positive));
            if (positive < 0.0L && positive >= -tolerance)
            {
                positive = 0.0L;
            }
            assert(positive >= 0.0L);
            return static_cast<double>(positive);
        }

    private:
        double m_PreviousArea = 0.0;
        bool m_IsDownZero = false;

        std::shared_ptr<const PreisachMeasure> m_Measure;
    };

    class DoubleArealPreisachModel : public DoublePreisachModel
    {
    public:
        DoubleArealPreisachModel(double L, double d, double k, bool keepDerivative = false) :
            DoublePreisachModel(L, d, k, keepDerivative)
        {
            m_UpperModel = std::make_unique<ArealPreisachModel>(m_L, keepDerivative);
            m_LowerModel = std::make_unique<ArealPreisachModel>(m_L, keepDerivative);
            SetD(d);
        }

        void SetMeasure(std::shared_ptr<const PreisachMeasure> measure)
        {
            static_cast<ArealPreisachModel &>(*m_UpperModel).SetMeasure(measure);
            static_cast<ArealPreisachModel &>(*m_LowerModel).SetMeasure(std::move(measure));
            ++m_ConfigurationRevision;
            ResetState();
        }

        std::unique_ptr<PreisachModelBase> CloneWorkingState() const override
        {
            auto copy = std::make_unique<DoubleArealPreisachModel>(m_L, GetD(), m_K);
            copy->m_UpperModel = m_UpperModel->CloneSingleWorkingState();
            copy->m_LowerModel = m_LowerModel->CloneSingleWorkingState();
            CopyBaseWorkingStateTo(*copy);
            return copy;
        }

        virtual double GetMaxArea() const override
        {
            return m_UpperModel->GetMaxArea() + m_LowerModel->GetMaxArea();
        }
    };
}

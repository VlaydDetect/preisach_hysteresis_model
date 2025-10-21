// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/isnan.hpp"
#include "Core/Constants.hpp"
#include "Debug/Profile.hpp"
#include "Ref/Ref.hpp"

#include <Eigen/Dense>

namespace mc
{
    using Vec = Eigen::VectorXd;
    
    class PreisachModelBase : public RefCounted
    {
    public:
        PreisachModelBase(double L) :
            m_L(L)
        {
            m_HistoryU = {};
            m_HistoryOutput = {};
            m_HistoryDerivative = {};
        }

        void ResetState()
        {
            m_HistoryU = {};
            m_HistoryOutput = {};
            m_HistoryDerivative = {};
            ResetImpl();
        }

        PreisachModelBase(const PreisachModelBase &) = delete;
        PreisachModelBase &operator=(const PreisachModelBase &) = delete;

        // virtual ~PreisachModelBase() override = default;

        // std::array<Matrix<double>, 2> HysteresisLoop() const
        std::array<std::vector<double>, 2> HysteresisLoop() const
        {
            return {m_HistoryU, m_HistoryOutput};
        }

        // std::array<Matrix<double>, 2> DerivativeHistory() const
        std::array<std::vector<double>, 2> DerivativeHistory() const
        {
            return {m_HistoryU, m_HistoryDerivative};
        }

        double GetL() const
        {
            return m_L;
        }

        virtual double GetMaxArea() const = 0;

        virtual double P(double u, int i = -1) = 0;

        virtual double DerivativeOperator(double t, double dt) = 0;

    protected:
        virtual void ResetImpl() = 0;

    protected:
        double m_L;

        std::vector<double> m_HistoryU{}, m_HistoryOutput{}, m_HistoryDerivative{};
    };

    class SinglePreisachModel : public PreisachModelBase
    {
    public:
        SinglePreisachModel(double L, bool keepDerivative = false, bool keepAnimation = false) :
            PreisachModelBase(L), m_KeepDerivative(keepDerivative), m_KeepAnimation(keepAnimation)
        {
            AL_PROFILE_FUNC("PreisachModelBase::Constructor");

            m_HistoryInterfaceMin = {};
            m_HistoryInterfaceMax = {};

            CleanupState();
        }

        void SetD(double d)
        {
            m_D = d;
            m_Bounds = {-m_L + m_D, m_L + m_D};
        }

        virtual double GetMaxArea() const override
        {
            return 2 * utils::power(m_L, 2);
        }

        virtual double P(const double u, const int i = -1) override
        {
            AL_PROFILE_FUNC("PreisachModelBase::P");
            if (i == m_PrevIndex)
            {
                // return m_HistoryOutput.back();
                return m_PreviousOutput;
            }
            m_PrevIndex = i;

            // The first input is x0
            if (m_HistoryU.empty())
            {
                // m_InterfaceMax = append(m_InterfaceMax, {u});
                m_InterfaceMax.push_back(u);
                // m_HistoryInterfaceMax = append(m_HistoryInterfaceMax, {u});
                // m_HistoryInterfaceMin = append(m_HistoryInterfaceMin, {consts::nan});
                m_HistoryInterfaceMax.push_back(u);
                m_HistoryInterfaceMin.push_back(consts::nan);
                m_PreviousInput = u;
                m_FirstElemType = ElementType::Max;
                m_LastElemType = ElementType::Max;
            }

            // m_HistoryU = append(m_HistoryU, {u});
            m_HistoryU.push_back(u);

            if (u < m_Bounds[0] || utils::essentiallyEqual(u, m_Bounds[0]))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::LowerBound");
                KeepMinOnly(m_Bounds[0]);

                m_PreviousInput = m_Bounds[0];
                m_PrevElemType = m_LastElemType;
                // m_HistoryOutput = append(m_HistoryOutput, {m_Bounds[0]});

                if (m_KeepDerivative)
                {
                    // m_HistoryDerivative = append(m_HistoryDerivative, {DerivativeOperator(i)});
                    m_HistoryDerivative.push_back(DerivativeOperator(i));
                }

                const double p = -GetMaxArea();
                m_HistoryOutput.push_back(p);
                m_PreviousOutput = p;
                return p;
            }
            if (u > m_Bounds[1] || utils::essentiallyEqual(u, m_Bounds[1]))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::UpperBound");
                KeepMaxOnly(m_Bounds[1]);

                m_PreviousInput = m_Bounds[1];
                m_PrevElemType = m_LastElemType;
                // m_HistoryOutput = append(m_HistoryOutput, {m_Bounds[1]});

                if (m_KeepDerivative)
                {
                    // m_HistoryDerivative = append(m_HistoryDerivative, {DerivativeOperator(i)});
                    m_HistoryDerivative.push_back(DerivativeOperator(i));
                }

                const double p = GetMaxArea();
                m_HistoryOutput.push_back(p);
                m_PreviousOutput = p;
                return p;
            }

            // ----------- Simplest cleanup -----------
            // if (!m_InterfaceMax.isempty() && m_FirstElemType == ElementType::Max && u > m_InterfaceMax[0])
            if (!m_InterfaceMax.empty() && m_FirstElemType == ElementType::Max && u > m_InterfaceMax[0])
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::MaxOnly");
                KeepMaxOnly(u);
            }
            // else if (!m_InterfaceMin.isempty() && m_FirstElemType == ElementType::Min && u < m_InterfaceMin[0])
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

                // m_InterfaceMax = append(m_InterfaceMax, {u});
                // m_HistoryInterfaceMax = append(m_HistoryInterfaceMax, {u});
                // m_HistoryInterfaceMin = append(m_HistoryInterfaceMin, {consts::nan});
                m_HistoryInterfaceMax.push_back(u);
                m_HistoryInterfaceMin.push_back(consts::nan);
                m_LastElemType = ElementType::Max;
            }
            else if (u < m_PreviousInput)
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::Min");
                CleanupInterfacesByMin(u);

                // m_InterfaceMin = append(m_InterfaceMin, {u});
                // m_HistoryInterfaceMin = append(m_HistoryInterfaceMin, {u});
                // m_HistoryInterfaceMax = append(m_HistoryInterfaceMax, {consts::nan});
                m_HistoryInterfaceMin.push_back(u);
                m_HistoryInterfaceMax.push_back(consts::nan);
                m_LastElemType = ElementType::Min;
            }
            // -------------------------------------

            double p = P_Impl(u);
            m_PreviousOutput = p;

            m_PreviousInput = u;
            m_PrevElemType = m_LastElemType;
            // m_HistoryOutput = append(m_HistoryOutput, {p});
            m_HistoryOutput.push_back(p);

            if (m_KeepDerivative)
            {
                // m_HistoryDerivative = append(m_HistoryDerivative, {DerivativeOperator(i)});
                m_HistoryDerivative.push_back(DerivativeOperator(i));
            }

            return p;
        }

        virtual double DerivativeOperator(double t, double dt) override
        {
            auto i = static_cast<uint32_t>(t / dt);

            if (i == m_PrevIndex)
            {
                return m_HistoryDerivative.back();
            }
            // m_PrevIndex = i;

            if (m_KeepDerivative)
            {
                return m_HistoryDerivative[i];
            }

            // return consts::nan;
            return DerivativeOperator(i);
        }

        double DerivativeOperator(uint32_t i)
        {
            return DerivativeOperator_Impl(i);
        }

        void CleanupState()
        {
            m_InterfaceMax = {};
            m_InterfaceMin = {};
        }

    protected:
        virtual double P_Impl(double u) = 0;

        virtual void ResetImpl() override
        {
            m_PreviousInput = consts::nan;
            m_PreviousOutput = consts::nan;
            m_InterfaceMax = {};
            m_InterfaceMin = {};
            m_PrevIndex = INT_MIN;
            m_HistoryInterfaceMin = {};
            m_HistoryInterfaceMax = {};
            m_FirstElemType = ElementType::Max;
            m_PrevElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }

        // TODO: need bounds?
        virtual double DerivativeOperator_Impl(uint32_t i)
        {
            double x_i = m_HistoryU[i];

            if (x_i < -m_L || x_i > m_L)
                return 0.0;

            // if (m_InterfaceMax.isempty())
            if (m_InterfaceMax.empty())
                return x_i - m_L;
            // if (m_InterfaceMin.isempty())
            if (m_InterfaceMin.empty())
                return x_i + m_L;

            double u = isnan(m_HistoryInterfaceMax[i]) ? m_InterfaceMax.back() : m_InterfaceMin.back();
            return x_i - u;
        }

        void CleanupInterfacesByMax(const double u)
        {
            AL_PROFILE_FUNC("PreisachModelBase::CleanupInterfacesByMax");
            switch (m_FirstElemType)
            {
            case ElementType::Max:
            {
                // const auto pos = std::distance(m_InterfaceMax.begin(), std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
                // m_InterfaceMax = insert(m_InterfaceMax, pos, u);
                // m_InterfaceMax = m_InterfaceMax[Slice(0, pos + 1)];
                // m_InterfaceMin = m_InterfaceMin[Slice(0, pos)];

                const auto pos = std::distance(m_InterfaceMax.begin(),
                                               std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
                m_InterfaceMax.insert(m_InterfaceMax.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos + 1, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos, m_InterfaceMin.end());

                break;
            }
            case ElementType::Min:
            {
                // const auto pos = std::distance(m_InterfaceMax.begin(), std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
                // m_InterfaceMax = insert(m_InterfaceMax, pos, u);
                // m_InterfaceMax = m_InterfaceMax[Slice(0, pos + 1)];
                // m_InterfaceMin = m_InterfaceMin[Slice(0, pos + 1)];

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
                // const auto pos = std::distance(m_InterfaceMin.begin(), std::ranges::lower_bound(m_InterfaceMin, u));
                // m_InterfaceMin = insert(m_InterfaceMin, pos, u);
                // m_InterfaceMax = m_InterfaceMax[Slice(0, pos + 1)];
                // m_InterfaceMin = m_InterfaceMin[Slice(0, pos + 1)];

                const auto pos = std::distance(m_InterfaceMin.begin(), std::ranges::lower_bound(m_InterfaceMin, u));
                m_InterfaceMin.insert(m_InterfaceMin.begin() + pos, u);
                m_InterfaceMax.erase(m_InterfaceMax.begin() + pos + 1, m_InterfaceMax.end());
                m_InterfaceMin.erase(m_InterfaceMin.begin() + pos + 1, m_InterfaceMin.end());

                break;
            }
            case ElementType::Min:
            {
                // const auto pos = std::distance(m_InterfaceMin.begin(), std::ranges::lower_bound(m_InterfaceMin, u));
                // m_InterfaceMin = insert(m_InterfaceMin, pos, u);
                // m_InterfaceMax = m_InterfaceMax[Slice(0, pos)];
                // m_InterfaceMin = m_InterfaceMin[Slice(0, pos + 1)];

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
            // m_InterfaceMin = mc::empty<double>();
            m_InterfaceMin = {};
            m_InterfaceMax = {u};

            // m_HistoryInterfaceMax = append(m_HistoryInterfaceMax, {u});
            // m_HistoryInterfaceMin = append(m_HistoryInterfaceMin, {consts::nan});
            m_HistoryInterfaceMax.push_back(u);
            m_HistoryInterfaceMin.push_back(consts::nan);
            m_FirstElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }

        void KeepMinOnly(double u)
        {
            m_InterfaceMin = {u};
            // m_InterfaceMax = mc::empty<double>();
            m_InterfaceMax = {};

            // m_InterfaceMin = append(m_InterfaceMin, {u});
            m_InterfaceMin.push_back(u);
            // m_HistoryInterfaceMin = append(m_HistoryInterfaceMin, {u});
            // m_HistoryInterfaceMax = append(m_HistoryInterfaceMax, {consts::nan});
            m_HistoryInterfaceMin.push_back(u);
            m_HistoryInterfaceMax.push_back(consts::nan);
            m_FirstElemType = ElementType::Min;
            m_LastElemType = ElementType::Min;
        }

    protected:
        double m_D = 0.0;
        std::array<double, 2> m_Bounds = {-m_L + m_D, m_L + m_D};
        double m_PreviousInput = consts::nan;
        double m_PreviousOutput = consts::nan;
        std::vector<double> m_InterfaceMax{}, m_InterfaceMin{};
        bool m_KeepDerivative = false, m_KeepAnimation = false;
        uint32 m_PrevIndex = INT_MIN;

        std::vector<double> m_HistoryInterfaceMin{}, m_HistoryInterfaceMax{};

        enum class ElementType : bool
        {
            Min = false,
            Max = true,
        };

        ElementType m_FirstElemType = ElementType::Max;
        ElementType m_PrevElemType = ElementType::Max;
        ElementType m_LastElemType = ElementType::Max;
    };

    class DoublePreisachModel : public PreisachModelBase
    {
    public:
        DoublePreisachModel(double L, double d) :
            PreisachModelBase(L), m_D({d, -d})
        {
            assert(d >= 0);

            m_HistoryU = {};
            m_HistoryOutput = {};

            // m_UpperModel = Ref<SinglePreisachModel>::Create(m_L + m_D[0], false, false);
            // m_LowerModel = Ref<SinglePreisachModel>::Create(m_L - m_D[1], false, false);
        }

        DoublePreisachModel(double L, const std::array<double, 2> &d) :
            PreisachModelBase(L), m_D({d[0], -d[1]})
        {
            assert(d[0] >= 0 && d[1] >= 0);

            m_HistoryU = {};
            m_HistoryOutput = {};

            // m_UpperModel = Ref<SinglePreisachModel>::Create(m_L + m_D[0], false, false);
            // m_LowerModel = Ref<SinglePreisachModel>::Create(m_L - m_D[1], false, false);
        }

        virtual double P(double u, int i = -1) override
        {
            if (i == m_PrevIndex)
            {
                // return m_HistoryOutput.back();
                return m_PreviousOutput;
            }
            m_PrevIndex = i;

            m_HistoryU.push_back(u);
            double p = consts::nan;
            /// TODO: если d == 1, то петли не соприкасаются из-за добавки +-d/2
            if (u >= -m_L + m_D[0])
            {
                double p1 = m_UpperModel->P(u, i);
                double p2 = m_LowerModel->P(u, i);
                p = p1 + p2 /*+ m_D[0] / 2.*/;
            }
            // m_D[1] is negative
            else if (u <= m_L + m_D[1])
            {
                double p1 = m_UpperModel->P(u, i);
                double p2 = m_LowerModel->P(u, i);
                p = p1 + p2 /*+ m_D[1] / 2.*/;
            }
            else
            {
                p = u;
            }
            m_HistoryOutput.push_back(p);
            m_PreviousOutput = p;
            return p;
        }

    protected:
        virtual void ResetImpl() override
        {
            m_UpperModel.Reset();
            m_LowerModel.Reset();
            m_PrevIndex = INT_MIN;
            m_PreviousOutput = consts::nan;
        }

    protected:
        Ref<SinglePreisachModel> m_UpperModel;
        Ref<SinglePreisachModel> m_LowerModel;

        std::array<double, 2> m_D;

        int32 m_PrevIndex = INT_MIN;
        double m_PreviousOutput = consts::nan;
    };
}

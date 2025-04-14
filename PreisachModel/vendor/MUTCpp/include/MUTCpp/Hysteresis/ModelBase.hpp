// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/isnan.hpp"
#include "Functions/append.hpp"
#include "Core/Constants.hpp"
#include "Debug/Profile.hpp"
#include "Functions/empty.hpp"
#include "Ref/Ref.hpp"

namespace mc
{
    class PreisachModelBase : public RefCounted
    {
    public:
        PreisachModelBase(double L, bool keepDerivative = false, bool keepAnimation = false) :
            m_L(L), m_KeepDerivative(keepDerivative), m_KeepAnimation(keepAnimation)
        {
            AL_PROFILE_FUNC("PreisachModelBase::Constructor");
            m_HistoryInterfaceMin = {};
            m_HistoryInterfaceMax = {};
            m_HistoryU = {};
            m_HistoryOutput = {};

            CleanupState();

            // For animation
            m_PointsX = asarray<double>({-L});
            m_PointsY = asarray<double>({-L});
            m_InterfaceX = asarray<double>({-L, -L});
            m_InterfaceY = asarray<double>({-L, -L});
        }

        PreisachModelBase(const PreisachModelBase&) = delete;
        PreisachModelBase& operator=(const PreisachModelBase&) = delete;

        virtual ~PreisachModelBase() override = default;

        double GetL() const
        {
            return m_L;
        }

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

        // std::array<Matrix<double>, 4> GetAnimationData() const
        std::tuple<std::vector<double>, Matrix<double>, Matrix<double>, std::vector<double>> GetAnimationData() const
        {
            return {m_HistoryU, m_HistoryInterfaceX, m_HistoryInterfaceY, m_HistoryOutput};
        }

        double P(double u, int i = -1)
        {
            AL_PROFILE_FUNC("PreisachModelBase::P");
            if (i == m_PrevIndex)
            {
                // return m_HistoryOutput.back();
                return m_PreviousOutput;
            }
            m_PrevIndex = i;

            // constexpr double d = 1.;
            // if (u >= 1.)
            // {
            //     u -= d;
            // }
            // else if (u <= -1.)
            // {
            //     u += d;
            // }
            // else if (u > -1. || u < 1.)
            // {
            //     m_PreviousOutput = u;
            //     return u;
            // }
            
            // The first input is x0
            // if (m_HistoryU.isempty())
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

            if (u < -m_L || utils::essentiallyEqual(u, -m_L))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::LowerBound");
                KeepMinOnly(-m_L);

                m_PreviousInput = -m_L;
                m_PrevElemType = m_LastElemType;
                // m_HistoryOutput = append(m_HistoryOutput, {-m_L});
                m_HistoryOutput.push_back(-m_L);

                if (m_KeepDerivative)
                {
                    // m_HistoryDerivative = append(m_HistoryDerivative, {DerivativeOperator(i)});
                    m_HistoryDerivative.push_back(DerivativeOperator(i));
                }

                m_PreviousOutput = -m_L;
                return -m_L;
            }
            if (u > m_L || utils::essentiallyEqual(u, m_L))
            {
                AL_PROFILE_SCOPE("PreisachModelBase::P::UpperBound");
                KeepMaxOnly(m_L);
                
                m_PreviousInput = m_L;
                m_PrevElemType = m_LastElemType;
                // m_HistoryOutput = append(m_HistoryOutput, {m_L});
                m_HistoryOutput.push_back(m_L);

                if (m_KeepDerivative)
                {
                    // m_HistoryDerivative = append(m_HistoryDerivative, {DerivativeOperator(i)});
                    m_HistoryDerivative.push_back(DerivativeOperator(i));
                }

                m_PreviousOutput = m_L;
                return m_L;
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

            if (m_KeepAnimation)
            {
                UpdateAnimationData(u);
            }

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

        double DerivativeOperator(double t, double dt)
        {
            auto i = static_cast<uint32_t>(t / dt);

            if (i == m_PrevIndex)
            {
                return m_HistoryDerivative.back();
            }
            m_PrevIndex = i;
            
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

        void UpdateAnimationData(double u)
        {
            AL_PROFILE_FUNC("PreisachModelBase::UpdateAnimationData");
            m_PointsX = m_InterfaceX[Slice(0, -1)];
            m_PointsY = m_InterfaceY[Slice(0, -1)];

            if (m_LastElemType != m_PrevElemType)
            {
                m_PointsX = append(m_PointsX, {m_HistoryU.back()});
                m_PointsY = append(m_PointsY, {m_HistoryU.back()});
            }

            if (m_LastElemType == ElementType::Max)
            {
                m_PointsY.putMask(m_PointsY <= u, u);
                m_PointsY[-1] = u;
            }
            else
            {
                m_PointsX.putMask(m_PointsX >= u, u);
                m_PointsX[-1] = u;
            }

            m_InterfaceX = append(m_PointsX, {u});
            m_InterfaceY = append(m_PointsY, {u});

            auto [iX, iY] = RemoveRedundantPoints(m_InterfaceX, m_InterfaceY);
            m_InterfaceX = iX;
            m_InterfaceY = iY;

            m_HistoryInterfaceX = append(m_HistoryInterfaceX, m_InterfaceX);
            m_HistoryInterfaceY = append(m_HistoryInterfaceY, m_InterfaceY);
        }

        static std::pair<Matrix<double>, Matrix<double>> RemoveRedundantPoints(
            const Matrix<double> &pointsX, const Matrix<double> &pointsY)
        {
            AL_PROFILE_FUNC("PreisachModelBase::RemoveRedundantPoints");
            auto removeInBetween = [](const Matrix<double> &arr) -> std::pair<Matrix<double>, Matrix<bool>>
            {
                auto whipeoutIndexes = empty<bool>(arr.shape());
                if (arr.size() < 3)
                {
                    whipeoutIndexes.fill(true);
                    return {arr, whipeoutIndexes};
                }

                whipeoutIndexes[0] = true;
                whipeoutIndexes[-1] = true;
                for (auto i : range(1, arr.size() - 1))
                {
                    whipeoutIndexes[i] = !all_equals(arr[i], arr[i - 1], arr[i + 1]);
                }
                return {arr[whipeoutIndexes], whipeoutIndexes};
            };

            auto [pointsX1, whipeoutIndexes1] = removeInBetween(pointsX);
            auto pointsY1 = pointsY[whipeoutIndexes1];

            auto [pointsY2, whipeoutIndexes2] = removeInBetween(pointsY1);
            auto pointsX2 = pointsX1[whipeoutIndexes2];

            return {pointsX2, pointsY2};
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

                const auto pos = std::distance(m_InterfaceMax.begin(), std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
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

                const auto pos = std::distance(m_InterfaceMax.begin(), std::ranges::lower_bound(m_InterfaceMax, u, std::greater<double>()));
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
        double m_L;
        double m_PreviousInput = consts::nan;
        double m_PreviousOutput = consts::nan;
        std::vector<double> m_InterfaceMax{}, m_InterfaceMin{};
        bool m_KeepDerivative = false, m_KeepAnimation = false;
        uint32 m_PrevIndex = -1;
        
        // Matrix<double> m_HistoryInterfaceMin{}, m_HistoryInterfaceMax{}, m_HistoryU{}, m_HistoryOutput{},
        //                m_HistoryDerivative{};
        std::vector<double> m_HistoryInterfaceMin{}, m_HistoryInterfaceMax{}, m_HistoryU{}, m_HistoryOutput{},
                       m_HistoryDerivative{};

        // ---- For Animation ----
        Matrix<double> m_InterfaceX{}, m_InterfaceY{}, m_PointsX{}, m_PointsY{};
        Matrix<double> m_HistoryInterfaceX{}, m_HistoryInterfaceY{};
        // -----------------------

        enum class ElementType : bool
        {
            Min = false,
            Max = true,
        };

        ElementType m_FirstElemType = ElementType::Max;
        ElementType m_PrevElemType = ElementType::Max;
        ElementType m_LastElemType = ElementType::Max;
    };
}

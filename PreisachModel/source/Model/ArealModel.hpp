#pragma once
#include <vector>
#include "ModelBase.hpp"
#include "NumCpp.hpp"
#include "Utils/Math.hpp"

#define NAN nc::constants::nan

class ArealPreisachModel : public PreisachModelBase
{
public:
    ArealPreisachModel(const double L) : m_L(L)
    {
        m_InterfaceMin = {};
        m_InterfaceMax = {};
    }

    double P(double u) override
    {
        m_CountToRemoveMins = 0;
        m_CountToRemoveMaxs = 0;
        
        m_HistoryU.emplace_back(u);

        if (m_HistoryU.size() == 394)
        {
            std::cout << '\n';
        }
        
        // ----------- Boundary cases -----------
        if (u <= -m_L)
        {
            m_PreviousArea = 0.0;
            m_HistoryOutput.emplace_back(0.0);
            // return - m_PreviousArea;
            return 0.0;
        }
        if (u >= m_L)
        {
            m_PreviousArea = GetMaxArea();
            m_HistoryOutput.emplace_back(GetMaxArea());
            // return GetMaxArea() - m_PreviousArea;`
            return GetMaxArea();
        }
        // --------------------------------------

        // The first input is x0
        if (m_HistoryU.empty() || eqf(m_PreviousArea, 0.0))
        {
            m_InterfaceMax.clear();
            m_InterfaceMin.clear();

            m_InterfaceMax.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(NAN);
            m_PreviousInput = u;
            m_FirstElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }
        else if (eqf(m_PreviousArea, GetMaxArea()))
        {
            m_InterfaceMax.clear();
            m_InterfaceMin.clear();

            m_InterfaceMin.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(NAN);
            m_PreviousInput = u;
            m_FirstElemType = ElementType::Min;
            m_LastElemType = ElementType::Min;
        }
        
        // ----------- Simplest cleanup -----------
        else if (!m_InterfaceMax.empty() && m_FirstElemType == ElementType::Max && u >= m_InterfaceMax.front())
        {
            m_InterfaceMin.clear();
            m_InterfaceMax.clear();

            m_InterfaceMax.emplace_back(u);
            m_FirstElemType = ElementType::Max;
            m_LastElemType = ElementType::Max;
        }
        else if (!m_InterfaceMin.empty() && m_FirstElemType == ElementType::Min && u <= m_InterfaceMin.front())
        {
            m_InterfaceMin.clear();
            m_InterfaceMax.clear();

            m_InterfaceMin.emplace_back(u);
            m_FirstElemType = ElementType::Min;
            m_LastElemType = ElementType::Min;
        }
        // ----------------------------------------
        
        // ----------- Check extrema -----------
        else if (u > m_PreviousInput)
        {
            GetCountsByMax();
            
            m_InterfaceMax.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(NAN);
            m_LastElemType = ElementType::Max;
        }
        else if (u < m_PreviousInput)
        {
            GetCountsByMin();
            
            m_InterfaceMin.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(NAN);
            m_LastElemType = ElementType::Min;
        }
        // -------------------------------------

        // std::cout << m_CountToRemoveMaxs << " " << m_CountToRemoveMins << '\n';

        const double p = CalculateOutput();
        m_HistoryOutput.emplace_back(p);

        m_InterfaceMax.erase(m_InterfaceMax.end() - m_CountToRemoveMaxs, m_InterfaceMax.end());
        m_InterfaceMin.erase(m_InterfaceMin.end() - m_CountToRemoveMins, m_InterfaceMin.end());
        
        m_PreviousInput = u;
        m_PrevElemType = m_LastElemType;

        // std::cout << "MAX size: " << m_InterfaceMax.size() << "   MIN size: " << m_InterfaceMin.size() << '\n';

        return p;
    }

private:
    void GetCountsByMax()
    {
        const double u = m_HistoryU.back();
        const auto countMaxs = static_cast<uint32_t>(std::ranges::count_if(m_InterfaceMax, [u](const double elem)
            {
                return elem < u;
            }));

        m_CountToRemoveMaxs = m_InterfaceMax.size() >= countMaxs ? countMaxs : 0;

        const auto countMins = countMaxs > 0
                                  ? (m_PrevElemType == ElementType::Max
                                         ? countMaxs - 1
                                         : countMaxs)
                                  : 0;

        m_CountToRemoveMins = m_InterfaceMin.size() >= countMins ? countMins : 0;
    }

    void GetCountsByMin()
    {
        const double u = m_HistoryU.back();
        const auto countMins = static_cast<uint32_t>(std::ranges::count_if(m_InterfaceMin, [u](const double elem)
            {
                return elem > u;
            }));

        m_CountToRemoveMins = m_InterfaceMin.size() >= countMins ? countMins : 0;

        const auto countMaxs = countMins > 0
                                  ? (m_PrevElemType == ElementType::Max
                                         ? countMins
                                         : countMins - 1)
                                  : 0;
        
        m_CountToRemoveMaxs = m_InterfaceMax.size() >= countMaxs ? countMaxs : 0;
    }
    
    double CalculateOutput()
    {
        const double u = m_HistoryU.back();

        // ----------- Simplest cases -----------
        double area = 0.0;
        if (m_InterfaceMin.empty())
        {
            area = std::pow(m_L + m_InterfaceMax.back(), 2) / 2;
            const double p = area - m_PreviousArea;
            m_PreviousArea = area;
            return p;
        }
        if (m_InterfaceMax.empty())
        {
            const auto x = m_InterfaceMin.back();
            area = (3 * m_L - x) / 2 * (m_L + x);
            const double p = area - m_PreviousArea;
            m_PreviousArea = area;
            return p;
        }
        // --------------------------------------

        // const double x = m_PreviousInput; // X_{k}
        // const double y = m_HistoryU[m_HistoryU.size() - 3]; // X_{k - 1}

        // if ((x - y) * (u - x) > 0) // last & next elems - both maxs or mins
        // {
        //     if (m_LastElemType == ElementType::Max)
        //     {
        //         const auto n = m_InterfaceMin.size();
        //         for (auto i = n - m_CountToRemoveMins + 1; i < n; i++)
        //         {
        //             area += (m_InterfaceMin[i] - m_InterfaceMin[i - 1]) * (u - m_InterfaceMax[j_i(i)]);
        //         }
        //         area += (x - m_InterfaceMin.back()) * (u - x);
        //         area += std::pow(u - x, 2) / 2;
        //     }
        //     else if (m_LastElemType == ElementType::Min)
        //     {
        //         const auto n = m_InterfaceMax.size();
        //         for (auto i = n - m_CountToRemoveMaxs + 1; i < n; i++)
        //         {
        //             area -= (m_InterfaceMax[i - 1] - m_InterfaceMax[i]) * (m_InterfaceMin[r_i(i)] - u);
        //         }
        //         area -= (m_InterfaceMax.back() - x) * (x - u);
        //         area -= std::pow(u - x, 2) / 2;
        //     }
        // }
        // else if ((x - y) * (u - x) < 0)
        // {
        //     if (m_LastElemType == ElementType::Max)
        //     {
        //         const auto n = m_InterfaceMin.size();
        //         for (auto i = n - m_CountToRemoveMins + 1; i < n; i++)
        //         {
        //             area += (m_InterfaceMin[i] - m_InterfaceMin[i - 1]) * (u - m_InterfaceMax[j_i(i)]);
        //             area += std::pow(u - x, 2) / 2;
        //         }
        //     }
        //     else if (m_LastElemType == ElementType::Min)
        //     {
        //         const auto n = m_InterfaceMax.size();
        //         for (auto i = n - m_CountToRemoveMaxs + 1; i < n; i++)
        //         {
        //             area -= (m_InterfaceMax[i - 1] - m_InterfaceMax[i]) * (m_InterfaceMin[r_i(i)] - u);
        //         }
        //         area -= std::pow(u - x, 2) / 2;
        //     }
        // }

        if (m_LastElemType == ElementType::Max)
        {
            // const auto n = m_InterfaceMin.size();
            // for (auto i = n - m_CountToRemoveMins + 1; i < n; i++)
            // {
            //     area += (m_InterfaceMin[i] - m_InterfaceMin[i - 1]) * (u - m_InterfaceMax[j_i(i)]);
            // }
        
            if (m_PrevElemType == m_LastElemType)
            {
                // area += (m_PreviousInput - m_InterfaceMin.back()) * (u - m_PreviousInput);
                area += (u - m_PreviousInput) / 2 * (u + m_PreviousInput - 2 * m_InterfaceMin.back());
            }
            else
            {
                area += std::pow(u - m_PreviousInput, 2) / 2;
            }
        }
        else if (m_LastElemType == ElementType::Min)
        {
            // const auto n = m_InterfaceMax.size();
            // for (auto i = n - m_CountToRemoveMaxs + 1; i < n; i++)
            // {
            //     area -= (m_InterfaceMax[i - 1] - m_InterfaceMax[i]) * (m_InterfaceMin[r_i(i)] - u);
            // }
        
            if (m_PrevElemType == m_LastElemType)
            {
                // area -= (m_InterfaceMax.back() - m_PreviousInput) * (m_PreviousInput - u);
                area -= (m_PreviousInput - u) / 2 * (2 * m_InterfaceMax.back() - u - m_PreviousInput);
            }
            else
            {
                area -= std::pow(m_PreviousInput - u, 2) / 2;
            }
        }

        m_PreviousArea += area;
        return area;
    }

    [[nodiscard]] double GetMaxArea() const
    {
        return 2 * std::pow(m_L, 2);
    }

    [[nodiscard]] size_t j_i(const size_t i) const { return m_FirstElemType == ElementType::Max ? i : i - 1; }
    [[nodiscard]] size_t r_i(const size_t i) const { return m_FirstElemType == ElementType::Min ? i : i - 1; }

private:
    double m_L;
    double m_PreviousInput = NAN;
    std::vector<double> m_InterfaceMin, m_InterfaceMax;
    double m_PreviousArea = 0.0;

    uint32_t m_CountToRemoveMins = 0, m_CountToRemoveMaxs = 0;

    enum class ElementType : bool
    {
        Min = false,
        Max = true,
    };

    ElementType m_FirstElemType = ElementType::Max;
    ElementType m_PrevElemType = ElementType::Max;
    ElementType m_LastElemType = ElementType::Max;
};

#pragma once
#include <vector>
#include <algorithm>
#include "ModelBase.hpp"
#include "NumCpp.hpp"
#include "Utils/Math.hpp"
#include "Utils/Array.hpp"

#define NAN nc::constants::nan

enum class InputState : bool
{
    Ascending = false,
    Descending = true,
};

enum class ElementType : bool
{
    Min = false,
    Max = true,
};

class ArealPreisachModel_V1 : public PreisachModelBase
{
public:
    ArealPreisachModel_V1(const double L) : m_L(L)
    {
        m_InterfaceMin = {};
        m_InterfaceMax = {};
    }

    // std::vector<double> ApplyInputs(const std::vector<double>& uu)
    // {
    //     std::vector<double> res(uu.size());
    //     for (const double u : uu)
    //     {
    //         res.emplace_back(P(u));
    //     }
    //     return res;
    // }

    double P(double u) override
    {
        // The first input is x0
        if (m_HistoryU.empty() || eqf(m_PreviousArea, 0.0) || eqf(m_PreviousArea, GetMaxArea()))
        {
            m_InterfaceMax.clear();
            m_InterfaceMin.clear();
            
            m_InterfaceMax.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(NAN);
            m_PreviousInput = u;
            m_FirstElemType = ElementType::Max;
            m_PrevElemType = ElementType::Max;
        }

        m_HistoryU.emplace_back(u);
        
        if (u > m_PreviousInput)
        {
            CleanupRedundantMaxs();
            
            m_InterfaceMax.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(NAN);
            m_LastElemType = ElementType::Max;

            if (m_PreviousArea == 0.0 || m_InterfaceMin.empty())
            {
                m_FirstElemType = ElementType::Max;
            }
        }
        else if (u < m_PreviousInput)
        {
            CleanupRedundantMins();

            m_InterfaceMin.emplace_back(u);
            m_HistoryInterfaceMin.emplace_back(u);
            m_HistoryInterfaceMax.emplace_back(NAN);
            m_LastElemType = ElementType::Min;

            if (m_PreviousArea == 0.0 || m_InterfaceMax.empty())
            {
                m_FirstElemType = ElementType::Min;
            }
        }

        if (!m_InterfaceMax.empty() && m_FirstElemType == ElementType::Max && u >= m_InterfaceMax[0])
        {
            m_InterfaceMin.clear();
            m_InterfaceMax.clear();
            
            m_InterfaceMax.emplace_back(u);
            m_FirstElemType = ElementType::Max;
        }

        const double p = CalculateOutput();
        m_HistoryOutput.emplace_back(p);

        m_PreviousInput = u;
        m_PrevElemType = m_LastElemType;
        
        return p;
    }

private:
    void CleanupRedundantMins()
    {
        double u = m_HistoryU.back();
        const size_t beforeSize = m_InterfaceMin.size();
        std::erase_if(m_InterfaceMin, [u](const double elem)
            {
                return elem >= u;
            });
        const size_t afterSize = m_InterfaceMin.size();
        
        if (beforeSize != afterSize && m_InterfaceMax.size() > 1)
        {
            if (std::ranges::any_of(m_HistoryInterfaceMin, [](const double elem) { return !nc::isnan(elem); }))
            {
                auto [leftBound, rightBound] = FindBoundsIndices(m_HistoryInterfaceMin, u);
                if (rightBound == -1) return;

                if (leftBound == -1)
                {
                    switch (m_FirstElemType)
                    {
                        case ElementType::Max:
                        {
                            auto x = m_InterfaceMax.front();
                            m_InterfaceMax.clear();
                            m_InterfaceMin.clear();
                            m_InterfaceMax.emplace_back(x);
                            break;
                        }
                        case ElementType::Min:
                        {
                            m_InterfaceMax.clear();
                            m_InterfaceMin.clear();
                            break;
                        }
                    }
                }
                else
                {
                    if (std::abs(leftBound - rightBound) > 1)
                    {
                        for (auto i = rightBound - 1; i > leftBound; i--)
                        {
                            if (!nc::isnan(m_HistoryInterfaceMax[i]))
                            {
                                double max = m_HistoryInterfaceMax[i];
                                std::erase_if(m_InterfaceMax, [max](const double elem)
                                {
                                    return elem < max;
                                });
                                break;
                            }
                        }
                    }
                    else
                    {
                        bool isNotNan = false;
                        for (auto i = rightBound; i < static_cast<int>(m_HistoryInterfaceMax.size()); i++)
                        {
                            if (!nc::isnan(m_HistoryInterfaceMax[i]))
                            {
                                isNotNan = true;
                            }
                            else if (nc::isnan(m_HistoryInterfaceMax[i]) && isNotNan)
                            {
                                double max = m_HistoryInterfaceMax[i - 1];
                                std::erase_if(m_InterfaceMax, [max](const double elem)
                                {
                                    return elem < max;
                                });
                                break;
                            }
                        } 
                    }
                }
            }
        }
    }

    void CleanupRedundantMaxs()
    {
        double u = m_HistoryU.back();
        const size_t beforeSize = m_InterfaceMax.size();
        std::erase_if(m_InterfaceMax, [u](const double elem)
        {
            return elem <= u;
        });
        const size_t afterSize = m_InterfaceMax.size();
        
        if (beforeSize != afterSize && m_InterfaceMin.size() > 1)
        {
            if (std::ranges::any_of(m_HistoryInterfaceMax, [](const double elem) { return !nc::isnan(elem); }))
            {
                
                
                // auto idx = FindLastGreater(m_HistoryInterfaceMax, u);
                // if (idx.has_value())
                // {
                //     double min = 1000.0;
                //     for (auto i = idx.value(); i > 0; i--)
                //     {
                //         if (!nc::isnan(m_HistoryInterfaceMin[i]))
                //         {
                //             min = m_HistoryInterfaceMin[i];
                //             break;
                //         }
                //     }
                //
                //     std::erase_if(m_InterfaceMin, [min](const double elem)
                //     {
                //        return elem > min;
                //     });
                // }
            }
        }
    }
    
    double CalculateOutput()
    {
        const double u = m_HistoryU.back();
        if (u <= -m_L)
        {
            m_PreviousArea = 0.0;
            return 0.0;
        }
        if (u >= m_L)
        {
            m_PreviousArea = GetMaxArea();
            return GetMaxArea();
        }
        
        double area = 0.0;
        if (m_InterfaceMin.empty())
        {
            area = std::pow(m_L + m_InterfaceMax.back(), 2) / 2;
            const double p = std::abs(area) - std::abs(m_PreviousArea);
            m_PreviousArea = area;
            if (p > 2 * std::pow(m_L, 2))
            {
                std::cout << "max size = " << m_HistoryInterfaceMin.size() << "\n";
            }
            return p;
        }
        if (m_InterfaceMax.empty())
        {
            const auto x = m_InterfaceMin.back();
            area = (3 * m_L - x) / 2 * (m_L + x);
            const double p = std::abs(area) - std::abs(m_PreviousArea);
            m_PreviousArea = area;
            if (p > 2 * std::pow(m_L, 2))
            {
                std::cout << "min size = " << m_HistoryInterfaceMin.size() << "\n";
            }
            return p;
        }

        if (m_FirstElemType == ElementType::Max && m_InterfaceMax.size() == 1 && m_InterfaceMin.size() == 1)
        {
            area = (m_L + 2 * m_InterfaceMax.back() - m_InterfaceMin.back()) / 2 * (m_L + m_InterfaceMin.back());
            const double p = std::abs(area) - std::abs(m_PreviousArea);
            if (p > 2 * std::pow(m_L, 2))
            {
                std::cout << "1max size = " << m_HistoryInterfaceMin.size() << "\n";
            }
            m_PreviousArea = area;
            return p;
        }
        if (m_FirstElemType == ElementType::Min && m_InterfaceMax.size() == 1 && m_InterfaceMin.size() == 1)
        {
            const double s1 = (3 * m_L - m_InterfaceMin.back()) / 2 * (m_L + m_InterfaceMin.back());
            const double s2 = std::pow(m_L + m_InterfaceMax.back(), 2) / 2;
            area = s1 + s2;
            const double p = std::abs(area) - std::abs(m_PreviousArea);
            if (p > 2 * std::pow(m_L, 2))
            {
                std::cout << "1min size = " << m_HistoryInterfaceMin.size() << "\n";
            }
            m_PreviousArea = area;
            return p;
        }

        if (m_PrevElemType == ElementType::Min)
        {
            if (u > m_PreviousInput)
            {
                area = std::pow(u - m_PreviousInput, 2) / 2;
            }
            if (u < m_PreviousInput)
            {
                const double lastMax = m_InterfaceMax.back();
                area = -(2 * lastMax - u - m_PreviousInput) / 2 * (m_PreviousInput - u);
            }
        }
        else if (m_PrevElemType == ElementType::Max)
        {
            if (u > m_PreviousInput)
            {
                const double lastMin = m_InterfaceMin.back();
                area = (m_PreviousInput + u - 2 * lastMin) / 2 * (u - m_PreviousInput);
            }
            if (u < m_PreviousInput)
            {
                area = -std::pow(m_PreviousInput - u, 2) / 2;
            }
        }
        
        // std::cout << "max size = " << m_InterfaceMax.size() << " ";
        // std::cout << "min size = " << m_InterfaceMin.size() << "\n";

        m_PreviousArea += area;
        return area;
        // return 0.0;
    }

    [[nodiscard]] double GetMaxArea() const
    {
        return 2 * std::pow(m_L, 2);
    }
    
private:
    double m_L;
    ElementType m_FirstElemType = ElementType::Max;
    ElementType m_PrevElemType = ElementType::Max;
    ElementType m_LastElemType = ElementType::Max;
    double m_PreviousInput = NAN;
    std::vector<double> m_InterfaceMin, m_InterfaceMax;
    double m_PreviousArea = 0.0;
};

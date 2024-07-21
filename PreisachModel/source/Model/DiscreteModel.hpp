#pragma once
#include "ModelBase.hpp"
#include <cstdint>
#include "NumCpp.hpp"
#include "Utils/Math.hpp"

class DiscretePreisachModel : public PreisachModelBase
{
public:
    DiscretePreisachModel(const double L, const double h) : m_L(L), m_H(h), m_Count(static_cast<uint32_t>(2 * L / h))
    {
        Init();
    }

    double P(const double u) override
    {
        m_HistoryU.emplace_back(u);
        
        for (int i = 0; i < m_Count; i++)
        {
            for (int j = 0; j < m_Count; j++)
            {
                if (i <= j)
                {
                    if (eqf(m_R(i, j), m_Down) && u >= -m_L + m_H * j)
                    {
                        m_R(i, j) = m_Up;
                    }

                    if (eqf(m_R(i, j), m_Up) && u <= -m_L + m_H * i)
                    {
                        m_R(i, j) = m_Down;
                    }
                }
            }
        }

        double p = 0.0;
        for (int i = 0; i < m_Count; i++)
        {
            for (int j = 0; j < m_Count; j++)
            {
                if (i <= j)
                {
                    p += m_R(i, j);
                }
            }
        }

        p /= m_Count * (m_Count + 1.0) / 2.0;

        m_HistoryOutput.emplace_back(p);

        const auto [c1, c2] = GetExtremaNums();
        std::cout << "MAX size: " << c1 << "   MIN size: " << c2 << '\n';

        return p;
    }

private:
    void Init()
    {
        m_R = nc::zeros<double>(m_Count);
        for (int i = 0; i < m_Count; i++)
        {
            for (int j = 0; j < m_Count; j++)
            {
                if (i <= j)
                {
                    m_R(i, j) = m_Down;
                }
                else
                {
                    m_R(i, j) = nc::constants::nan;
                }
            }
        }
    }

    std::pair<double, double> GetExtremaNums()
    {
        uint32_t countMaxs = 0;
        uint32_t countMins = 0;
        for (int i = 0; i < m_Count; i++)
        {
            for (int j = 0; j < m_Count; j++)
            {
                if (i <= j)
                {
                    if (eqf(m_R(i, j), m_Up))
                    {
                        countMaxs++;
                    }
                    else if (eqf(m_R(i, j), m_Down))
                    {
                        countMins++;
                    }
                }
            }
        }

        return {countMaxs / (m_Count * (m_Count + 1.0) / 2.0), countMins / (m_Count * (m_Count + 1.0) / 2.0)};
    }

private:
    nc::NdArray<double> m_R;
    double m_L, m_H;
    uint32_t m_Count;
    double m_Up = 1.0, m_Down = -1.0;
};

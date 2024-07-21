#pragma once

class PreisachModelBase
{
public:
    PreisachModelBase()
    {
        m_HistoryInterfaceMin = {};
        m_HistoryInterfaceMax = {};

        m_HistoryU = {};
        m_HistoryOutput = {};
    }
    
    virtual ~PreisachModelBase() = default;

    std::pair<std::vector<double>, std::vector<double>> HysteresisLoop()
    {
        return {m_HistoryU, m_HistoryOutput};
    }
    
    virtual double P(double u) = 0;

protected:
    std::vector<double> m_HistoryInterfaceMin, m_HistoryInterfaceMax, m_HistoryU, m_HistoryOutput;
};
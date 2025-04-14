#pragma once

#include "MUTCpp.hpp"

inline std::vector<std::pair<double, int>> mLCE_DivergenceDegree(mc::ode::DynamicalSystem *system, double e,
                                                                 mc::Matrix<double> &traj1, mc::Matrix<double> &traj2,
                                                                 double T = 8.0, int M = 30, bool useLog = true,
                                                                 uint8_t n = 7, bool linearize = true,
                                                                 bool useMult = false)
{
    double mLCE = 0.0;
    mc::Matrix<double> history = mc::zeros<double>(mc::Shape(1, M));

    traj1 = {};
    traj2 = {};

    for (int i = 1; i < M + 1; ++i)
    {
        // auto w = mc::random::rand<double>(static_cast<mc::Shape>(system->GetDimension()));
        // auto w = mc::random::rand<double>(linearize ? mc::Shape(3, 1) : mc::Shape(1, 3));
        // TODO: row or col???
        // auto w = mc::random::rand<double>(mc::Shape(1, 3));
        auto w = mc::zeros<double>(mc::Shape(1, 3));
        w[0] = 0.23;
        w[1] = 0.2;
        w[2] = 0.0;
        mc::print(w);
        // w /= mc::norm(w)[0];
        // mc::print(w);
        // w *= e;
        // mc::print(w);
        traj2.append(w(0, w.cSlice()), mc::Axis::ROW);

        for (int j = 0; j < static_cast<int>(T / system->GetDeltaTime()); ++j)
        {
            w = linearize ? system->NextLTM(w) : system->NextTM(w);
            traj2.append(w(0, w.cSlice()), mc::Axis::ROW);
            auto a = system->Forward(1);
            auto b = mc::split(a, mc::Matrix{system->GetDimension()});
            for (const auto &elem : b)
            {
                traj1.append(elem, mc::Axis::ROW);
            }
        }

        if (useMult)
        {
            mLCE *= useLog ? mc::log(mc::norm(w)[0] / e) : mc::norm(w)[0] / e;
        }
        else
        {
            mLCE += useLog ? mc::log(mc::norm(w)[0] / e) : mc::norm(w)[0] / e;
        }
        history[i - 1] = mLCE / (i * T);
        w /= mc::norm(w)[0];
        w *= e;
        traj2.append(w(0, w.cSlice()), mc::Axis::ROW);
    }

    std::vector<std::pair<double, int>> res = {};
    for (uint8_t i = 1; i < n + 1; i++)
    {
        auto a = (useMult ? std::pow(mLCE, 1 / M) : mLCE) / (M * mc::power(T, i));
        res.emplace_back(a, i);
    }

    return res;
}

inline std::pair<double, mc::Matrix<double>> mLCE_Kuznetsov(mc::ode::DynamicalSystem *system, double e, double T = 8.0,
                                                            int M = 30, bool useLog = true)
{
    double mLCE = 0.0;
    mc::Matrix<double> history = mc::zeros<double>(mc::Shape(1, M));
    for (int i = 1; i < M + 1; ++i)
    {
        auto w = mc::random::rand<double>(static_cast<mc::Shape>(system->GetDimension()));
        w = normalize(w);
        mc::print(w);
        w *= e;

        for (int j = 0; j < static_cast<int>(T / system->GetDeltaTime()); ++j)
        {
            w = system->NextLTM(w);
            system->Forward(1);
        }

        mLCE += useLog ? mc::log(mc::norm(w)[0] / e) : mc::norm(w)[0] / e;
        history[i - 1] = mLCE / (i * T);
        w = normalize(w) * e;
    }

    mLCE /= M * T;
    return {mLCE, history};
}

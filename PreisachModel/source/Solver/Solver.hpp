#pragma once
#include <vector>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <iostream>

#include "Model/ModelBase.hpp"

struct OscillatorParams
{
    double gamma = 0.1;
    double w02 = 1.0;
    double w = 1.0;
    double A = 0.5;
    double dt = 0.1;
    double time = 600.0;

    double x0 = 0.0;
    double v0 = 0.0;

    double E = 1.6;
    PreisachModelBase* Preisach = nullptr;
};

inline std::pair<std::vector<double>, std::vector<double>> OscillatorSolver(const OscillatorParams& params)
{
    const auto steps = static_cast<uint32_t>(params.time / params.dt);
    std::vector<double> x(steps);
    std::vector<double> v(steps);

    x[0] = params.x0;
    v[0] = params.v0;

    for (uint32_t i = 1; i < steps; i++)
    {
        const double t = i * params.dt;
        const double output = params.Preisach == nullptr ? 0.0 : params.Preisach->P(x[i - 1]);
        const double force = params.A * std::sin(params.w * t) + params.E * output;
        const double acceleration = force - params.gamma * v[i - 1] - params.w02 * x[i - 1];
        v[i] = v[i - 1] + acceleration * params.dt;
        x[i] = x[i - 1] + v[i] * params.dt;
        // std::cout << x[i] << '\n';
    }

    return {x, v};
}
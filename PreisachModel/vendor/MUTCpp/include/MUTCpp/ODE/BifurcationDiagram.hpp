// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <unordered_map>

#include "DSArgs.hpp"
#include "Json/JsonDocument.hpp"

namespace mc
{
    namespace ode
    {
        inline json::JsonDocument BifurcationDiagram(Ref<DynamicalSystem> system,
                                                     const std::pair<std::string, std::vector<Vote>> &bifurcationParam,
                                                     double researchTime,
                                                     double timeForward)
        {
            const auto &[param_name, param_value] = bifurcationParam;

            json::JsonDocument message({"name", "trajs", param_name});
            message.AddField("name", "BifurcationDiagram");

            WriteVotesToDoc(message, param_name, param_value, param_value[0].Type());

            std::vector<Eigen::MatrixXd> trajs;

            for (const auto &val : param_value)
            {
                system->Reset();

                system->Forward(timeForward);

                system->SetArg(param_name, val);

                const auto traj = system->Forward(researchTime);
                trajs.push_back(traj);
            }

            message.AddField("trajs", trajs);

            return message;
        }
    }
}

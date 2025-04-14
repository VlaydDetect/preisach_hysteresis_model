// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <vector>
#include <unordered_map>

namespace mc
{
    namespace utils
    {
        template <typename Key, typename Val>
        std::vector<std::unordered_map<Key, Val>> getNestedCombinations(
            const std::unordered_map<Key, std::vector<Val>> &data)
        {
            if (data.empty())
                return {};

            std::vector<std::unordered_map<Key, Val>> result;

            std::vector<Key> keys;
            std::vector<std::vector<Val>> vals;
            for (const auto &[key, vec] : data)
            {
                keys.push_back(key);
                vals.push_back(vec);
            }

            std::vector<uint32> indices(vals.size(), 0);

            while (true)
            {
                std::unordered_map<Key, Val> combination;
                for (size_t i = 0; i < vals.size(); ++i)
                {
                    combination[keys[i]] = vals[i][indices[i]];
                }
                result.push_back(combination);

                size_t level = vals.size();
                while (level > 0)
                {
                    --level;
                    if (++indices[level] < vals[level].size())
                        break;
                    indices[level] = 0;
                }

                if (level == 0 && indices[0] == 0)
                    break;
            }

            return result;
        }
    }
}

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

            std::vector<Key> keys;
            std::vector<std::vector<Val>> vals;
            for (const auto &[k,v] : data)
            {
                keys.push_back(k);
                vals.push_back(v);
            }

            std::vector<size_t> indices(vals.size(), 0);
            std::vector<std::unordered_map<Key, Val>> result;

            while (true)
            {
                std::unordered_map<Key, Val> comb;
                for (size_t i = 0; i < vals.size(); ++i)
                {
                    comb[keys[i]] = vals[i][indices[i]];
                }
                result.push_back(std::move(comb));

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

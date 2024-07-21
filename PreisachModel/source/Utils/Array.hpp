#pragma once
#include <vector>
#include <optional>
#include <enumerate.hpp>

inline std::optional<size_t> FindFirstGreater(const std::vector<double>& vec, double x)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] > x) return i;
    }

    return std::nullopt;
}

inline std::optional<size_t> FindLastLess(const std::vector<double>& vec, double x)
{
    for (size_t i = vec.size() - 1; i > 0; i--)
    {
        if (vec[i] < x) return i;
    }

    return std::nullopt;
}

inline std::optional<size_t> FindLastGreater(const std::vector<double>& vec, double x)
{
    for (size_t i = vec.size() - 1; i > 0; i--)
    {
        if (vec[i] > x) return i;
    }

    return std::nullopt;
}

inline std::optional<size_t> FindFirstLess(const std::vector<double>& vec, double x)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] < x) return i;
    }

    return std::nullopt;
}

template<typename T>
std::vector<std::pair<size_t, T>> Enumerate(const std::vector<T>& vec)
{
    std::vector<std::pair<size_t, T>> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
    {
        result.emplace_back(i, vec[i]);
    }

    return result;
}

// struct Bound
// {
//     double value = NAN;
//     int32_t index = -1;
//
//     Bound(double val, int32_t idx) : value(val), index(idx) {}
// };

// inline std::vector<std::pair<size_t, size_t>> FindBoundsIndices(const std::vector<double>& vec, double x)
// {
//     auto sortedVec = vec;
//     std::erase_if(sortedVec, [](const double elem) { return nc::isnan(elem); });
//     std::ranges::sort(sortedVec);
//
//     auto lower = std::ranges::lower_bound(sortedVec, x);
//     if (lower == sortedVec.begin())
//     {
//         return {{nc::constants::nan, sortedVec.front()}};
//     }
//
//     if (lower == sortedVec.end())
//     {
//         return {{sortedVec.back(), nc::constants::nan}};
//     }
//
//     double leftBound = *(lower - 1);
//     double rightBound = *lower;
//
//     size_t leftIndex = std::distance(vec.begin(), std::ranges::find(vec, leftBound));
//     size_t rightIndex = std::distance(vec.begin(), std::ranges::find(vec, rightBound));
//     
//     return {{leftIndex, rightIndex}};
// }

inline std::pair<int32_t, int32_t> FindBoundsIndices(const std::vector<double>& vec, double x)
{
    auto sortedVec = vec;
    std::erase_if(sortedVec, [](const double elem) { return nc::isnan(elem); });
    std::ranges::sort(sortedVec);

    auto lower = std::ranges::lower_bound(sortedVec, x);
    if (lower == sortedVec.begin())
    {
        return {-1, sortedVec.front()};
    }

    if (lower == sortedVec.end())
    {
        return {sortedVec.back(), -1};
    }

    double leftBound = *(lower - 1);
    double rightBound = *lower;

    int32_t leftIndex = std::distance(vec.begin(), std::ranges::find(vec, leftBound));
    int32_t rightIndex = std::distance(vec.begin(), std::ranges::find(vec, rightBound));
    
    return {leftIndex, rightIndex};
}
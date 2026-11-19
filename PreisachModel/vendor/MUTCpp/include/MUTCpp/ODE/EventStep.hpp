#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace mc::ode
{
    struct EventTolerance
    {
        double time = 1e-10;
        double state = 1e-9;
        int subdivisions = 32;
    };

    namespace detail
    {
        struct EventSurface
        {
            int coordinate;
            double level;
            std::optional<Eigen::Index> extremum;

            double Value(const Eigen::Vector2d &x) const { return x(coordinate) - level; }
            bool operator==(const EventSurface &) const = default;
        };

        struct LocatedEvent
        {
            double elapsed;
            std::vector<EventSurface> surfaces;
        };

        template <typename Sample>
        std::optional<LocatedEvent> FindFirstCrossing(const Sample &sample, double step,
                                                       const std::vector<EventSurface> &surfaces,
                                                       const EventTolerance &tolerance)
        {
            struct Endpoint
            {
                double t;
                double value;
            };

            const Eigen::Vector2d start = sample(0.0);
            std::vector<Endpoint> left;
            left.reserve(surfaces.size());
            for (const auto &surface : surfaces)
            {
                left.push_back({0.0, surface.Value(start)});
            }

            for (int part = 1; part <= tolerance.subdivisions; ++part)
            {
                const double right_time = step * (static_cast<double>(part) / tolerance.subdivisions);
                const Eigen::Vector2d right_state = sample(right_time);
                std::optional<LocatedEvent> first;
                for (std::size_t i = 0; i < surfaces.size(); ++i)
                {
                    const auto &surface = surfaces[i];
                    const double value = surface.Value(right_state);
                    if (!std::isfinite(value))
                    {
                        throw std::runtime_error("Non-finite event function");
                    }
                    double right_sign = value;
                    if (right_sign == 0.0 && part == tolerance.subdivisions && left[i].value != 0.0)
                    {
                        // Check the outgoing side to distinguish a corner at the endpoint from a touch.
                        right_sign = surface.Value(sample(right_time + tolerance.time));
                    }
                    if (left[i].value != 0.0 && right_sign != 0.0 &&
                        std::signbit(left[i].value) != std::signbit(right_sign))
                    {
                        double lo = left[i].t;
                        double hi = right_time;
                        double root = hi;
                        if (value != 0.0)
                        {
                            for (;;)
                            {
                                root = std::midpoint(lo, hi);
                                const double g = surface.Value(sample(root));
                                if (g == 0.0 || (hi - lo <= tolerance.time && std::abs(g) <= tolerance.state))
                                {
                                    break;
                                }
                                if (root == lo || root == hi || !std::isfinite(g))
                                {
                                    throw std::runtime_error("Event cannot be localized within the requested tolerances");
                                }
                                if (std::signbit(g) == std::signbit(left[i].value))
                                    lo = root;
                                else
                                    hi = root;
                            }
                        }
                        if (!first || root + tolerance.time < first->elapsed)
                        {
                            first = LocatedEvent{root, {surface}};
                        }
                        else if (std::abs(root - first->elapsed) <= tolerance.time)
                        {
                            first->elapsed = std::min(root, first->elapsed);
                            first->surfaces.push_back(surface);
                        }
                    }
                    if (value != 0.0)
                    {
                        left[i] = {right_time, value};
                    }
                }
                if (first)
                {
                    return first;
                }
            }
            return std::nullopt;
        }
    }
}

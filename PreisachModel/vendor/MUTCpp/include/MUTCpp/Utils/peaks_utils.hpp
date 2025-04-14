// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <print>
#include <tuple>

#include "Functions/copy.hpp"
#include "Functions/all_comprasions.hpp"
#include "Functions/empty.hpp"
#include "MUTCpp/Functions/argsort.hpp"
#include "MUTCpp/Functions/ceil.hpp"
#include "MUTCpp/Functions/max.hpp"
#include "MUTCpp/Functions/min.hpp"
#include "MUTCpp/Functions/ones.hpp"
#include "MUTCpp/Functions/ones_like.hpp"
#include "MUTCpp/Functions/range.hpp"
#include "MUTCpp/Functions/vstack.hpp"
#include "MUTCpp/Vector/StaticVector.hpp"
#include "Matrix.hpp"

namespace mc
{

    /// @brief Find local maxima in a 1D array.
    /// @details This function finds all local maxima in a 1D array and returns the indices
    ///     for their edges and midpoints (rounded down for even plateau sizes).
    ///
    /// @note:
    /// - This function is fast enough and can
    /// detect maxima that are more than one sample wide. However, this comes at
    /// the cost of being only applicable to 1D arrays.
    /// - A maximum is defined as one or more samples of equal value that are
    /// surrounded on both sides by at least one smaller sample.
    ///     
    /// @param signal: The array to search for local maxima.
    /// @return An array of three matrices:
    ///     `midpoints`: Returns indices of midpoints of local maxima in `x`.
    ///     `left_edges`: Returns indices of edges to the left of local maxima in `x`.
    ///     `right_edges`: Returns indices of edges to the right of local maxima in `x`.
    inline std::array<Matrix<uint32>, 3> local_maxima_1d(const Matrix<double> &signal)
    {
        if (!signal.isflat())
        {
            THROW_INVALID_ARGUMENT_ERROR("`x` must be a 1-D array");
        }

        Matrix<double> x = copy(signal);
        if (x.numCols() == 1)
        {
            x.reshape(x.numRows());
        }

        // Preallocate, there can't be more maxima than half the size of `x`

        Matrix<uint32> midpoints = empty<uint32>(x.numRows(), x.numCols() / 2),
                       left_edges = empty<uint32>(x.numRows(), x.numCols() / 2),
                       right_edges = empty<uint32>(x.numRows(), x.numCols() / 2);
        //Pointer to the end of valid area in allocated arrays
        uint32 m = 0;

        // Pointer to current sample, first one can't be maxima
        uint32 i = 1;
        uint32 i_max = x.numCols() - 1; // Last sample can't be maxima
        while (i < i_max)
        {
            // Test if the previous sample is smaller
            if (x[i - 1] < x[i])
            {
                uint32 i_ahead = i + 1; // Index to look ahead of current sample

                // Find the next sample that is unequal to x[i]
                while (i_ahead < i_max && x[i_ahead] == x[i])
                {
                    i_ahead++;
                }

                // Maxima is found if the next unequal sample is smaller than x[i]
                if (x[i_ahead] < x[i])
                {
                    // left_edges.put(m, i);
                    left_edges[m] = i;
                    // right_edges.put(m, i_ahead - 1);
                    right_edges[m] = i_ahead - 1;
                    // midpoints.put(m, (left_edges[m] + right_edges[m]) / 2);
                    midpoints[m] = (left_edges[m] + right_edges[m]) / 2;
                    m++;
                    // Skip samples that can't be maximum
                    i = i_ahead;
                }
            }
            i++;
        }

        // Keep only valid part of array memory.
        midpoints.resizeSlow(1, m);
        left_edges.resizeSlow(1, m);
        right_edges.resizeSlow(1, m);

        return {midpoints, left_edges, right_edges};
    }

    /// Parse condition arguments for `find_peaks`.
    /// @param interval: Either a number or a 2-element sequence of the former. The first value is always
    ///     interpreted as `imin` and the second, if supplied, as `imax`.
    /// @return imin, imax: number or ``consts::nan``. Minimal and maximal value in `argument`.
    inline std::array<double, 2> unpack_condition_args(const StaticVector<double, 2> &interval)
    {
        if (interval.empty())
        {
            return {consts::nan, consts::nan};
        }

        double imin = interval[0];
        double imax = consts::nan;
        if (interval.size() == 2)
        {
            imax = interval[1];
        }

        return {imin, imax};
    }

    /// Evaluate where the generic property of peaks confirms to an interval.
    /// @param peak_properties: An array with properties for each peak.
    /// @param pmin: Lower interval boundary for `peak_properties`. ``None`` is interpreted as an open border.
    /// @param pmax: Upper interval boundary for `peak_properties`. ``None`` is interpreted as an open border.
    /// @return: A boolean mask evaluating to true where `peak_properties` confirms to the interval.
    inline Matrix<bool> select_by_property(const Matrix<double> &peak_properties, const double pmin, const double pmax)
    {
        auto keep = ones_like<bool>(peak_properties);
        if (!isnan(pmin))
        {
            keep &= (pmin <= peak_properties);
        }
        if (!isnan(pmax))
        {
            keep &= (peak_properties <= pmax);
        }
        return keep;
    }

    /// Evaluate which peaks fulfill the threshold condition.
    /// @param x: A 1-D array which is indexable by `peaks`.
    /// @param peaks: Indices of peaks in `x`.
    /// @param tmin, tmax: Minimal and / or maximal required thresholds. ``consts::nan`` is interpreted as an open border.
    /// @return An array of three matrices:
    ///     keep: A boolean mask evaluating to true where `peaks` fulfill the threshold condition.
    ///     left_thresholds, right_thresholds: Array matching `peak` containing the thresholds of each peak on both sides;
    inline std::tuple<Matrix<bool>, Matrix<double>, Matrix<double>> select_by_peak_threshold(
        const Matrix<double> &x, const Matrix<uint32> &peaks,
        double tmin, double tmax)
    {
        auto stacked_thresholds = vstack({x[peaks] - x[peaks - static_cast<uint32>(1)],
                                          x[peaks] - x[peaks + static_cast<uint32>(1)]});
        Matrix<bool> keep = ones_like<bool>(peaks);
        if (!isnan(tmin))
        {
            auto min_thresholds = min(stacked_thresholds);
            keep &= (tmin <= min_thresholds);
        }
        if (!isnan(tmax))
        {
            auto max_thresholds = max(stacked_thresholds);
            keep &= (max_thresholds <= tmax);
        }
        return {keep, stacked_thresholds(0, stacked_thresholds.cSlice()),
                stacked_thresholds(1, stacked_thresholds.cSlice())};
    }

    /// Evaluate which peaks fulfill the distance condition.
    /// @param peaks: Indices of peaks in `vector`.
    /// @param priority: An array matching `peaks` used to determine the priority of each peak.
    ///     A peak with a higher priority value is kept over one with a lower one.
    /// @param distance: Minimal distance that peaks must be spaced.
    /// @return: A boolean mask evaluating to true where `peaks` fulfill the distance condition.
    inline Matrix<bool> select_by_peak_distance(const Matrix<uint32> &peaks, const Matrix<double> &priority,
                                                double distance)
    {

        int32 peaks_size = peaks.numCols();
        // Round up because actual peak distance can only be natural number
        uint32 distance_ = static_cast<uint32>(ceil(distance));
        Matrix<bool> keep = ones<bool>(1, peaks_size);

        // Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
        // for `peaks` sorted by position). This allows to iterate `peaks` and `keep`
        // with `j` by order of `priority` while still maintaining the ability to
        // step to neighbouring peaks with (`j` + 1) or (`j` - 1).
        Matrix<uint32> priority_to_position = argsort(priority);

        for (const auto i : range(peaks_size - 1, -1, -1))
        {
            // "Translate" `i` to `j` which points to the current peak whose neighbors are to be evaluated
            auto j = priority_to_position[i];
            if (keep[j] == 0)
            {
                // Skip evaluation for peak already marked as "don't keep"
                continue;
            }

            auto k = static_cast<int32>(j - 1);
            // Flag "earlier" peaks for removal until minimal distance is exceeded
            while (0 <= k && peaks[j] - peaks[k] < distance_)
            {
                keep[k] = false;
                k--;
            }

            k = j + 1;
            while (k < peaks_size && peaks[k] - peaks[j] < distance_)
            {
                keep[k] = false;
                k++;
            }
        }

        return keep;
    }

    /// Ensure argument `wlen` is of type `np.intp` and larger than 1.
    /// The original `value` rounded up to an integer or -1 if `value` was None.
    inline int32 arg_wlen_as_expected(int32 value)
    {
        if (value <= 0)
        {
            // -1 signals that no value was
            value = -1;
        }
        else if (value < 1)
        {
            value = 1;
        }
        else
        {
            std::string errMsg = "`wlen` must be larger than 1, was " + value;
            THROW_INVALID_ARGUMENT_ERROR(errMsg);
        }

        return value;
    }

    /// Calculate the prominence of each peak in a signal.
    /// @param x: A signal with peaks.
    /// @param peaks: Indices of peaks in `x`.
    /// @param wlen: A window length in samples (see `peak_prominences`) which is rounded up
    ///     to the nearest odd integer. If smaller than 2, the entire signal `x` is used.
    /// @return An array of three matrices:
    ///     `prominences`: The calculated prominences for each peak in `peaks`.
    ///     `left_bases`, `right_bases`: The peaks' bases as indices in `x` to the left and right of each peak.
    inline std::tuple<Matrix<double>, Matrix<uint32>, Matrix<uint32>> peak_prominences(
        const Matrix<double> &x, const Matrix<uint32> &peaks, int32 wlen)
    {
        auto prominences = empty<double>(1, peaks.numCols());
        auto left_bases = empty<uint32>(1, peaks.numCols()), right_bases = empty<uint32>(1, peaks.numCols());
        bool show_warning = false;

        for (const auto peak_nr : range(peaks.numCols()))
        {
            int32 peak = static_cast<int32>(peaks[peak_nr]);
            int32 i_min = 0;
            int32 i_max = x.numCols() - 1;
            if (!all_less(i_min, peak, i_max))
            {
                std::string errMsg = std::format("peak {} is not a valid index for `x`", peak);
                THROW_INVALID_ARGUMENT_ERROR(errMsg);
            }

            if (2 <= wlen)
            {
                // Adjust a window around the evaluated peak (within bounds);
                // if wlen is even the resulting window length is implicitly
                // rounded to the next odd integer
                i_min = max(peak - wlen / 2, i_min);
                i_max = min(peak + wlen / 2, i_max);
            }

            // Find the left base in an interval [i_min, peak]
            int32 i = left_bases[peak_nr] = peak;
            double left_min = x[peak];
            while (i_min <= i and x[i] <= x[peak])
            {
                if (x[i] < left_min)
                {
                    left_min = x[i];
                    left_bases[peak_nr] = i;
                }
                i -= 1;
            }

            // Find the right base in an interval [peak, i_max]
            i = right_bases[peak_nr] = peak;
            double right_min = x[peak];
            while (i <= i_max and x[i] <= x[peak])
            {
                if (x[i] < right_min)
                {
                    right_min = x[i];
                    right_bases[peak_nr] = i;
                }
                i += 1;
            }

            prominences[peak_nr] = x[peak] - max(left_min, right_min);
            if (prominences[peak_nr] == 0)
            {
                show_warning = true;
            }
        }

        if (show_warning)
        {
            std::print("WARNING: some peaks have a prominence of 0");
        }

        return {prominences, left_bases, right_bases};
    }

    /// Calculate the width of each peak in a signal.
    /// @param x: A signal with peaks.
    /// @param peaks: Indices of peaks in `x`.
    /// @param rel_height: Chooses the relative height at which the peak width is measured as a percentage of its prominence (see `peak_widths`).
    /// @param prominences: Prominences of each peak in `peaks` as returned by `peak_prominences`.
    /// @param left_bases, right_bases: Left and right bases of each peak in `peaks` as returned by `peak_prominences`.
    /// @return An array of four matrices:
    ///     `widths`: The widths for each peak in samples.
    ///     `width_heights`: The height of the contour lines at which the `widths` where evaluated.
    ///     `left_ips`, `right_ips`: Interpolated positions of left and right intersection points of a horizontal line at the respective evaluation height.
    inline std::array<Matrix<double>, 4> peak_widths(const Matrix<double> &x, const Matrix<uint32> &peaks,
                                                     double rel_height,
                                                     const Matrix<double> &prominences,
                                                     const Matrix<uint32> &left_bases,
                                                     const Matrix<uint32> &right_bases)
    {
        Matrix<double> widths = empty<double>(1, peaks.numCols()), width_heights = empty<double>(1, peaks.numCols()),
                       left_ips = empty<double>(1, peaks.numCols()), right_ips = empty<double>(1, peaks.numCols());
        bool show_warning = false;

        if (rel_height < 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("`rel_height` must be greater or equal to 0.0");
        }
        if (!all_equals(peaks.numCols(), prominences.numCols(), left_bases.numCols(), right_bases.numCols()))
        {
            THROW_INVALID_ARGUMENT_ERROR("arrays in `prominence_data` must have the same shape as `peaks`");
        }

        for (const auto p : range(peaks.numCols()))
        {
            uint32 i_min = left_bases[p];
            uint32 i_max = right_bases[p];
            uint32 peak = peaks[p];
            if (!(0 <= i_min && i_min <= peak && peak <= i_max && i_max < x.numCols()))
            {
                THROW_INVALID_ARGUMENT_ERROR(std::format("prominence data is invalid for peak {}", peak));
            }
            double height = width_heights[p] = x[peak] - prominences[p] * rel_height;

            // Find intersection point on the left side
            uint32 i = peak;
            while (i_min < i && height < x[i])
            {
                i--;
            }
            double left_ip = i;
            if (x[i] < height)
            {
                // Interpolate if true intersection height is between samples
                left_ip += (height - x[i]) / (x[i + 1] - x[i]);
            }

            // Find intersection point on the right side
            i = peak;
            while (i < i_max && height < x[i])
            {
                i++;
            }
            double right_ip = i;
            if (x[i] < height)
            {
                // Interpolate if true intersection height is between samples
                right_ip -= (height - x[i]) / (x[i - 1] - x[i]);
            }

            widths[p] = right_ip - left_ip;
            if (widths[p] == 0)
            {
                show_warning = true;
            }
            left_ips[p] = left_ip;
            right_ips[p] = right_ip;
        }

        if (show_warning)
        {
            std::print("WARNING: some peaks have a width of 0");
        }
        return {widths, width_heights, left_ips, right_ips};
    }
}

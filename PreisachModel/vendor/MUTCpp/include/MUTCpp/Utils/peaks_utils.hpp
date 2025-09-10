// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <print>
#include <tuple>

#include "Functions/all_comprasions.hpp"
#include "MUTCpp/Functions/max.hpp"
#include "MUTCpp/Functions/min.hpp"
#include "MUTCpp/Functions/ones.hpp"
#include "MUTCpp/Functions/range.hpp"
#include "MUTCpp/Vector/StaticVector.hpp"

namespace mc
{
    using ArrayXd = Eigen::ArrayXd;
    using ArrayXI = Eigen::Array<Eigen::Index, Eigen::Dynamic, 1>;
    using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;


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
    inline std::array<ArrayXI, 3> local_maxima_1d(const ArrayXd &signal)
    {
        if (signal.rows() <= 2)
        {
            return {ArrayXI(), ArrayXI(), ArrayXI()};
        }

        std::vector<Eigen::Index> mids, lefts, rights;
        mids.reserve(signal.size() / 2);
        lefts.reserve(signal.size() / 2);
        rights.reserve(signal.size() / 2);

        const int n = static_cast<int>(signal.size());
        int i = 1;
        const int i_max = n - 1;

        while (i < i_max)
        {
            if (signal[i - 1] < signal[i])
            {
                int i_ahead = i + 1;
                while (i_ahead < i_max && signal[i_ahead] == signal[i])
                {
                    ++i_ahead;
                }
                if (signal[i_ahead] < signal[i])
                {
                    lefts.push_back(i);
                    rights.push_back(i_ahead - 1);
                    mids.push_back((i + i_ahead - 1) / 2);
                    i = i_ahead;
                }
            }
            ++i;
        }

        const ArrayXI mid = Eigen::Map<ArrayXI>(mids.data(), mids.size());
        const ArrayXI left = Eigen::Map<ArrayXI>(lefts.data(), lefts.size());
        const ArrayXI right = Eigen::Map<ArrayXI>(rights.data(), rights.size());
        return {mid, left, right};
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
        double imax = interval.size() == 2 ? interval[1] : consts::nan;

        return {imin, imax};
    }

    /// Evaluate where the generic property of peaks confirms to an interval.
    /// @param peak_properties: An array with properties for each peak.
    /// @param pmin: Lower interval boundary for `peak_properties`. ``None`` is interpreted as an open border.
    /// @param pmax: Upper interval boundary for `peak_properties`. ``None`` is interpreted as an open border.
    /// @return: A boolean mask evaluating to true where `peak_properties` confirms to the interval.
    inline ArrayXb select_by_property(const ArrayXd &peak_properties, const double pmin, const double pmax)
    {
        ArrayXb keep = ArrayXb::Constant(peak_properties.size(), true);
        if (!isnan(pmin))
        {
            keep = keep && (peak_properties >= pmin);
        }
        if (!isnan(pmax))
        {
            keep = keep && (peak_properties <= pmax);
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
    inline std::tuple<ArrayXb, ArrayXd, ArrayXd> select_by_peak_threshold(
        const ArrayXd &x, const ArrayXI &peaks, double tmin, double tmax)
    {
        ArrayXd center = x(peaks);             // x[peaks]
        ArrayXd left   = x(peaks - 1);  // x[peaks-1]
        ArrayXd right  = x(peaks + 1);  // x[peaks+1]

        // 2 x N
        Eigen::MatrixXd stacked_thresholds(2, peaks.size());
        stacked_thresholds.row(0) = (center - left).matrix().transpose();
        stacked_thresholds.row(1) = (center - right).matrix().transpose();

        ArrayXb keep = ArrayXb::Constant(peaks.size(), true);
        
        if (!isnan(tmin))
        {
            Eigen::RowVectorXd min_thresholds = stacked_thresholds.colwise().minCoeff();
            keep = keep && (min_thresholds.array() >= tmin);
        }
        if (!isnan(tmax))
        {
            Eigen::RowVectorXd max_thresholds = stacked_thresholds.colwise().maxCoeff();
            keep = keep && (max_thresholds.array() <= tmax);
        }

        return {
            keep,
            stacked_thresholds.row(0).transpose().array(),
            stacked_thresholds.row(1).transpose().array()
        };
    }

    /// Evaluate which peaks fulfill the distance condition.
    /// @param peaks: Indices of peaks in `vector`.
    /// @param priority: An array matching `peaks` used to determine the priority of each peak.
    ///     A peak with a higher priority value is kept over one with a lower one.
    /// @param distance: Minimal distance that peaks must be spaced.
    /// @return: A boolean mask evaluating to true where `peaks` fulfill the distance condition.
    inline ArrayXb select_by_peak_distance(const ArrayXI &peaks, const ArrayXd &priority, double distance)
    {
        const int peaks_size = peaks.size();
        if (peaks_size == 0)
            return ArrayXb();
        uint32 dist = static_cast<uint32>(std::ceil(distance));
        ArrayXb keep = ArrayXb::Constant(peaks_size, true);

        // Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
        // for `peaks` sorted by position). This allows to iterate `peaks` and `keep`
        // with `j` by order of `priority` while still maintaining the ability to
        // step to neighbouring peaks with (`j` + 1) or (`j` - 1).
        ArrayXI priority_to_position = Eigen::argsort(priority).array();

        for (const auto i : range(peaks_size - 1, -1, -1))
        {
            // "Translate" `i` to `j` which points to the current peak whose neighbors are to be evaluated
            int j = priority_to_position[i];
            if (keep[j] == 0)
            {
                // Skip evaluation for peak already marked as "don't keep"
                continue;
            }

            int k = j - 1;
            // Flag "earlier" peaks for removal until minimal distance is exceeded
            while (0 <= k && peaks[j] - peaks[k] < dist)
            {
                keep[k] = false;
                k--;
            }

            k = j + 1;
            while (k < peaks_size && peaks[k] - peaks[j] < dist)
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
    inline std::tuple<ArrayXd, ArrayXI, ArrayXI> peak_prominences(
        const ArrayXd &x, const ArrayXI &peaks, int32 wlen)
    {
        const int n = peaks.size();
        ArrayXd prominences(n);
        ArrayXI left_bases(n), right_bases(n);
        bool show_warning = false;

        for (const auto peak_nr : range(n))
        {
            int32 peak = peaks[peak_nr];
            int32 i_min = 0;
            int32 i_max = x.size() - 1;
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
    inline std::array<ArrayXd, 4> peak_widths(const ArrayXd &x, const ArrayXI &peaks,
                                                     double rel_height,
                                                     const ArrayXd &prominences,
                                                     const ArrayXI &left_bases,
                                                     const ArrayXI &right_bases)
    {
        const int n = peaks.size();
        ArrayXd widths(n), width_heights(n), left_ips(n), right_ips(n);
        bool show_warning = false;

        if (rel_height < 0)
        {
            THROW_INVALID_ARGUMENT_ERROR("`rel_height` must be greater or equal to 0.0");
        }
        if (!all_equals(peaks.size(), prominences.size(), left_bases.size(), right_bases.size()))
        {
            THROW_INVALID_ARGUMENT_ERROR("arrays in `prominence_data` must have the same shape as `peaks`");
        }

        for (const auto p : range(n))
        {
            uint32 i_min = left_bases[p];
            uint32 i_max = right_bases[p];
            uint32 peak = peaks[p];
            if (!(0 <= i_min && i_min <= peak && peak <= i_max && i_max < x.size()))
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

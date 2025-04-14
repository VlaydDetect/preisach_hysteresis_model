// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"
#include "isnan.hpp"
#include "Debug/Profile.hpp"
#include "MUTCpp/Utils/peaks_utils.hpp"
#include "Vector/StaticVector.hpp"

namespace mc
{
    /**
     * A struct containing properties of the peaks which were
     * calculated as intermediate results during evaluation of the specified conditions.
     *
     * @details
     * 'peak_heights': If `height` is given, the height of each peak in `x`.\n
     * 'left_thresholds', 'right_thresholds': If `threshold` is given, these keys contain a peaks vertical distance to its neighboring samples.\n
     * 'prominences', 'right_bases', 'left_bases': If `prominence` is given, these keys are accessible. See `peak_prominences` for a description of their content.\n
     * 'width_heights', 'left_ips', 'right_ips': If `width` is given, these keys are accessible. See `peak_widths` for a description of their content.\n
     * 'plateau_sizes', left_edges', 'right_edges': If `plateau_size` is given, these keys are accessible and contain the indices of a peak's edges (edges are still part of the plateau) and the calculated plateau sizes.\n
     * @details
     * See Also:\n
     * peak_prominences: Directly calculate the prominence of peaks.\n
     * peak_widths: Directly calculate the width of peaks.\n
    */
    struct PeaksProperties
    {
    public:
        Matrix<double> peak_heights = {consts::nan};
        Matrix<double> left_thresholds = {consts::nan}, right_thresholds = {consts::nan};
        Matrix<double> prominences = {consts::nan};
        Matrix<uint32> right_bases, left_bases;
        Matrix<double> widths = {consts::nan}, width_heights = {consts::nan}, left_ips = {consts::nan}, right_ips = {consts::nan};
        Matrix<uint32> plateau_sizes, left_edges, right_edges;

        auto as_tuple()
        {
            return std::forward_as_tuple(peak_heights, left_thresholds, right_thresholds, prominences, right_bases,
                                         left_bases,
                                         width_heights, left_ips, right_ips, plateau_sizes, left_edges, right_edges);
        }

        template <typename Func>
        void foreach(Func &&func)
        {
            foreach(std::forward<Func>(func), std::make_index_sequence<std::tuple_size_v<decltype(as_tuple())>>{});
        }

    private:
        template <typename Func, std::size_t... Is>
        void foreach(Func &&func, std::index_sequence<Is...>)
        {
            (func(std::get<Is>(as_tuple())), ...);
        }
    };

    /// @brief Find peaks inside a signal (1-D Matrix) based on peak properties.
    /// This function takes a 1-D array and finds all local maxima by simple comparison of neighboring values.
    /// Optionally, a subset of these peaks can be selected by specifying conditions for a peak's properties.
    ///
    /// Referenced by scipy.signal.find_peaks [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html].
    ///
    /// @note: In the context of this function, a peak or local maximum is defined as any
    /// sample whose two direct neighbors have a smaller amplitude. For flat peaks
    /// (more than one sample of equal amplitude wide) the index of the middle
    /// sample is returned (rounded down in case the number of samples is even).
    /// For noisy signals, the peak locations can be off because the noise might
    /// change the position of local maxima. In those cases, consider smoothing the
    /// signal before searching for peaks or using other peak finding and fitting
    /// methods (like `find_peaks_cwt`).\n
    ///
    /// @note:
    /// Some additional comments on specifying conditions:
    /// - Almost all conditions (excluding `distance` and 'wlen') can be given as half-open or
    ///   closed intervals, e.g., ``1`` or ``(1, consts::nan)`` defines the half-open
    ///   interval [1, @f$\infinity@f$] while ``(consts::nan, 1)`` defines the interval
    ///   [@f$-\infinity@f$, 1]. The empty open interval ``(@f$\infinity@f$, @f$\infinity@f$)`` can be specified
    ///   as well, which returns the matching properties without exclusion of peaks.
    /// - The border is always included in the interval used to select valid peaks.
    /// - The conditions are evaluated in the following order: `plateau_size`,
    ///   `height`, `threshold`, `distance`, `prominence`, `width`. In most cases,
    ///   this order is the fastest one because faster operations are applied first
    ///   to reduce the number of peaks that need to be evaluated later.
    /// - While indices in `peaks` are guaranteed to be at least `distance` samples
    ///   apart, edges of flat peaks may be closer than the allowed `distance`.
    /// - Use `wlen` to reduce the time it takes to evaluate the conditions for
    ///   `prominence` or `width` if `x` is large or has many local maxima
    ///   (see `peak_prominences`).
    ///
    /// @param x: A signal with peaks.
    /// @param height (Optional, default empty): Required height of peaks. Either a number,``consts::nan``
    ///     or a 2-element array of the former. The first element is
    ///     always interpreted as the minimal and the second, if supplied, as the
    ///     maximal required height.
    /// @param threshold (Optional, default empty): Required threshold of peaks, the vertical distance to its neighboring
    ///     samples. Either a number, ``consts::nan`` or a 2-element sequence of the former.
    ///     The first element is always interpreted as the  minimal and the second, if supplied, as the maximal required threshold.
    /// @param distance (Optional, default 0): Required minimal horizontal distance (>= 1) in samples between
    ///     neighboring peaks.Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
    /// @param prominence (Optional, default empty): Required prominence of peaks. Either a number, ``consts::nan``
    ///     or a 2-element sequence of the former. The first element is always interpreted as the minimal and the second,
    ///     if supplied, as the maximal required prominence.
    /// @param width (Optional, default empty): Required width of peaks in samples. Either a number, ``consts::nan``
    ///     or a 2-element sequence of the former. The first element is always interpreted as the  minimal and the second,
    ///     if supplied, as the maximal required width.
    /// @param wlen (Optional, default 0): Used for calculation of the peaks prominences, thus it is only used if
    ///     one of the arguments `prominence` or `width` is given. See argument
    ///     `wlen` in `peak_prominences` for a full description of its effects.
    /// @param rel_height (Optional, default consts::nan): Used for calculation of the peaks width, thus it is only used if `width`
    ///     is given. See argument `rel_height` in `peak_widths` for a full description of its effects.
    /// @param plateau_size (Optional, default empty): Required size of the flat top of peaks in samples. Either a number,
    ///     ``consts::nan`` or a 2-element sequence of the former. The first element is always interpreted as the minimal
    ///     and the second, if supplied as the maximal required plateau size.
    /// @return Pair of matrix of peaks and `@link PeaksProperties`.
    inline std::pair<Matrix<uint32>, PeaksProperties> find_peaks(const Matrix<double> &x,
                                                                 const StaticVector<double, 2> &height = {},
                                                                 const StaticVector<double, 2> &threshold = {},
                                                                 double distance = consts::nan,
                                                                 const StaticVector<double, 2> &prominence = {},
                                                                 const StaticVector<double, 2> &width = {},
                                                                 int wlen = 0, double rel_height = consts::nan,
                                                                 const StaticVector<double, 2> &plateau_size = {})
    {
        AL_PROFILE_FUNC("find_peaks");
        if (!x.isflat())
        {
            THROW_INVALID_ARGUMENT_ERROR("`x` must be a 1-D array");
        }

        if (!isnan(distance) && distance < 1)
        {
            THROW_INVALID_ARGUMENT_ERROR("`distance` must be greater or equal to 1");
        }

        auto [peaks, left_edges, right_edges] = local_maxima_1d(x);
        PeaksProperties properties;

        if (!plateau_size.empty())
        {
            // Evaluate plateau size
            auto plateau_sizes = (right_edges - left_edges) + static_cast<uint32>(1);
            auto [pmin, pmax] = unpack_condition_args(plateau_size);
            auto keep = select_by_property(plateau_sizes.astype<double>(), pmin, pmax);
            peaks = peaks[keep];
            properties.plateau_sizes = plateau_sizes[keep];
            properties.left_edges = left_edges[keep];
            properties.right_edges = right_edges[keep];
        }

        if (!height.empty())
        {
            // Evaluate height condition
            auto peak_heights = x[peaks];
            auto [hmin, hmax] = unpack_condition_args(height);
            auto keep = select_by_property(peak_heights, hmin, hmax);
            peaks = peaks[keep];
            properties.peak_heights = peak_heights[keep];
        }

        if (!threshold.empty())
        {
            // Evaluate threshold condition
            auto [tmin, tmax] = unpack_condition_args(threshold);
            auto [keep, left_thresholds, right_thresholds] = select_by_peak_threshold(x, peaks, tmin, tmax);
            peaks = peaks[keep];
            properties.left_thresholds = left_thresholds[keep];
            properties.right_thresholds = right_thresholds[keep];
        }

        if (!isnan(distance))
        {
            auto keep = select_by_peak_distance(peaks, x[peaks], distance);
            peaks = peaks[keep];
            properties.foreach([keep](auto &elem)
            {
                elem = elem[keep];
            });
        }

        if (!prominence.empty() && !width.empty())
        {
            // Calculate prominence (required for both conditions)
            wlen = arg_wlen_as_expected(wlen);
            auto [new_prominences, new_left_bases, new_right_bases] = peak_prominences(x, peaks, wlen);
            properties.prominences = new_prominences;
            properties.left_bases = new_left_bases;
            properties.right_bases = new_right_bases;
        }

        if (!prominence.empty())
        {
            // Evaluate prominence condition
            auto [pmin, pmax] = unpack_condition_args(prominence);
            auto keep = select_by_property(properties.prominences, pmin, pmax);
            peaks = peaks[keep];
            properties.foreach([keep](auto &elem)
            {
                elem = elem[keep];
            });
        }

        if (!width.empty())
        {
            // Calculate widths
            auto [new_widths, new_width_heights, new_left_ips, new_right_ips] = peak_widths(
                x, peaks, rel_height, properties.prominences, properties.left_bases, properties.right_bases);
            properties.widths = new_widths;
            properties.width_heights = new_width_heights;
            properties.left_ips = new_left_ips;
            properties.right_ips = new_right_ips;
            // Evaluate width condition
            auto [wmin, wmax] = unpack_condition_args(width);
            auto keep = select_by_property(properties.widths, wmin, wmax);
            peaks = peaks[keep];
            properties.foreach([keep](auto &elem)
            {
                elem = elem[keep];
            });
        }

        return {peaks, properties};
    }
}

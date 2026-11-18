#pragma once

#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include "miniz/miniz.h"
#include "vendor/json/nlohmann_json.hpp"

namespace Experiment
{
    class Archive
    {
    public:
        Archive(const std::filesystem::path &directory, const std::string &kind);
        ~Archive();
        Archive(const Archive &) = delete;
        Archive &operator=(const Archive &) = delete;

        nlohmann::json &Parameters() { return m_Manifest["parameters"]; }
        nlohmann::json &Results() { return m_Manifest["results"]; }
        void AddPlot(nlohmann::json plot) { m_Manifest["plots"].push_back(std::move(plot)); }

        nlohmann::json AddSeries(const std::string &name, std::span<const double> values, const std::string &axis)
        {
            const std::array shape{static_cast<Eigen::Index>(values.size())};
            return AddDoubleArray(name, values, shape, {axis});
        }

        nlohmann::json AddDoubleArray(const std::string &name, std::span<const double> values,
                                     std::span<const Eigen::Index> shape, std::initializer_list<std::string> axes);

        template <typename Derived>
        nlohmann::json AddArray(const std::string &name, const Eigen::DenseBase<Derived> &array,
                                std::initializer_list<std::string> axes)
        {
            const auto &values = array.eval();
            const bool vector = axes.size() == 1;
            if ((!vector && axes.size() != 2) || (vector && values.rows() != 1 && values.cols() != 1))
            {
                throw std::invalid_argument("NPY array rank does not match its axes");
            }
            using Scalar = typename Derived::Scalar;
            static_assert(std::endian::native == std::endian::little);
            static_assert(std::is_arithmetic_v<Scalar> && !std::is_same_v<Scalar, bool>);
            const char kind = std::is_floating_point_v<Scalar> ? 'f' : (std::is_signed_v<Scalar> ? 'i' : 'u');
            const std::string dtype = std::string(sizeof(Scalar) == 1 ? "|" : "<") + kind
                + std::to_string(sizeof(Scalar));
            const std::array shape{vector ? values.size() : values.rows(), values.cols()};
            const auto data = std::as_bytes(std::span(values.data(), static_cast<std::size_t>(values.size())));
            AddArrayBytes(name, axes, std::span(shape).first(axes.size()), dtype,
                          !vector && !Derived::PlainObject::IsRowMajor, data);
            return {{"$array", name}};
        }

        std::filesystem::path Publish();

    private:
        void AddArrayBytes(const std::string &name, std::initializer_list<std::string> axes,
                           std::span<const Eigen::Index> shape, const std::string &dtype,
                           bool fortran_order, std::span<const std::byte> data);
        void AddEntry(const std::string &path, std::string_view prefix, std::span<const std::byte> data);
        void Cleanup() noexcept;

        std::filesystem::path m_Path;
        std::filesystem::path m_WorkDirectory;
        std::ofstream m_File;
        mz_zip_archive m_Zip{};
        nlohmann::json m_Manifest;
    };
}

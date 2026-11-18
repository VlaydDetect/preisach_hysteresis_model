#include "ExperimentArchive.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <random>
#include <system_error>

#include <Windows.h>

namespace Experiment
{
    namespace
    {
        bool IsIdentifier(const std::string &name)
        {
            return !name.empty() && std::ranges::all_of(name, [](char ch)
            {
                return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_';
            });
        }

        struct EntrySource
        {
            std::string_view prefix;
            std::span<const std::byte> data;
        };

        size_t ReadEntry(void *opaque, mz_uint64 offset, void *output, size_t count)
        {
            const auto &entry = *static_cast<EntrySource *>(opaque);
            auto *destination = static_cast<std::byte *>(output);
            size_t copied = 0;
            if (offset < entry.prefix.size())
            {
                copied = std::min(count, entry.prefix.size() - static_cast<size_t>(offset));
                std::memcpy(destination, entry.prefix.data() + offset, copied);
                offset += copied;
            }
            if (copied < count && offset - entry.prefix.size() < entry.data.size())
            {
                const auto data_offset = static_cast<size_t>(offset - entry.prefix.size());
                const auto part = std::min(count - copied, entry.data.size() - data_offset);
                std::memcpy(destination + copied, entry.data.data() + data_offset, part);
                copied += part;
            }
            return copied;
        }

        size_t WriteZip(void *opaque, mz_uint64 offset, const void *buffer, size_t count)
        {
            auto &file = *static_cast<std::ofstream *>(opaque);
            file.seekp(static_cast<std::streamoff>(offset));
            file.write(static_cast<const char *>(buffer), static_cast<std::streamsize>(count));
            return file ? count : 0;
        }

        std::string NpyHeader(std::span<const Eigen::Index> shape, std::string_view dtype, bool fortran_order)
        {
            std::string shape_text = "(";
            for (const auto dimension : shape)
            {
                if (shape_text.size() != 1) shape_text += ", ";
                shape_text += std::to_string(dimension);
            }
            shape_text += shape.size() == 1 ? ",)" : ")";
            std::string header = std::format(
                "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
                dtype, fortran_order ? "True" : "False", shape_text);
            constexpr size_t fixed_size = 10;
            const auto padding = (64 - (fixed_size + header.size() + 1) % 64) % 64;
            header.append(padding, ' ');
            header.push_back('\n');
            if (header.size() > 10'000)
            {
                throw std::invalid_argument("NPY header exceeds the archive reader's limit");
            }
            std::string prefix = "\x93NUMPY\x01";
            prefix.push_back('\0');
            prefix.push_back(static_cast<char>(header.size() & 0xff));
            prefix.push_back(static_cast<char>(header.size() >> 8));
            return prefix + header;
        }
    }

    Archive::Archive(const std::filesystem::path &directory, const std::string &kind)
        : m_Manifest{{"schema_version", 1}, {"experiment_kind", kind}, {"parameters", nlohmann::json::object()},
                     {"arrays", nlohmann::json::object()}, {"plots", nlohmann::json::array()}}
    {
        if (!IsIdentifier(kind))
        {
            throw std::invalid_argument("Invalid experiment identifier");
        }
        std::filesystem::create_directories(directory);
        std::random_device random;
        const auto stem = std::format("{}-{:08x}{:08x}", kind, random(), random());
        m_Path = directory / (stem + ".zip");
        m_WorkDirectory = directory / (stem + ".part");
        if (!std::filesystem::create_directory(m_WorkDirectory))
        {
            throw std::runtime_error("Temporary archive path already exists");
        }
        m_File.open(m_WorkDirectory / "archive.zip", std::ios::binary);
        m_Zip.m_pWrite = WriteZip;
        m_Zip.m_pIO_opaque = &m_File;
        if (!m_File || !mz_zip_writer_init(&m_Zip, 0))
        {
            Cleanup();
            throw std::runtime_error("Could not open the temporary experiment archive");
        }
    }

    Archive::~Archive()
    {
        Cleanup();
    }

    void Archive::Cleanup() noexcept
    {
        if (m_Zip.m_pState)
        {
            mz_zip_writer_end(&m_Zip);
        }
        if (m_File.is_open())
        {
            m_File.close();
        }
        std::error_code error;
        std::filesystem::remove(m_WorkDirectory / "archive.zip", error);
        std::filesystem::remove(m_WorkDirectory, error);
    }

    void Archive::AddEntry(const std::string &path, std::string_view prefix, std::span<const std::byte> data)
    {
        EntrySource source{prefix, data};
        if (!m_Zip.m_pState || !mz_zip_writer_add_read_buf_callback(
                &m_Zip, path.c_str(), ReadEntry, &source, prefix.size() + data.size(), nullptr, nullptr,
                0, MZ_NO_COMPRESSION, nullptr, 0, nullptr, 0))
        {
            Cleanup();
            throw std::runtime_error("Could not write experiment archive entry: " + path);
        }
    }

    nlohmann::json Archive::AddDoubleArray(const std::string &name, std::span<const double> values,
                                           std::span<const Eigen::Index> shape,
                                           std::initializer_list<std::string> axes)
    {
        static_assert(sizeof(double) == 8 && std::endian::native == std::endian::little);
        if (shape.empty() || shape.size() != axes.size())
        {
            throw std::invalid_argument("NPY array rank does not match its axes");
        }
        std::size_t elements = 1;
        for (const auto extent : shape)
        {
            if (extent < 0 || (extent != 0 && elements > std::numeric_limits<std::size_t>::max() /
                                                               static_cast<std::size_t>(extent)))
            {
                throw std::invalid_argument("Invalid NPY array extent");
            }
            elements *= static_cast<std::size_t>(extent);
        }
        if (elements != values.size())
        {
            throw std::invalid_argument("NPY array size does not match its shape");
        }
        AddArrayBytes(name, axes, shape, "<f8", false, std::as_bytes(values));
        return {{"$array", name}};
    }

    void Archive::AddArrayBytes(const std::string &name, std::initializer_list<std::string> axes,
                                std::span<const Eigen::Index> shape, const std::string &dtype,
                                bool fortran_order, std::span<const std::byte> data)
    {
        auto &arrays = m_Manifest.at("arrays");
        if (!IsIdentifier(name) || arrays.contains(name))
        {
            throw std::invalid_argument("Invalid or duplicate array identifier: " + name);
        }
        AddEntry("arrays/" + name + ".npy", NpyHeader(shape, dtype, fortran_order), data);
        arrays[name] = {{"axes", axes}, {"shape", shape}, {"dtype", dtype}};
    }

    std::filesystem::path Archive::Publish()
    {
        const auto manifest = m_Manifest.dump();
        AddEntry("manifest.json", manifest, {});
        if (!mz_zip_writer_finalize_archive(&m_Zip) || !mz_zip_writer_end(&m_Zip))
        {
            throw std::runtime_error("Could not finish experiment archive");
        }
        m_File.close();
        if (!m_File)
        {
            throw std::runtime_error("Could not close experiment archive");
        }
        const auto source = m_WorkDirectory / "archive.zip";
        if (!MoveFileW(source.c_str(), m_Path.c_str()))
        {
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Could not publish experiment archive");
        }
        return m_Path;
    }
}

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>

namespace Net
{
    void SendArchive(const std::filesystem::path &path, std::uint16_t port = 8888);

    namespace detail
    {
        template <typename Send>
        void SendAll(std::span<const char> bytes, Send send)
        {
            while (!bytes.empty())
            {
                const auto sent = send(bytes);
                if (sent == 0 || sent > bytes.size())
                {
                    throw std::runtime_error("Socket stopped accepting archive data");
                }
                bytes = bytes.subspan(sent);
            }
        }
    }
}

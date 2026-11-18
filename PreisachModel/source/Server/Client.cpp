#include <WinSock2.h>

#include "Client.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <system_error>

namespace Net
{
    namespace
    {
        [[noreturn]] void SocketError(const char *operation, int code = WSAGetLastError())
        {
            throw std::system_error(code, std::system_category(), operation);
        }

        class Connection
        {
        public:
            explicit Connection(std::uint16_t port)
            {
                WSADATA data;
                if (const int error = WSAStartup(MAKEWORD(2, 2), &data); error != 0)
                {
                    SocketError("WSAStartup", error);
                }
                try
                {
                    m_Socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                    if (m_Socket == INVALID_SOCKET)
                    {
                        SocketError("socket");
                    }
                    Connect(port);
                }
                catch (...)
                {
                    Close();
                    throw;
                }
            }

            ~Connection() { Close(); }
            Connection(const Connection &) = delete;
            Connection &operator=(const Connection &) = delete;

            void Send(std::span<const char> bytes) const
            {
                detail::SendAll(bytes, [&](std::span<const char> remaining)
                {
                    const int sent = send(m_Socket, remaining.data(), static_cast<int>(remaining.size()), 0);
                    if (sent == SOCKET_ERROR)
                    {
                        SocketError("send archive");
                    }
                    return static_cast<std::size_t>(sent);
                });
            }

            void Confirm() const
            {
                if (shutdown(m_Socket, SD_SEND) == SOCKET_ERROR)
                {
                    SocketError("shutdown archive upload");
                }
                char status;
                const int count = recv(m_Socket, &status, 1, 0);
                if (count == SOCKET_ERROR)
                {
                    SocketError("receive archive acknowledgement");
                }
                if (count == 0 || status != '\0')
                {
                    throw std::runtime_error("Python server did not confirm the archive");
                }
            }

        private:
            void Connect(std::uint16_t port) const
            {
                sockaddr_in address{};
                address.sin_family = AF_INET;
                address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
                address.sin_port = htons(port);
                u_long mode = 1;
                if (ioctlsocket(m_Socket, FIONBIO, &mode) == SOCKET_ERROR)
                {
                    SocketError("configure socket");
                }
                if (connect(m_Socket, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == SOCKET_ERROR)
                {
                    const int error = WSAGetLastError();
                    if (error != WSAEWOULDBLOCK)
                    {
                        SocketError("connect to Python", error);
                    }
                    fd_set writable;
                    fd_set failed;
                    FD_ZERO(&writable);
                    FD_ZERO(&failed);
                    FD_SET(m_Socket, &writable);
                    FD_SET(m_Socket, &failed);
                    timeval timeout{5, 0};
                    const int ready = select(0, nullptr, &writable, &failed, &timeout);
                    if (ready == SOCKET_ERROR)
                    {
                        SocketError("wait for Python connection");
                    }
                    if (ready == 0)
                    {
                        SocketError("connect to Python", WSAETIMEDOUT);
                    }
                    int result = 0;
                    int result_size = sizeof(result);
                    if (getsockopt(m_Socket, SOL_SOCKET, SO_ERROR,
                                   reinterpret_cast<char *>(&result), &result_size) == SOCKET_ERROR)
                    {
                        SocketError("read connection status");
                    }
                    if (result != 0)
                    {
                        SocketError("connect to Python", result);
                    }
                }

                mode = 0;
                if (ioctlsocket(m_Socket, FIONBIO, &mode) == SOCKET_ERROR)
                {
                    SocketError("configure socket");
                }
                const DWORD timeout_ms = 15'000;
                const auto *timeout = reinterpret_cast<const char *>(&timeout_ms);
                if (setsockopt(m_Socket, SOL_SOCKET, SO_SNDTIMEO, timeout, sizeof(timeout_ms)) == SOCKET_ERROR ||
                    setsockopt(m_Socket, SOL_SOCKET, SO_RCVTIMEO, timeout, sizeof(timeout_ms)) == SOCKET_ERROR)
                {
                    SocketError("configure archive transfer timeout");
                }
            }

            void Close() const noexcept
            {
                if (m_Socket != INVALID_SOCKET)
                {
                    closesocket(m_Socket);
                }
                WSACleanup();
            }

            SOCKET m_Socket = INVALID_SOCKET;
        };
    }

    void SendArchive(const std::filesystem::path &path, std::uint16_t port)
    {
        std::ifstream archive(path, std::ios::binary);
        const std::uint64_t size = std::filesystem::file_size(path);
        if (!archive || size == 0)
        {
            throw std::runtime_error("Could not open the saved experiment archive");
        }

        Connection connection(port);
        std::array<char, 8> length;
        for (std::size_t i = 0; i < length.size(); ++i)
        {
            length[i] = static_cast<char>((size >> (8 * (length.size() - i - 1))) & 0xff);
        }
        connection.Send(length);

        std::array<char, 64 * 1024> buffer;
        for (std::uint64_t remaining = size; remaining != 0;)
        {
            const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
            if (!archive.read(buffer.data(), static_cast<std::streamsize>(count)))
            {
                throw std::runtime_error("Could not read the saved experiment archive");
            }
            connection.Send(std::span(buffer).first(count));
            remaining -= count;
        }
        connection.Confirm();
    }
}

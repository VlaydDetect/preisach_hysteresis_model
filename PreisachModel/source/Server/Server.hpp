#pragma once

#include <cassert>
#include <utility>
#include <WinSock2.h>

#include "MUTCpp/Json/JsonDocument.hpp"

// #pragma comment(lib, "Ws2_32.lib")
#pragma warning(disable: 4996)

namespace Net
{
    enum
    {
        BUFFER_SIZE = 1024
    };
    
    class Server
    {
    public:
        Server(uint16_t port, std::string ipAddress)
            : m_WSA{}, m_ServerSocket(INVALID_SOCKET), m_IPAddress(std::move(ipAddress)),
              m_Port(port), m_Info{}, m_InfoLength(sizeof(m_Info))
        {
        }
        ~Server()
        {
            WSACleanup();
            closesocket(m_ServerSocket);
        }

        void Connect()
        {
            Init();

            printf("Connection started.\n");
            const int connectionError = connect(m_ServerSocket, reinterpret_cast<sockaddr*>(&m_Info), m_InfoLength);
            assert(!connectionError && "Couldn't connect socket!!!");
            printf("Connection success!\n");
        }

        void Send(const std::string& message)
        {
            printf("Data Message Size: %llu", message.size());
            const int sended = sendto(m_ServerSocket, message.c_str(), static_cast<int>(message.size()), 0,
                                      reinterpret_cast<sockaddr*>(&m_Info), m_InfoLength);
            if (sended == SOCKET_ERROR)
            {
                printf("Send failed!!! Error: %d\n", WSAGetLastError());
                exit(EXIT_FAILURE);
            }
            
        }
        
        void SendDataMessage(const mc::json::JsonDocument& message)
        {
            std::string msg = message.ToString();
            Send(msg);
        }
        
        void Stop();

    private:
        void Init()
        {
            m_Info.sin_family = AF_INET;
            m_Info.sin_port = htons(m_Port);
            m_Info.sin_addr.s_addr = inet_addr(m_IPAddress.c_str());

            printf("WSA Init!\n");
            int startupError = WSAStartup(MAKEWORD(2, 2), &m_WSA);
            assert(!startupError && "Couldn't init WSA!!!");
            printf("WSA Success!\n");

            printf("Creating Socket.\n");
            // m_ServerSocket = socket(AF_INET, SOCK_DGRAM, 0);
            m_ServerSocket = socket(AF_INET, SOCK_STREAM, 0);
            assert((m_ServerSocket != SOCKET_ERROR) && "Couldn't create Socket!!!");
            printf("Socket created success!\n");
        }
        
        // void Receive()
        // {
        //     m_RecvLength = recvfrom(m_ServerSocket, m_Buffer, BUFFER_SIZE, 0, reinterpret_cast<sockaddr*>(&m_Info), &m_InfoLength);
        //     if (m_RecvLength == SOCKET_ERROR)
        //     {
        //         printf("Receive failed!!! Error: %d\n", WSAGetLastError());
        //         exit(EXIT_FAILURE);
        //     }
        // }
        // void Process()
        // {
        //     printf("Packet from: %s:%d\n", inet_ntoa(m_Info.sin_addr), ntohs(m_Info.sin_port));
        //
        //     for (int i = 0; i < m_RecvLength; i++)
        //     {
        //         printf("%c", m_Buffer[i]);
        //     }
        // }

    private:
        WSADATA m_WSA;
        SOCKET m_ServerSocket;
        std::string m_IPAddress;
        uint16_t m_Port;
        char m_Buffer[BUFFER_SIZE];
        std::string m_Message;
        sockaddr_in m_Info;
        int m_InfoLength;
        int m_RecvLength;
    };
}

#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>

#include "MUTCpp.hpp"

class FileWriter : public mc::RefCounted
{
public:
    FileWriter(const std::string& filePath)
    {
        m_FilePath = std::filesystem::absolute(filePath);
        std::filesystem::create_directories(m_FilePath.parent_path());
        m_File = std::ofstream(m_FilePath);
        std::cout << m_FilePath;
    }

    virtual ~FileWriter() override
    {
        if (m_File.is_open())
        {
            m_File.close();
        }
    }

    void Write(const std::vector<double>& value)
    {
        // m_File.open(m_FilePath);
        if (m_File.is_open())
        {
            for (const auto v : value)
            {
                m_File << v << '\n';
            }

            m_File.close();
        }
    }

    void Write(const std::string& text)
    {
        if (m_File.is_open())
        {
            m_File << text;
            m_File.close();
        }
        else
        {
            THROW_RUNTIME_ERROR(std::format("File is not open: {}", m_FilePath.string()));
        }
    }

private:
    std::filesystem::path m_FilePath;
    std::ofstream m_File;
};

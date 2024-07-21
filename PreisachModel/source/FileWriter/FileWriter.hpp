#pragma once
#include <string>
#include <fstream>
#include <vector>

class FileWriter
{
public:
    FileWriter(std::string filePath) : m_FilePath(std::move(filePath))
    {
    }

    ~FileWriter()
    {
        if (m_File.is_open())
        {
            m_File.close();
        }
    }

    void Write(const std::vector<double>& value)
    {
        m_File.open(m_FilePath);
        for (const auto v : value)
        {
            m_File << v << '\n';
        }

        m_File.close();
    }

private:
    std::string m_FilePath;
    std::ofstream m_File;
};

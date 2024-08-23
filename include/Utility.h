#pragma once

#include <Windows.h>
#include <commdlg.h>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>

namespace utils
{
    inline std::wstring SelectFile(const wchar_t *filter)
    {
        OPENFILENAMEW ofn;         // 公共对话框结构
        wchar_t szFile[260] = {0}; // 存储文件名的缓冲区

        // 初始化OPENFILENAME结构
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL; // 如果你有窗口句柄，可以在这里设置
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
        ofn.lpstrFilter = filter;
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = NULL;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = NULL;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

        // 显示打开对话框
        if (GetOpenFileNameW(&ofn))
        {
            return ofn.lpstrFile;
        }
        return L"";
    }

    inline std::wstring ConvertToWString(const std::string &str)
    {
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
        return wstrTo;
    }

    inline void LogToFile(const std::string &message)
    {
        // 获取当前日期
        time_t now = time(0);
        tm *localTime = localtime(&now);

        char fileName[100];
        strftime(fileName, sizeof(fileName), "vrlog-%Y-%m-%d.txt", localTime);

        // 打开文件
        static bool isFirstLog = true;
        std::ofstream logFile;

        if (isFirstLog)
        {
            // 第一次记录时清空文件
            logFile.open(fileName, std::ios::out | std::ios::trunc);
            isFirstLog = false;
        }
        else
        {
            // 追加写入
            logFile.open(fileName, std::ios::out | std::ios::app);
        }

        if (logFile.is_open())
        {
            logFile << message << std::endl;
            logFile.close();
        }
        else
        {
            std::cerr << "Unable to open log file: " << fileName << std::endl;
        }
    }
}

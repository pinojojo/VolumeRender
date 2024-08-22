#include <iostream>
#include <string>
#include <vector>

#include "VolumeRenderAPI.h"

int main()
{
    std::string command;
    while (true)
    {
        std::cout << "Enter command (create, list, render, exit): ";
        std::cin >> command;

        if (command == "create")
        {
            auto *instancePtr = Init();

            if (instancePtr == nullptr)
            {
                std::cerr << "Failed to create instance" << std::endl;
            }
            else
            {
                std::cout << "Instance created, the address is " << instancePtr << std::endl;
            }
        }
        else if (command == "list")
        {
            List();
        }
        else if (command == "render")
        {
            int index = 0;
            int width = 800;
            int height = 600;

            std::string arg;
            while (std::cin >> arg)
            {
                if (arg == "--i")
                {
                    std::cin >> index;
                }
                else if (arg == "--width")
                {
                    std::cin >> width;
                }
                else if (arg == "--height")
                {
                    std::cin >> height;
                }
                else
                {
                    break; // 停止读取参数
                }
            }

            // 执行渲染逻辑
            Render(width, height);
        }
        else if (command == "exit")
        {
            std::cout << "Exiting program." << std::endl;
            break;
        }
        else
        {
            std::cerr << "Unknown command: " << command << std::endl;
        }
    }

    return 0;
}

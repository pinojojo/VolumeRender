#pragma once

extern "C"
{
    /**
     * @brief 初始化一个VolumeRender实例
     *
     */
    __declspec(dllexport) void *Init();

    __declspec(dllexport) int Test();

    /**
     * @brief 获取VolumeRender实例的状态,以json的dump格式返回
     *
     */
    __declspec(dllexport) const char *Status();

    __declspec(dllexport) void List();

    __declspec(dllexport) int Render(int w, int h);

    __declspec(dllexport) int Cleanup(void *instance);
}

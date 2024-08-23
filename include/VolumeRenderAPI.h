#pragma once

extern "C"
{

    enum ErrorCode
    {
        VOLUME_RENDER_SUCCESS = 0,
        VOLUME_RENDER_INVALID_INSTANCE = 1,
        VOLUME_RENDER_INVALID_PARAMETER = 2,
        VOLUME_RENDER_INSTANCE_NOT_FOUND = 3,
        VOLUME_RENDER_RENDER_FAILED = 4,
        // 其他错误码...
    };

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

    /**
     * @brief
     *
     */
    __declspec(dllexport) void Retrieve(void *buffer, int width, int height);

    __declspec(dllexport) void List();

    __declspec(dllexport) int Render(int w, int h);

    __declspec(dllexport) int Cleanup(void *instance);
}

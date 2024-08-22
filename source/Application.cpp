/*
 * MIT License
 *
 * Copyright(c) 2022 Mikhail Gorobets
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this softwareand associated documentation files(the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright noticeand this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "Application.h"

#include <glfw/glfw3.h>
#include <glfw/glfw3native.h>
#include <imgui/imgui_impl_dx11.h>
#include <imgui/imgui_impl_glfw.h>
#include <implot/implot.h>
#include <fmt/printf.h>

#include <wincodec.h> // WIC的头文件，用于保存图片

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

Application::Application(ApplicationDesc const &desc)
    : m_ApplicationDesc(desc)
{

    this->InitializeSDL();
    this->InitializeD3D11();
    this->InitializeImGUI();
}

Application::~Application()
{

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_pWindow);
    glfwTerminate();
}

void Application::Resize(int32_t width, int32_t height)
{
    if (width == 0 || width == 0)
        return;

    // 解除绑定任何RTV和DSV
    m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);

    // 释放RTV关联资源
    for (uint32_t frameID = 0; frameID < FrameCount; frameID++)
    {
        m_pRTV[frameID].Reset();
        m_pD3D11BackBuffers[frameID].Reset();
        m_pD3D11BackBuffersDummy[frameID].Reset();
        m_pD3D12BackBuffers[frameID].Reset();
    }

    // 释放DSV关联资源
    m_pDSV.Reset();
    m_pDepthStencil.Reset();

    m_pImmediateContext->Flush();
    this->WaitForGPU();

    // 调整交换链对应的buffer的大小
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    DX::ThrowIfFailed(m_pSwapChain->GetDesc1(&swapChainDesc));
    DX::ThrowIfFailed(m_pSwapChain->ResizeBuffers(FrameCount, width, height, swapChainDesc.Format, swapChainDesc.BufferUsage));

    // 重新根据调整过大小的buffer来创建RTV
    for (uint32_t frameID = 0; frameID < FrameCount; frameID++)
    {
        DX::ThrowIfFailed(m_pSwapChain->GetBuffer(frameID, IID_PPV_ARGS(&m_pD3D12BackBuffers[frameID])));

        D3D11_RESOURCE_FLAGS d3d11Flags = {D3D11_BIND_RENDER_TARGET};
        DX::ThrowIfFailed(m_pD3D11On12Device->CreateWrappedResource(m_pD3D12BackBuffers[frameID].Get(), &d3d11Flags, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET, IID_PPV_ARGS(&m_pD3D11BackBuffersDummy[frameID])));
        DX::ThrowIfFailed(m_pD3D11On12Device->CreateWrappedResource(m_pD3D12BackBuffers[frameID].Get(), &d3d11Flags, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT, IID_PPV_ARGS(&m_pD3D11BackBuffers[frameID])));

        D3D11_RENDER_TARGET_VIEW_DESC descRTV = {};
        descRTV.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        descRTV.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        DX::ThrowIfFailed(m_pDevice->CreateRenderTargetView(m_pD3D11BackBuffers[frameID].Get(), &descRTV, m_pRTV[frameID].GetAddressOf()));
    }

    // 重建深度缓冲纹理
    D3D11_TEXTURE2D_DESC descDepth = {};
    descDepth.Width = width;
    descDepth.Height = height;
    descDepth.MipLevels = 1;
    descDepth.ArraySize = 1;
    descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    descDepth.SampleDesc.Count = 1;
    descDepth.SampleDesc.Quality = 0;
    descDepth.Usage = D3D11_USAGE_DEFAULT;
    descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    descDepth.CPUAccessFlags = 0;
    descDepth.MiscFlags = 0;

    DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&descDepth, nullptr, m_pDepthStencil.GetAddressOf()));

    // 重建DSV
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {};
    descDSV.Format = descDepth.Format;
    descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0;
    DX::ThrowIfFailed(m_pDevice->CreateDepthStencilView(m_pDepthStencil.Get(), &descDSV, m_pDSV.GetAddressOf()));

    // 更新窗口大小
    m_ApplicationDesc.Width = width;
    m_ApplicationDesc.Height = height;
}

void SaveRenderTargetToPNG(ID3D11Device *m_pDevice, ID3D11DeviceContext *m_pImmediateContext, ID3D11RenderTargetView *m_pRTV, const char *filename)
{
    // 获取当前的渲染目标视图
    ID3D11Texture2D *pRenderTarget = nullptr;
    m_pRTV->GetResource(reinterpret_cast<ID3D11Resource **>(&pRenderTarget));

    // 创建一个用于读取数据的临时纹理
    D3D11_TEXTURE2D_DESC desc;
    pRenderTarget->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    ID3D11Texture2D *pStagingTexture = nullptr;
    m_pDevice->CreateTexture2D(&desc, nullptr, &pStagingTexture);

    // 将渲染目标的内容复制到临时纹理
    m_pImmediateContext->CopyResource(pStagingTexture, pRenderTarget);

    // 映射临时纹理以读取数据
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    m_pImmediateContext->Map(pStagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);

    // 读取像素数据
    BYTE *pData = reinterpret_cast<BYTE *>(mappedResource.pData);
    UINT rowPitch = mappedResource.RowPitch;
    UINT imageSize = rowPitch * desc.Height;

    // 创建一个缓冲区来存储图像数据
    std::vector<BYTE> imageData(imageSize);
    memcpy(imageData.data(), pData, imageSize);

    // 解除映射
    m_pImmediateContext->Unmap(pStagingTexture, 0);

    // 保存为 PNG 图像
    stbi_write_png(filename, desc.Width, desc.Height, 4, imageData.data(), rowPitch);

    // 释放资源
    pStagingTexture->Release();
    pRenderTarget->Release();
}

void Application::Run()
{

    glfwSetWindowUserPointer(m_pWindow, this);
    glfwSetMouseButtonCallback(m_pWindow, GLFW_WindowCallbacks::MouseButtonCallback);
    glfwSetCursorPosCallback(m_pWindow, GLFW_WindowCallbacks::MouseMoveCallback);
    glfwSetScrollCallback(m_pWindow, GLFW_WindowCallbacks::MouseScrollCallback);
    glfwSetWindowSizeCallback(m_pWindow, GLFW_WindowCallbacks::ResizeWindowCallback);

    bool needToExit = false;

    while (!glfwWindowShouldClose(m_pWindow) && !needToExit)
    {
        glfwPollEvents();

        this->Update(this->CalculateFrameTime());

        ImGui_ImplDX11_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        const uint32_t frameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

        // 两个调用确保 dummy 资源在 Direct3D 11 和 Direct3D 12 之间正确同步
        m_pD3D11On12Device->AcquireWrappedResources(m_pD3D11BackBuffersDummy[frameIndex].GetAddressOf(), 1);
        m_pD3D11On12Device->ReleaseWrappedResources(m_pD3D11BackBuffersDummy[frameIndex].GetAddressOf(), 1);

        // 获取non-dummy资源, 用于渲染
        m_pD3D11On12Device->AcquireWrappedResources(m_pD3D11BackBuffers[frameIndex].GetAddressOf(), 1);
        this->RenderFrame(m_pRTV[frameIndex]);

#ifdef WINDOWLESS
        { // 保存到本地

            static uint32_t frameCounter = 0;
            std::cout << "frame" << frameCounter << std::endl;
            if (frameCounter == 30)
                SaveRenderTargetToPNG(m_pDevice.Get(),
                                      m_pImmediateContext.Get(),
                                      m_pRTV[frameIndex].Get(), "content/Output/Render.png");
            frameCounter++;

            if (frameCounter == 100)
                needToExit = true;
        }
#endif

        this->RenderGUI(m_pRTV[frameIndex]);
        ImGui::Render();

        m_pAnnotation->BeginEvent(L"Render pass: ImGui");
        m_pImmediateContext->OMSetRenderTargets(1, m_pRTV[frameIndex].GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);
        m_pAnnotation->EndEvent();
        m_pD3D11On12Device->ReleaseWrappedResources(m_pD3D11BackBuffers[frameIndex].GetAddressOf(), 1);
        m_pImmediateContext->Flush();

        m_pSwapChain->Present(m_ApplicationDesc.IsVSync ? 1 : 0, 0);
    }

    this->WaitForGPU();
}

// API测试用的
auto Application::Run(int loopCount) -> void
{
    int count = 0;
    while (count < loopCount)
    {
        std::cout << "Running " << count << std::endl;
        glfwPollEvents();

        this->Update(this->CalculateFrameTime());

        ImGui_ImplDX11_NewFrame();

        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();

        const uint32_t frameIndex = m_pSwapChain->GetCurrentBackBufferIndex();

        // 两个调用确保 dummy 资源在 Direct3D 11 和 Direct3D 12 之间正确同步
        m_pD3D11On12Device->AcquireWrappedResources(m_pD3D11BackBuffersDummy[frameIndex].GetAddressOf(), 1);
        m_pD3D11On12Device->ReleaseWrappedResources(m_pD3D11BackBuffersDummy[frameIndex].GetAddressOf(), 1);

        // 获取non-dummy资源, 用于渲染
        m_pD3D11On12Device->AcquireWrappedResources(m_pD3D11BackBuffers[frameIndex].GetAddressOf(), 1);
        this->RenderFrame(m_pRTV[frameIndex]);

        { // 保存到本地

            static uint32_t frameCounter = 0;
            std::string fileName = fmt::format("content/Render{}.png", frameCounter);

            SaveRenderTargetToPNG(m_pDevice.Get(),
                                  m_pImmediateContext.Get(),
                                  m_pRTV[frameIndex].Get(), fileName.c_str());
            frameCounter++;
        }

        this->RenderGUI(m_pRTV[frameIndex]);
        ImGui::Render();

        m_pAnnotation->BeginEvent(L"Render pass: ImGui");
        m_pImmediateContext->OMSetRenderTargets(1, m_pRTV[frameIndex].GetAddressOf(), nullptr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);
        m_pAnnotation->EndEvent();
        m_pD3D11On12Device->ReleaseWrappedResources(m_pD3D11BackBuffers[frameIndex].GetAddressOf(), 1);
        m_pImmediateContext->Flush();

        m_pSwapChain->Present(m_ApplicationDesc.IsVSync ? 1 : 0, 0);

        count++;
    }

    this->WaitForGPU();

    std::cout << "Loop count: " << loopCount << " finised" << std::endl;
}

auto Application::GetDesc() const -> ApplicationDesc
{
    return m_ApplicationDesc;
}

void Application::InitializeSDL()
{

    glfwInit();

#ifdef WINDOWLESS
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // 设置窗口属性为隐藏
#endif

    if (m_ApplicationDesc.IsOffscreen)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // 设置窗口属性为隐藏

    m_pWindow = glfwCreateWindow(m_ApplicationDesc.Width, m_ApplicationDesc.Height, m_ApplicationDesc.Tittle.c_str(), nullptr, nullptr);
}

void Application::InitializeD3D11()
{

    HWND hWindow = glfwGetWin32Window(m_pWindow);
    uint32_t dxgiFactoryFlags = 0;
    uint32_t d3d11DeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
    {
        DX::ComPtr<ID3D12Debug> pDebugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&pDebugController))))
        {
            pDebugController->EnableDebugLayer();
            dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            d3d11DeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
        }
    }
#endif

    // 寻找最佳的高性能D3D12硬件适配器(非软件模拟的)
    auto GetHardwareAdapter = [](DX::ComPtr<IDXGIFactory> pFactory) -> DX::ComPtr<IDXGIAdapter1>
    {
        DX::ComPtr<IDXGIAdapter1> pAdapter;
        DX::ComPtr<IDXGIFactory6> pFactoryNew;
        pFactory.As(&pFactoryNew);

        bool bFound = false;

        for (uint32_t adapterID = 0; DXGI_ERROR_NOT_FOUND != pFactoryNew->EnumAdapterByGpuPreference(adapterID, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&pAdapter)); adapterID++)
        {
            DXGI_ADAPTER_DESC1 desc = {};
            pAdapter->GetDesc1(&desc);

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;

            if (SUCCEEDED(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_1, __uuidof(ID3D12Device), nullptr)))
            {
                bFound = true;
                break;
            }
        }

        if (!bFound)
        {
            std::cout << "Not Found High Performance Adapter" << std::endl;
        }

        return pAdapter;
    };

    DX::ComPtr<IDXGIFactory4> pFactory;
    DX::ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&pFactory)));

    // 创建D3D12设备
    DX::ComPtr<IDXGIAdapter1> pAdapter = GetHardwareAdapter(pFactory);
    DX::ThrowIfFailed(D3D12CreateDevice(pAdapter.Get(), D3D_FEATURE_LEVEL_11_1, IID_PPV_ARGS(&m_pD3D12Device)));

#if defined(_DEBUG)
    DX::ComPtr<ID3D12InfoQueue> infoQueue;
    if (SUCCEEDED(m_pD3D12Device->QueryInterface(IID_PPV_ARGS(&infoQueue))))
    {

        D3D12_MESSAGE_SEVERITY severities[] = {
            D3D12_MESSAGE_SEVERITY_INFO,
        };

        D3D12_MESSAGE_ID denyIds[] = {
            D3D12_MESSAGE_ID_INVALID_DESCRIPTOR_HANDLE};

        D3D12_INFO_QUEUE_FILTER filter = {};
        filter.DenyList.NumSeverities = _countof(severities);
        filter.DenyList.pSeverityList = severities;
        filter.DenyList.NumIDs = _countof(denyIds);
        filter.DenyList.pIDList = denyIds;
        DX::ThrowIfFailed(infoQueue->PushStorageFilter(&filter));
    }
#endif

    { // 创建命令队列
        D3D12_COMMAND_QUEUE_DESC desc = {};
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        DX::ThrowIfFailed(m_pD3D12Device->CreateCommandQueue(&desc, IID_PPV_ARGS(&m_pD3D12CmdQueue)));
    }

    { // 配置交换链
        DXGI_SWAP_CHAIN_DESC1 desc = {};
        desc.BufferCount = FrameCount; // 三重缓冲
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.SampleDesc.Count = 1;

        DX::ComPtr<IDXGISwapChain1> pSwapChain;
        DX::ThrowIfFailed(pFactory->CreateSwapChainForHwnd(m_pD3D12CmdQueue.Get(), hWindow, &desc, nullptr, nullptr, &pSwapChain));
        DX::ThrowIfFailed(pFactory->MakeWindowAssociation(hWindow, DXGI_MWA_NO_ALT_ENTER)); // 关联到窗口
        DX::ThrowIfFailed(pSwapChain.As(&m_pSwapChain));
    }

    { // 创建RTV

        // 创建D3D11设备和上下文
        DX::ThrowIfFailed(D3D11On12CreateDevice(m_pD3D12Device.Get(), d3d11DeviceFlags, nullptr, 0, reinterpret_cast<IUnknown **>(m_pD3D12CmdQueue.GetAddressOf()), 1, 0, &m_pDevice, &m_pImmediateContext, nullptr));
        DX::ThrowIfFailed(m_pDevice.As(&m_pD3D11On12Device));
        DX::ThrowIfFailed(m_pImmediateContext.As(&m_pAnnotation));

        // 创建RTV，总共三个
        for (uint32_t frameID = 0; frameID < FrameCount; frameID++)
        {
            // 从交换链获取帧缓冲，并存放于后台缓冲区指针数组中
            DX::ThrowIfFailed(m_pSwapChain->GetBuffer(frameID, IID_PPV_ARGS(&m_pD3D12BackBuffers[frameID])));

            // 包装为D3D11资源形式的后台缓冲区
            D3D11_RESOURCE_FLAGS d3d11Flags = {D3D11_BIND_RENDER_TARGET};
            DX::ThrowIfFailed(m_pD3D11On12Device->CreateWrappedResource(m_pD3D12BackBuffers[frameID].Get(), &d3d11Flags, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET, IID_PPV_ARGS(&m_pD3D11BackBuffersDummy[frameID])));
            DX::ThrowIfFailed(m_pD3D11On12Device->CreateWrappedResource(m_pD3D12BackBuffers[frameID].Get(), &d3d11Flags, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT, IID_PPV_ARGS(&m_pD3D11BackBuffers[frameID])));

            // 由后台缓冲区创建RTV
            D3D11_RENDER_TARGET_VIEW_DESC descRTV = {};
            descRTV.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            descRTV.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            DX::ThrowIfFailed(m_pDevice->CreateRenderTargetView(m_pD3D11BackBuffers[frameID].Get(), &descRTV, m_pRTV[frameID].GetAddressOf()));
        }
    }

    { // 创建DSV

        // 创建深度模板缓冲区
        D3D11_TEXTURE2D_DESC descDepth = {};
        descDepth.Width = m_ApplicationDesc.Width;
        descDepth.Height = m_ApplicationDesc.Height;
        descDepth.MipLevels = 1;
        descDepth.ArraySize = 1;
        descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        descDepth.SampleDesc.Count = 1;
        descDepth.SampleDesc.Quality = 0;
        descDepth.Usage = D3D11_USAGE_DEFAULT;
        descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        descDepth.CPUAccessFlags = 0;
        descDepth.MiscFlags = 0;

        // 创建深度模板缓冲区纹理
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&descDepth, nullptr, m_pDepthStencil.GetAddressOf()));

        // 创建DSV
        D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {};
        descDSV.Format = descDepth.Format;
        descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        descDSV.Texture2D.MipSlice = 0;
        DX::ThrowIfFailed(m_pDevice->CreateDepthStencilView(m_pDepthStencil.Get(), &descDSV, m_pDSV.GetAddressOf()));
    }

    // 创建Fence，用于同步CPU和GPU
    DX::ThrowIfFailed(m_pD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(m_pD3D12Fence.GetAddressOf())));
    m_FenceEvent = CreateEvent(nullptr, false, false, nullptr);
    if (m_FenceEvent == nullptr)
        DX::ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));

    std::cout << "D3D11 Device Created" << std::endl;
}

void Application::InitializeImGUI()
{

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    const auto &io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("content/Fonts/Roboto-Medium.ttf", 16.0f);
    ImPlot::GetStyle().AntiAliasedLines = true;
    ImGui::StyleColorsDark();

    ImGui_ImplDX11_Init(m_pDevice.Get(), m_pImmediateContext.Get());
    ImGui_ImplGlfw_InitForOther(m_pWindow, false);
}

float Application::CalculateFrameTime()
{

    const auto nowTime = std::chrono::high_resolution_clock::now();
    const auto delta = std::chrono::duration_cast<std::chrono::duration<float>>(nowTime - m_LastFrame).count();
    m_LastFrame = nowTime;
    return delta;
}

void Application::WaitForGPU()
{

    m_FenceValue++;
    DX::ThrowIfFailed(m_pD3D12CmdQueue->Signal(m_pD3D12Fence.Get(), m_FenceValue));
    DX::ThrowIfFailed(m_pD3D12Fence->SetEventOnCompletion(m_FenceValue, m_FenceEvent));
    WaitForSingleObjectEx(m_FenceEvent, INFINITE, false);
}

void GLFW_WindowCallbacks::MouseButtonCallback(GLFWwindow *pWindow, int32_t button, int32_t action, int32_t modes)
{

    const auto pApplication = static_cast<Application *>(glfwGetWindowUserPointer(pWindow));
    const auto pState = &pApplication->m_GLFWState;

    auto MouseButtonCall = [&](GLFWWindowState::MouseButton button, int32_t action) -> void
    {
        switch (action)
        {
        case GLFW_PRESS:
            pState->MouseState[button] = GLFWWindowState::MousePress;
            break;
        case GLFW_RELEASE:
            pState->MouseState[button] = GLFWWindowState::MouseRelease;
            break;
        default:
            break;
        }
    };

    switch (button)
    {
    case GLFW_MOUSE_BUTTON_RIGHT:
        MouseButtonCall(GLFWWindowState::MouseButtonRight, action);
        break;
    case GLFW_MOUSE_BUTTON_LEFT:
        MouseButtonCall(GLFWWindowState::MouseButtonLeft, action);
        break;
    default:
        break;
    }
}

void GLFW_WindowCallbacks::MouseMoveCallback(GLFWwindow *pWindow, double mousePosX, double mousePosY)
{

    const auto pApplication = static_cast<Application *>(glfwGetWindowUserPointer(pWindow));
    const auto pState = &pApplication->m_GLFWState;

    if (pState->MouseState[GLFWWindowState::MouseButtonRight] == GLFWWindowState::MousePress)
    {
        const double dX = mousePosX - pState->PreviousMousePositionX;
        const double dY = mousePosY - pState->PreviousMousePositionY;
        pApplication->EventMouseMove(static_cast<float>(dX), static_cast<float>(dY));
    }

    pState->PreviousMousePositionX = mousePosX;
    pState->PreviousMousePositionY = mousePosY;
}

void GLFW_WindowCallbacks::MouseScrollCallback(GLFWwindow *pWindow, double offsetX, double offsetY)
{

    const auto pApplication = static_cast<Application *>(glfwGetWindowUserPointer(pWindow));
    pApplication->EventMouseWheel(static_cast<float>(offsetY));
}

void GLFW_WindowCallbacks::ResizeWindowCallback(GLFWwindow *pWindow, int32_t width, int32_t height)
{

    const auto pApplication = static_cast<Application *>(glfwGetWindowUserPointer(pWindow));
    pApplication->Resize(width, height);
}

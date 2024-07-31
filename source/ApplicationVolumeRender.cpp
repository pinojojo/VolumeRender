/*
 * MIT License
 *
 * Copyright(c) 2021-2023 Mikhail Gorobets
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

#include "ApplicationVolumeRender.h"
#include <directx-tex/DDSTextureLoader.h>
#include <imgui/imgui.h>
#include <implot/implot.h>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <d3dcompiler.h>
#include <fstream>
#include <iostream>
#include <random>
#include "Utility.h"

#include <locale>
#include <codecvt>

#include <opencv2/opencv.hpp>

// 将数值对齐到最近的2的幂次方
unsigned int AlignToNearestPowerOf2(unsigned int value)
{
    if (value == 0)
        return 1;

    // 如果已经是2的幂次方，则直接返回
    if ((value & (value - 1)) == 0)
        return value;

    // 找到大于等于该值的最小的2的幂次方
    unsigned int powerOf2 = 1;
    while (powerOf2 < value)
    {
        powerOf2 <<= 1;
    }

    // 找到小于该值的最大的2的幂次方
    unsigned int lowerPowerOf2 = powerOf2 >> 1;

    // 返回更接近的那个
    return (value - lowerPowerOf2) < (powerOf2 - value) ? lowerPowerOf2 : powerOf2;
}

bool LoadVolumeDataFromTiff(const std::string &tiff_path,
                            std::vector<uint16_t> &intensity,
                            int &width, int &height, int &depth)
{
    std::vector<cv::Mat> layers;

    // 使用 imreadmulti 读取多层图像
    if (!cv::imreadmulti(tiff_path, layers, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR))
    {
        std::cout << "Successfully loaded " << layers.size() << " layers." << std::endl;
        return false;
    }

    width = layers[0].cols;
    height = layers[0].rows;
    depth = layers.size();

    if (width <= 0 || height <= 0 || depth <= 0)
    {
        std::cout << "Invalid volume data." << std::endl;
        return false;
    }

    if (layers[0].depth() != CV_16U && layers[0].depth() != CV_8U)
    {
        std::cout << "Invalid volume data format, depth must be 8 or 16 bits." << std::endl;
        return false;
    }

    // 先保证每个图片的宽高都是2的N次方，如果不是需要，补齐
    for (int z = 0; z < depth; z++)
    {
        int new_width = AlignToNearestPowerOf2(width);
        int new_height = AlignToNearestPowerOf2(height);

        if (new_width != width || new_height != height)
        {
            cv::Mat layer(new_height, new_width, layers[z].type(), cv::Scalar(0));
            layers[z].copyTo(layer(cv::Rect(0, 0, width, height)));
            layers[z] = layer;
        }

        width = new_width;
        height = new_height;
    }

    // 先转为16bit
    for (int z = 0; z < depth; z++)
    {
        if (layers[z].depth() != CV_16U)
        {
            cv::Mat layer;
            layers[z].convertTo(layer, CV_16U);
            layers[z] = layer;
        }
    }

    if (intensity.size() != width * height * depth)
    {
        intensity.resize(width * height * depth);
    }

    for (int z = 0; z < depth; z++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                intensity[z * width * height + y * width + x] = layers[z].at<uint16_t>(y, x);
            }
        }
    }
}

std::vector<double> GetHistogram(const std::vector<uint16_t> &intensity)
{
    std::vector<double> histogram(65536, 0.0);

    for (auto value : intensity)
    {
        histogram[value]++;
    }

    return histogram;
}

void GetMinMaxOfHistogram(const std::vector<double> &histogram, uint16_t &min, uint16_t &max)
{
    min = 0;
    max = 65535;

    for (int i = 0; i < 65536; i++)
    {
        if (histogram[i] > 0)
        {
            min = i;
            break;
        }
    }

    for (int i = 65535; i >= 0; i--)
    {
        if (histogram[i] > 0)
        {
            max = i;
            break;
        }
    }
}

std::vector<double> GetCDF(const std::vector<double> &histogram)
{
    std::vector<double> cdf(65536, 0.0);

    cdf[0] = histogram[0];
    for (int i = 1; i < 65536; i++)
    {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    return cdf;
}

uint16_t EvaluateIntensityOfPercentile(const std::vector<double> &cdf, double percentile)
{
    auto total = cdf[65535];
    auto target = total * percentile;

    for (int i = 0; i < 65536; i++)
    {
        if (cdf[i] >= target)
        {
            return i;
        }
    }

    return 65535;
}

struct FrameBuffer
{

    Hawk::Math::Mat4x4 ProjectionMatrix;
    Hawk::Math::Mat4x4 ViewMatrix;
    Hawk::Math::Mat4x4 WorldMatrix;
    Hawk::Math::Mat4x4 NormalMatrix;

    Hawk::Math::Mat4x4 InvProjectionMatrix;
    Hawk::Math::Mat4x4 InvViewMatrix;
    Hawk::Math::Mat4x4 InvWorldMatrix;
    Hawk::Math::Mat4x4 InvNormalMatrix;

    Hawk::Math::Mat4x4 ViewProjectionMatrix;
    Hawk::Math::Mat4x4 NormalViewMatrix;
    Hawk::Math::Mat4x4 WorldViewProjectionMatrix;

    Hawk::Math::Mat4x4 InvViewProjectionMatrix;
    Hawk::Math::Mat4x4 InvNormalViewMatrix;
    Hawk::Math::Mat4x4 InvWorldViewProjectionMatrix;

    uint32_t FrameIndex;

    float StepSize;
    Hawk::Math::Vec2 FrameOffset;

    Hawk::Math::Vec2 InvRenderTargetDim;
    Hawk::Math::Vec2 RenderTargetDim;

    float Density;
    Hawk::Math::Vec3 BoundingBoxMin;

    float Exposure;
    Hawk::Math::Vec3 BoundingBoxMax;

    Hawk::Math::Vec4 GridLineInfo;
    Hawk::Math::Vec4 GridLineInfoStart;
    Hawk::Math::Vec4 GridLineInfoStep;
};

struct DispatchIndirectBuffer
{
    uint32_t ThreadGroupX;
    uint32_t ThreadGroupY;
    uint32_t ThreadGroupZ;
};

struct DrawInstancedIndirectBuffer
{
    uint32_t VertexCount;
    uint32_t InstanceCount;
    uint32_t VertexOffset;
    uint32_t InstanceOffset;
};

ApplicationVolumeRender::ApplicationVolumeRender(ApplicationDesc const &desc)
    : Application(desc), m_RandomGenerator(m_RandomDevice()), m_RandomDistribution(-0.5f, +0.5f)
{

    this->InitializeShaders();
    this->InitializeTransferFunction();
    this->InitializeSamplerStates();
    this->InitializeRenderTextures();
    this->InitializeTileBuffer();
    this->InitializeBuffers();
    this->InitializeVolumeTexture();
    this->InitializeEnvironmentMap();
}

void ApplicationVolumeRender::InitializeShaders()
{

    auto compileShader = [](auto fileName, auto entryPoint, auto target, auto macros) -> DX::ComPtr<ID3DBlob>
    {
        DX::ComPtr<ID3DBlob> pCodeBlob;
        DX::ComPtr<ID3DBlob> pErrorBlob;

        uint32_t flags = 0;
#if defined(_DEBUG)
        flags |= D3DCOMPILE_DEBUG;
#else
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

        if (FAILED(D3DCompileFromFile(fileName, macros, D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint, target, flags, 0, pCodeBlob.GetAddressOf(), pErrorBlob.GetAddressOf())))
            throw std::runtime_error(static_cast<const char *>(pErrorBlob->GetBufferPointer()));
        return pCodeBlob;
    };

    // TODO AMD 8x8x1 NV 8x4x1
    const auto threadSizeX = std::to_string(8);
    const auto threadSizeY = std::to_string(8);

    D3D_SHADER_MACRO macros[] = {
        {"THREAD_GROUP_SIZE_X", threadSizeX.c_str()},
        {"THREAD_GROUP_SIZE_Y", threadSizeY.c_str()},
        {nullptr, nullptr}};

    auto pBlobCSGeneratePrimaryRays = compileShader(L"content/Shaders/ComputePrimaryRays.hlsl", "GenerateRays", "cs_5_0", macros);
    auto pBlobCSComputeDiffuseLight = compileShader(L"content/Shaders/ComputeRadiance.hlsl", "ComputeRadiance", "cs_5_0", macros);
    auto pBlobCSAccumulate = compileShader(L"content/Shaders/Accumulation.hlsl", "Accumulate", "cs_5_0", macros);
    auto pBlobCSComputeTiles = compileShader(L"content/Shaders/ComputeTiles.hlsl", "ComputeTiles", "cs_5_0", macros);
    auto pBlobCSToneMap = compileShader(L"content/Shaders/ToneMap.hlsl", "ToneMap", "cs_5_0", macros);
    auto pBlobCSComputeGradient = compileShader(L"content/Shaders/ComputeGradient.hlsl", "ComputeGradient", "cs_5_0", macros);
    auto pBlobCSGenerateMipLevel = compileShader(L"content/Shaders/ComputeLevelOfDetail.hlsl", "GenerateMipLevel", "cs_5_0", macros);
    auto pBlobCSResetTiles = compileShader(L"content/Shaders/ComputeTiles.hlsl", "ResetTiles", "cs_5_0", macros);

    auto pBlobVSTextureBlit = compileShader(L"content/Shaders/TextureBlit.hlsl", "BlitVS", "vs_5_0", macros);
    auto pBlobPSTextureBlit = compileShader(L"content/Shaders/TextureBlit.hlsl", "BlitPS", "ps_5_0", macros);

    auto pBlobVSDebugTiles = compileShader(L"content/Shaders/DebugTiles.hlsl", "DebugTilesVS", "vs_5_0", macros);
    auto pBlobGSDebugTiles = compileShader(L"content/Shaders/DebugTiles.hlsl", "DebugTilesGS", "gs_5_0", macros);
    auto pBlobPSDegugTiles = compileShader(L"content/Shaders/DebugTiles.hlsl", "DebugTilesPS", "ps_5_0", macros);

    auto pBlobVSGridLine = compileShader(L"content/Shaders/RenderLine.hlsl", "RenderLineVS", "vs_5_0", macros);
    auto pBlobPSGridLine = compileShader(L"content/Shaders/RenderLine.hlsl", "RenderLinePS", "ps_5_0", macros);

    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSGeneratePrimaryRays->GetBufferPointer(), pBlobCSGeneratePrimaryRays->GetBufferSize(), nullptr, m_PSOGeneratePrimaryRays.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSComputeDiffuseLight->GetBufferPointer(), pBlobCSComputeDiffuseLight->GetBufferSize(), nullptr, m_PSOComputeDiffuseLight.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSAccumulate->GetBufferPointer(), pBlobCSAccumulate->GetBufferSize(), nullptr, m_PSOAccumulate.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSComputeTiles->GetBufferPointer(), pBlobCSComputeTiles->GetBufferSize(), nullptr, m_PSOComputeTiles.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSToneMap->GetBufferPointer(), pBlobCSToneMap->GetBufferSize(), nullptr, m_PSOToneMap.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSGenerateMipLevel->GetBufferPointer(), pBlobCSGenerateMipLevel->GetBufferSize(), nullptr, m_PSOGenerateMipLevel.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSComputeGradient->GetBufferPointer(), pBlobCSComputeGradient->GetBufferSize(), nullptr, m_PSOComputeGradient.pCS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateComputeShader(pBlobCSResetTiles->GetBufferPointer(), pBlobCSResetTiles->GetBufferSize(), nullptr, m_PSOResetTiles.pCS.ReleaseAndGetAddressOf()));

    DX::ThrowIfFailed(m_pDevice->CreateVertexShader(pBlobVSTextureBlit->GetBufferPointer(), pBlobVSTextureBlit->GetBufferSize(), nullptr, m_PSOBlit.pVS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreatePixelShader(pBlobPSTextureBlit->GetBufferPointer(), pBlobPSTextureBlit->GetBufferSize(), nullptr, m_PSOBlit.pPS.ReleaseAndGetAddressOf()));
    m_PSOBlit.PrimitiveTopology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    DX::ThrowIfFailed(m_pDevice->CreateVertexShader(pBlobVSDebugTiles->GetBufferPointer(), pBlobVSDebugTiles->GetBufferSize(), nullptr, m_PSODegugTiles.pVS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreateGeometryShader(pBlobGSDebugTiles->GetBufferPointer(), pBlobGSDebugTiles->GetBufferSize(), nullptr, m_PSODegugTiles.pGS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreatePixelShader(pBlobPSDegugTiles->GetBufferPointer(), pBlobPSDegugTiles->GetBufferSize(), nullptr, m_PSODegugTiles.pPS.ReleaseAndGetAddressOf()));
    m_PSODegugTiles.PrimitiveTopology = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;

    DX::ThrowIfFailed(m_pDevice->CreateVertexShader(pBlobVSGridLine->GetBufferPointer(), pBlobVSGridLine->GetBufferSize(), nullptr, m_PSOGridLine.pVS.ReleaseAndGetAddressOf()));
    DX::ThrowIfFailed(m_pDevice->CreatePixelShader(pBlobPSGridLine->GetBufferPointer(), pBlobPSGridLine->GetBufferSize(), nullptr, m_PSOGridLine.pPS.ReleaseAndGetAddressOf()));
    m_PSOGridLine.PrimitiveTopology = D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
}

void ApplicationVolumeRender::InitializeVolumeTexture()
{

#ifdef USE_TIFF

    std::vector<uint16_t> intensity;
    int width, height, depth;
    LoadVolumeDataFromTiff("content/Textures/mouse_stack.tif", intensity, width, height, depth);
    m_DimensionX = width;
    m_DimensionY = height;
    m_DimensionZ = depth;
    m_DimensionMipLevels = static_cast<uint16_t>(std::ceil(std::log2(std::max(std::max(m_DimensionX, m_DimensionY), m_DimensionZ)))) + 1;

#else
    std::unique_ptr<FILE, decltype(&fclose)> pFile(fopen("content/Textures/manix.dat", "rb"), fclose);
    if (!pFile)
        throw std::runtime_error("Failed to open file: " + std::string("Data/Textures/manix.dat"));

    fread(&m_DimensionX, sizeof(uint16_t), 1, pFile.get());
    fread(&m_DimensionY, sizeof(uint16_t), 1, pFile.get());
    fread(&m_DimensionZ, sizeof(uint16_t), 1, pFile.get());

    std::vector<uint16_t> intensity(size_t(m_DimensionX) * size_t(m_DimensionY) * size_t(m_DimensionZ));
    fread(intensity.data(), sizeof(uint16_t), m_DimensionX * m_DimensionY * m_DimensionZ, pFile.get());
    m_DimensionMipLevels = static_cast<uint16_t>(std::ceil(std::log2(std::max(std::max(m_DimensionX, m_DimensionY), m_DimensionZ)))) + 1;

    auto NormalizeIntensity = [](uint16_t intensity, uint16_t min, uint16_t max) -> uint16_t
    {
        return static_cast<uint16_t>(std::round(std::numeric_limits<uint16_t>::max() * ((intensity - min) / static_cast<F32>(max - min))));
    };
    uint16_t tmin = 0 << 12; // Min HU [0, 4096]
    uint16_t tmax = 1 << 12; // Max HU [0, 4096]
    std::cout << "tmin: " << tmin << " tmax: " << tmax << std::endl;
    for (size_t index = 0u; index < std::size(intensity); index++)
        intensity[index] = NormalizeIntensity(intensity[index], tmin, tmax);

#endif

    std::cout << "UpdateVolumeTexture: " << m_DimensionX << " " << m_DimensionY << " " << m_DimensionZ << " " << m_DimensionMipLevels << std::endl;

    { // 统计直方图
        m_HistogramData = GetHistogram(intensity);
        m_IsHistogramUpdated = true;

        uint16_t min, max;
        GetMinMaxOfHistogram(m_HistogramData, min, max);
        std::cout << "Min: " << min << " Max: " << max << std::endl;

        uint16_t p5 = EvaluateIntensityOfPercentile(GetCDF(m_HistogramData), 0.05);
        uint16_t p95 = EvaluateIntensityOfPercentile(GetCDF(m_HistogramData), 0.95);
        std::cout << "P5: " << p5 << " P95: " << p95 << std::endl;
    }

    { // 生成D3D资源
        DX::ComPtr<ID3D11Texture3D> pTextureIntensity;
        D3D11_TEXTURE3D_DESC desc = {};
        desc.Width = m_DimensionX;
        desc.Height = m_DimensionY;
        desc.Depth = m_DimensionZ;
        desc.Format = DXGI_FORMAT_R16_UNORM;
        desc.MipLevels = m_DimensionMipLevels;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        ;
        DX::ThrowIfFailed(m_pDevice->CreateTexture3D(&desc, nullptr, pTextureIntensity.GetAddressOf()));

        for (uint32_t mipLevelID = 0; mipLevelID < desc.MipLevels; mipLevelID++)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC descSRV = {};
            descSRV.Format = DXGI_FORMAT_R16_UNORM;
            descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            descSRV.Texture3D.MipLevels = 1;
            descSRV.Texture3D.MostDetailedMip = mipLevelID;

            DX::ComPtr<ID3D11ShaderResourceView> pSRVVolumeIntensity;
            DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureIntensity.Get(), &descSRV, pSRVVolumeIntensity.GetAddressOf()));
            m_pSRVVolumeIntensity.push_back(pSRVVolumeIntensity);
        }

        for (uint32_t mipLevelID = 0; mipLevelID < desc.MipLevels; mipLevelID++)
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV = {};
            descUAV.Format = DXGI_FORMAT_R16_UNORM;
            descUAV.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
            descUAV.Texture3D.MipSlice = mipLevelID;
            descUAV.Texture3D.FirstWSlice = 0;
            descUAV.Texture3D.WSize = std::max(m_DimensionZ >> mipLevelID, 1);

            DX::ComPtr<ID3D11UnorderedAccessView> pUAVVolumeIntensity;
            DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureIntensity.Get(), &descUAV, pUAVVolumeIntensity.GetAddressOf()));
            m_pUAVVolumeIntensity.push_back(pUAVVolumeIntensity);
        }

        D3D11_BOX box = {0, 0, 0, desc.Width, desc.Height, desc.Depth};
        m_pImmediateContext->UpdateSubresource(pTextureIntensity.Get(), 0, &box, std::data(intensity), sizeof(uint16_t) * desc.Width, sizeof(uint16_t) * desc.Height * desc.Width);

        for (uint32_t mipLevelID = 1; mipLevelID < desc.MipLevels - 1; mipLevelID++)
        {
            uint32_t threadGroupX = std::max(static_cast<uint32_t>(std::ceil((m_DimensionX >> mipLevelID) / 4.0f)), 1u);
            uint32_t threadGroupY = std::max(static_cast<uint32_t>(std::ceil((m_DimensionY >> mipLevelID) / 4.0f)), 1u);
            uint32_t threadGroupZ = std::max(static_cast<uint32_t>(std::ceil((m_DimensionZ >> mipLevelID) / 4.0f)), 1u);

            ID3D11ShaderResourceView *ppSRVTextures[] = {m_pSRVVolumeIntensity[mipLevelID - 1].Get()};
            ID3D11UnorderedAccessView *ppUAVTextures[] = {m_pUAVVolumeIntensity[mipLevelID + 0].Get()};
            ID3D11SamplerState *ppSamplers[] = {m_pSamplerLinear.Get()};

            ID3D11UnorderedAccessView *ppUAVClear[] = {nullptr};
            ID3D11ShaderResourceView *ppSRVClear[] = {nullptr};
            ID3D11SamplerState *ppSamplerClear[] = {nullptr};

            auto renderPassName = std::format(L"Render Pass: Compute Mip Map [{}] ", mipLevelID);
            m_pAnnotation->BeginEvent(renderPassName.c_str());
            m_PSOGenerateMipLevel.Apply(m_pImmediateContext);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVTextures), ppSRVTextures);
            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVTextures), ppUAVTextures, nullptr);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplers), ppSamplers);
            m_pImmediateContext->Dispatch(threadGroupX, threadGroupY, threadGroupZ);

            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplerClear), ppSamplerClear);
            m_pAnnotation->EndEvent();
        }
        m_pImmediateContext->Flush();
    }

    {
        DX::ComPtr<ID3D11Texture3D> pTextureGradient;
        D3D11_TEXTURE3D_DESC desc = {};
        desc.Width = m_DimensionX;
        desc.Height = m_DimensionY;
        desc.Depth = m_DimensionZ;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.MipLevels = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        DX::ThrowIfFailed(m_pDevice->CreateTexture3D(&desc, nullptr, pTextureGradient.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureGradient.Get(), nullptr, m_pSRVGradient.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureGradient.Get(), nullptr, m_pUAVGradient.ReleaseAndGetAddressOf()));
        {
            const auto threadGroupX = static_cast<uint32_t>(std::ceil(m_DimensionX / 4.0f));
            const auto threadGroupY = static_cast<uint32_t>(std::ceil(m_DimensionY / 4.0f));
            const auto threadGroupZ = static_cast<uint32_t>(std::ceil(m_DimensionZ / 4.0f));

            ID3D11ShaderResourceView *ppSRVTextures[] = {m_pSRVVolumeIntensity[0].Get(), m_pSRVOpacityTF.Get()};
            ID3D11UnorderedAccessView *ppUAVTextures[] = {m_pUAVGradient.Get()};
            ID3D11SamplerState *ppSamplers[] = {m_pSamplerPoint.Get(), m_pSamplerLinear.Get()};

            ID3D11UnorderedAccessView *ppUAVClear[] = {nullptr};
            ID3D11ShaderResourceView *ppSRVClear[] = {nullptr, nullptr};
            ID3D11SamplerState *ppSamplerClear[] = {nullptr, nullptr};

            m_pAnnotation->BeginEvent(L"Render Pass: Compute Gradient");
            m_PSOComputeGradient.Apply(m_pImmediateContext);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVTextures), ppSRVTextures);
            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVTextures), ppUAVTextures, nullptr);
            m_pImmediateContext->Dispatch(threadGroupX, threadGroupY, threadGroupZ);

            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplerClear), ppSamplerClear);
            m_pAnnotation->EndEvent();
        }
        m_pImmediateContext->Flush();
    }
}

void ApplicationVolumeRender::UpdateVolumeTexture(const std::vector<uint16_t> &intensity, int width, int height, int depth)
{
    m_pImmediateContext->Flush();

    m_DimensionX = width;
    m_DimensionY = height;
    m_DimensionZ = depth;
    m_DimensionMipLevels = static_cast<uint16_t>(std::ceil(std::log2(std::max(std::max(m_DimensionX, m_DimensionY), m_DimensionZ)))) + 1;

    std::cout << "UpdateVolumeTexture: " << m_DimensionX << " " << m_DimensionY << " " << m_DimensionZ << " " << m_DimensionMipLevels << std::endl;

    { // 清理旧的资源
        for (auto &pSRV : m_pSRVVolumeIntensity)
            pSRV.Reset();

        for (auto &pUAV : m_pUAVVolumeIntensity)
            pUAV.Reset();

        m_pSRVVolumeIntensity.clear();
        m_pUAVVolumeIntensity.clear();

        m_pSRVGradient.Reset();
        m_pUAVGradient.Reset();
        m_pImmediateContext->Flush();
        this->WaitForGPU();
    }

    std::cout << "Clear old resources." << std::endl;

    {
        // 创建一个3D纹理
        DX::ComPtr<ID3D11Texture3D> pTextureIntensity;
        D3D11_TEXTURE3D_DESC desc = {};
        desc.Width = m_DimensionX;
        desc.Height = m_DimensionY;
        desc.Depth = m_DimensionZ;
        desc.Format = DXGI_FORMAT_R16_UNORM;
        desc.MipLevels = m_DimensionMipLevels;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        ;
        DX::ThrowIfFailed(m_pDevice->CreateTexture3D(&desc, nullptr, pTextureIntensity.GetAddressOf()));

        std::cout << "Create 3D texture." << std::endl;

        // 为每一个mipmap级别都创建一个SRV
        for (uint32_t mipLevelID = 0; mipLevelID < desc.MipLevels; mipLevelID++)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC descSRV = {};
            descSRV.Format = DXGI_FORMAT_R16_UNORM;
            descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            descSRV.Texture3D.MipLevels = 1;
            descSRV.Texture3D.MostDetailedMip = mipLevelID;

            DX::ComPtr<ID3D11ShaderResourceView> pSRVVolumeIntensity;
            DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureIntensity.Get(), &descSRV, pSRVVolumeIntensity.GetAddressOf()));
            m_pSRVVolumeIntensity.push_back(pSRVVolumeIntensity);
        }

        std::cout << "Create SRV." << std::endl;

        // 为每一个mipmap级别都创建一个UAV
        for (uint32_t mipLevelID = 0; mipLevelID < desc.MipLevels; mipLevelID++)
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV = {};
            descUAV.Format = DXGI_FORMAT_R16_UNORM;
            descUAV.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
            descUAV.Texture3D.MipSlice = mipLevelID;
            descUAV.Texture3D.FirstWSlice = 0;
            descUAV.Texture3D.WSize = std::max(m_DimensionZ >> mipLevelID, 1);

            DX::ComPtr<ID3D11UnorderedAccessView> pUAVVolumeIntensity;
            DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureIntensity.Get(), &descUAV, pUAVVolumeIntensity.GetAddressOf()));
            m_pUAVVolumeIntensity.push_back(pUAVVolumeIntensity);
        }

        std::cout << "Create UAV." << std::endl;

        // 更新mipmap级别0的数据，也就是原始数据
        D3D11_BOX box = {0, 0, 0, desc.Width, desc.Height, desc.Depth};
        m_pImmediateContext->UpdateSubresource(pTextureIntensity.Get(), 0, &box, std::data(intensity), sizeof(uint16_t) * desc.Width, sizeof(uint16_t) * desc.Height * desc.Width);
        std::cout << "Update Subresource." << std::endl;

        // 生成mipmap
        for (uint32_t mipLevelID = 1; mipLevelID < desc.MipLevels - 1; mipLevelID++)
        {
            uint32_t threadGroupX = std::max(static_cast<uint32_t>(std::ceil((m_DimensionX >> mipLevelID) / 4.0f)), 1u);
            uint32_t threadGroupY = std::max(static_cast<uint32_t>(std::ceil((m_DimensionY >> mipLevelID) / 4.0f)), 1u);
            uint32_t threadGroupZ = std::max(static_cast<uint32_t>(std::ceil((m_DimensionZ >> mipLevelID) / 4.0f)), 1u);

            ID3D11ShaderResourceView *ppSRVTextures[] = {m_pSRVVolumeIntensity[mipLevelID - 1].Get()};
            ID3D11UnorderedAccessView *ppUAVTextures[] = {m_pUAVVolumeIntensity[mipLevelID + 0].Get()};
            ID3D11SamplerState *ppSamplers[] = {m_pSamplerLinear.Get()};

            ID3D11UnorderedAccessView *ppUAVClear[] = {nullptr};
            ID3D11ShaderResourceView *ppSRVClear[] = {nullptr};
            ID3D11SamplerState *ppSamplerClear[] = {nullptr};

            auto renderPassName = std::format(L"Render Pass: Compute Mip Map [{}] ", mipLevelID);
            m_pAnnotation->BeginEvent(renderPassName.c_str());
            m_PSOGenerateMipLevel.Apply(m_pImmediateContext);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVTextures), ppSRVTextures);
            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVTextures), ppUAVTextures, nullptr);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplers), ppSamplers);
            m_pImmediateContext->Dispatch(threadGroupX, threadGroupY, threadGroupZ);

            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplerClear), ppSamplerClear);
            m_pAnnotation->EndEvent();
        }
        std::cout << "Generate Mip Map." << std::endl;
        m_pImmediateContext->Flush();
        this->WaitForGPU();
    }

    std::cout << "UpdateVolumeTexture Done." << std::endl;

    { // 生成梯度纹理
        DX::ComPtr<ID3D11Texture3D> pTextureGradient;
        D3D11_TEXTURE3D_DESC desc = {};
        desc.Width = m_DimensionX;
        desc.Height = m_DimensionY;
        desc.Depth = m_DimensionZ;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.MipLevels = 1;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.Usage = D3D11_USAGE_DEFAULT;
        DX::ThrowIfFailed(m_pDevice->CreateTexture3D(&desc, nullptr, pTextureGradient.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureGradient.Get(), nullptr, m_pSRVGradient.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureGradient.Get(), nullptr, m_pUAVGradient.ReleaseAndGetAddressOf()));
        {
            const auto threadGroupX = static_cast<uint32_t>(std::ceil(m_DimensionX / 4.0f));
            const auto threadGroupY = static_cast<uint32_t>(std::ceil(m_DimensionY / 4.0f));
            const auto threadGroupZ = static_cast<uint32_t>(std::ceil(m_DimensionZ / 4.0f));

            ID3D11ShaderResourceView *ppSRVTextures[] = {m_pSRVVolumeIntensity[0].Get(), m_pSRVOpacityTF.Get()};
            ID3D11UnorderedAccessView *ppUAVTextures[] = {m_pUAVGradient.Get()};
            ID3D11SamplerState *ppSamplers[] = {m_pSamplerPoint.Get(), m_pSamplerLinear.Get()};

            ID3D11UnorderedAccessView *ppUAVClear[] = {nullptr};
            ID3D11ShaderResourceView *ppSRVClear[] = {nullptr, nullptr};
            ID3D11SamplerState *ppSamplerClear[] = {nullptr, nullptr};

            m_pAnnotation->BeginEvent(L"Render Pass: Compute Gradient");
            m_PSOComputeGradient.Apply(m_pImmediateContext);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVTextures), ppSRVTextures);
            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVTextures), ppUAVTextures, nullptr);
            m_pImmediateContext->Dispatch(threadGroupX, threadGroupY, threadGroupZ);

            m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
            m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
            m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplerClear), ppSamplerClear);
            m_pAnnotation->EndEvent();
        }
        m_pImmediateContext->Flush();
        this->WaitForGPU();
    }

    std::cout << "UpdateGradient Done." << std::endl;
}

void ApplicationVolumeRender::InitializeTransferFunction()
{

    nlohmann::json root;
    std::ifstream file("content/TransferFunctions/ManixTransferFunction.json");
    file >> root;

    m_OpacityTransferFunc.Clear();
    m_DiffuseTransferFunc.Clear();
    m_SpecularTransferFunc.Clear();
    m_EmissionTransferFunc.Clear();
    m_RoughnessTransferFunc.Clear();

    auto ExtractVec3FromJson = [](auto const &tree, auto const &key) -> Hawk::Math::Vec3
    {
        Hawk::Math::Vec3 v{};
        uint32_t index = 0;
        for (auto &e : tree[key])
        {
            v[index] = e.template get<float>();
            index++;
        }
        return v;
    };

    for (auto const &e : root["NodesColor"])
    {
        const auto intensity = e["Intensity"].get<float>();
        const auto diffuse = ExtractVec3FromJson(e, "Diffuse");
        const auto specular = ExtractVec3FromJson(e, "Specular");
        const auto roughness = e["Roughness"].get<float>();

        m_DiffuseTransferFunc.AddNode(intensity, diffuse);
        m_SpecularTransferFunc.AddNode(intensity, specular);
        m_RoughnessTransferFunc.AddNode(intensity, roughness);
    }

    for (auto const &e : root["NodesOpacity"])
        m_OpacityTransferFunc.AddNode(e["Intensity"].get<F32>(), e["Opacity"].get<F32>());

    m_pSRVOpacityTF = m_OpacityTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
    m_pSRVDiffuseTF = m_DiffuseTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
    m_pSRVSpecularTF = m_SpecularTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
    m_pSRVRoughnessTF = m_RoughnessTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
}

void ApplicationVolumeRender::InitializeSamplerStates()
{

    auto createSamplerState = [this](auto filter, auto addressMode) -> DX::ComPtr<ID3D11SamplerState>
    {
        D3D11_SAMPLER_DESC desc = {};
        desc.Filter = filter;
        desc.AddressU = addressMode;
        desc.AddressV = addressMode;
        desc.AddressW = addressMode;
        desc.MaxAnisotropy = D3D11_MAX_MAXANISOTROPY;
        desc.MaxLOD = FLT_MAX;
        desc.ComparisonFunc = D3D11_COMPARISON_NEVER;

        DX::ComPtr<ID3D11SamplerState> pSamplerState;
        DX::ThrowIfFailed(m_pDevice->CreateSamplerState(&desc, pSamplerState.GetAddressOf()));
        return pSamplerState;
    };

    m_pSamplerPoint = createSamplerState(D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER);
    m_pSamplerLinear = createSamplerState(D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D11_TEXTURE_ADDRESS_BORDER);
    m_pSamplerAnisotropic = createSamplerState(D3D11_FILTER_ANISOTROPIC, D3D11_TEXTURE_ADDRESS_WRAP);
}

void ApplicationVolumeRender::InitializeRenderTextures()
{

    { // 散射纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureDiffuse;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureDiffuse.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureDiffuse.Get(), nullptr, m_pSRVDiffuse.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureDiffuse.Get(), nullptr, m_pUAVDiffuse.ReleaseAndGetAddressOf()));
    }

    { // 镜面反射纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureSpecular;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureSpecular.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureSpecular.Get(), nullptr, m_pSRVSpecular.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureSpecular.Get(), nullptr, m_pUAVSpecular.ReleaseAndGetAddressOf()));
    }

    { // 辐射度纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureDiffuseLight;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureDiffuseLight.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureDiffuseLight.Get(), nullptr, m_pSRVRadiance.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureDiffuseLight.Get(), nullptr, m_pUAVRadiance.ReleaseAndGetAddressOf()));
    }

    { // 法线纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureNormal;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureNormal.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureNormal.Get(), nullptr, m_pSRVNormal.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureNormal.Get(), nullptr, m_pUAVNormal.ReleaseAndGetAddressOf()));
    }

    { // 深度纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R32_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureDepth;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureDepth.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureDepth.Get(), nullptr, m_pSRVDepth.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureDepth.Get(), nullptr, m_pUAVDepth.ReleaseAndGetAddressOf()));
    }

    { // 颜色和光照的累加纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureColorSum;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureColorSum.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureColorSum.Get(), nullptr, m_pSRVColorSum.ReleaseAndGetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureColorSum.Get(), nullptr, m_pUAVColorSum.ReleaseAndGetAddressOf()));
    }

    { // 调色纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.ArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.Width = m_ApplicationDesc.Width;
        desc.Height = m_ApplicationDesc.Height;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        DX::ComPtr<ID3D11Texture2D> pTextureToneMap;
        DX::ThrowIfFailed(m_pDevice->CreateTexture2D(&desc, nullptr, pTextureToneMap.GetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pTextureToneMap.Get(), nullptr, m_pSRVToneMap.GetAddressOf()));
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pTextureToneMap.Get(), nullptr, m_pUAVToneMap.GetAddressOf()));
    }
}

void ApplicationVolumeRender::InitializeBuffers()
{

    m_pConstantBufferFrame = DX::CreateConstantBuffer<FrameBuffer>(m_pDevice);
    m_pDispatchIndirectBufferArgs = DX::CreateIndirectBuffer<DispatchIndirectBuffer>(m_pDevice, DispatchIndirectBuffer{1, 1, 1});
    m_pDrawInstancedIndirectBufferArgs = DX::CreateIndirectBuffer<DrawInstancedIndirectBuffer>(m_pDevice, DrawInstancedIndirectBuffer{0, 1, 0, 0});

    // 创建GridLine所需的buffer，并且默认填充两条线的顶点
    std::vector<DX::Vertex> demoVertices = {
        {DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f), DirectX::XMFLOAT4(1.0f, 0.0f, 0.0f, 1.0f)},
        {DirectX::XMFLOAT3(1.0f, 1.0f, 0.0f), DirectX::XMFLOAT4(0.0f, 1.0f, 0.0f, 1.0f)}};
    m_pGridLineVertexBuffer = DX::CreateStructuredBuffer<DX::Vertex>(m_pDevice, demoVertices.size(), false, true, demoVertices.data());
}

void ApplicationVolumeRender::InitializeTileBuffer()
{

    const auto threadGroupsX = static_cast<uint32_t>(std::ceil(m_ApplicationDesc.Width / 8));
    const auto threadGroupsY = static_cast<uint32_t>(std::ceil(m_ApplicationDesc.Height / 8));

    DX::ComPtr<ID3D11Buffer> pBuffer = DX::CreateStructuredBuffer<uint32_t>(m_pDevice, threadGroupsX * threadGroupsY, false, true, nullptr);
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC desc = {};
        desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        desc.BufferEx.FirstElement = 0;
        desc.BufferEx.NumElements = threadGroupsX * threadGroupsY;
        DX::ThrowIfFailed(m_pDevice->CreateShaderResourceView(pBuffer.Get(), &desc, m_pSRVDispersionTiles.ReleaseAndGetAddressOf()));
    }

    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC desc = {};
        desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = 0;
        desc.Buffer.NumElements = threadGroupsX * threadGroupsY;
        desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
        DX::ThrowIfFailed(m_pDevice->CreateUnorderedAccessView(pBuffer.Get(), &desc, m_pUAVDispersionTiles.ReleaseAndGetAddressOf()));
    }
}

void ApplicationVolumeRender::InitializeEnvironmentMap()
{
    DX::ThrowIfFailed(DirectX::CreateDDSTextureFromFile(m_pDevice.Get(), L"content/Textures/qwantani_2k.dds", nullptr, m_pSRVEnvironment.GetAddressOf()));
}

void ApplicationVolumeRender::Resize(int32_t width, int32_t height)
{
    Base::Resize(width, height);
    InitializeRenderTextures();
    InitializeTileBuffer();
    m_FrameIndex = 0;
}

void ApplicationVolumeRender::EventMouseWheel(float delta)
{

    m_Zoom -= m_ZoomSensitivity * m_DeltaTime * delta;
    m_FrameIndex = 0;
}

void ApplicationVolumeRender::EventMouseMove(float x, float y)
{

    m_Camera.Rotate(Hawk::Components::Camera::LocalUp, m_DeltaTime * -m_RotateSensitivity * x);
    m_Camera.Rotate(m_Camera.Right(), m_DeltaTime * -m_RotateSensitivity * y);
    m_FrameIndex = 0;
}

void ApplicationVolumeRender::Update(float deltaTime)
{

    m_DeltaTime = deltaTime;

    try
    {
        if (m_IsReloadShader)
        {
            InitializeShaders();
            m_IsReloadShader = false;
        }

        if (m_IsReloadTransferFunc)
        {
            InitializeTransferFunction();
            m_IsReloadTransferFunc = false;
        }

        if (m_IsVolumeDataLoaded)
        {
            // 读取tif数据
            std::vector<uint16_t> intensity;
            int width, height, depth;

            if (LoadVolumeDataFromTiff(m_VolumeDataPath, intensity, width, height, depth))
            {
                // 计算Histogram
                auto histogram = GetHistogram(intensity);

                // 计算最小值和最大值
                uint16_t min, max;
                GetMinMaxOfHistogram(histogram, min, max);
                std::cout << "min: " << min << " max: " << max << std::endl;

                // 计算CDF
                auto cdf = GetCDF(histogram);

                // 计算CDF的百分之5和百分之95，作为最小值和最大值
                auto p5 = EvaluateIntensityOfPercentile(cdf, 0.05f);
                auto p95 = EvaluateIntensityOfPercentile(cdf, 0.95f);

                std::cout << "p5: " << p5 << " p95: " << p95 << std::endl;

                // 依据p5和p95的值，重新映射强度值
                for (auto &e : intensity)
                {
                    double intensityNorm = (e - p5) / (p95 - p5);
                    intensityNorm = (std::clamp)(intensityNorm, 0.0, 1.0);
                    e = static_cast<uint16_t>(intensityNorm * std::numeric_limits<uint16_t>::max());
                }

                try
                {
                    UpdateVolumeTexture(intensity, width, height, depth);
                }
                catch (std::exception const &e)
                {
                    std::cout << "update volume data error " << e.what() << std::endl;
                }

                m_IsFirstFrameAfterVolumeDataLoad = true;
                m_IsVolumeDataLoaded = false;
                m_FrameIndex = 0;
            }
        }

        if (m_IsRecreateOpacityTexture) // 更新透明度映射表
        {
            m_pSRVOpacityTF = m_OpacityTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
            m_IsRecreateOpacityTexture = false;
            m_FrameIndex = 0;
        }

        if (m_IsRecreateDiffuseTexture) // 更新漫反射映射表
        {
            m_pSRVDiffuseTF = m_DiffuseTransferFunc.GenerateTexture(m_pDevice, m_SamplingCount);
            m_IsRecreateDiffuseTexture = false;
            m_FrameIndex = 0;
        }
    }
    catch (std::exception const &e)
    {
        std::cout << e.what() << std::endl;
    }

    // 实际物体的尺寸
    Hawk::Math::Vec3 scaleVector = {0.488f * m_DimensionX, 0.488f * m_DimensionY, 0.7f * m_DimensionZ}; // 通常z的间距要比XY的像素间距要大，当然要根据具体的扫描间距来

    double maxDim = (std::max)({scaleVector.x, scaleVector.y, scaleVector.z});
    double gridSpacing = maxDim / 6.0;
    gridSpacing = std::ceil(gridSpacing / 10.0) * 10.0;
    std::vector<double> xTicks, yTicks, zTicks;
    for (double i = 0; i <= scaleVector.x; i += gridSpacing)
        xTicks.push_back(i - scaleVector.x * 0.5);
    for (double i = 0; i <= scaleVector.y; i += gridSpacing)
        yTicks.push_back(i - scaleVector.y * 0.5);
    for (double i = 0; i <= scaleVector.z; i += gridSpacing)
        zTicks.push_back(i - scaleVector.z * 0.5);

    scaleVector /= (std::max)({scaleVector.x, scaleVector.y, scaleVector.z});

    // 对ticks也进行NDC空间的缩放
    for (auto &e : xTicks)
        e /= maxDim;
    for (auto &e : yTicks)
        e /= maxDim;
    for (auto &e : zTicks)
        e /= maxDim;

    m_GridLineVertexCnt = 2 * 2 * (xTicks.size() + yTicks.size() + zTicks.size());

    Hawk::Math::Mat4x4 V = m_Camera.ToMatrix();
    Hawk::Math::Mat4x4 P = Hawk::Math::Orthographic(m_Zoom * (m_ApplicationDesc.Width / static_cast<F32>(m_ApplicationDesc.Height)), m_Zoom, -1.0f, 1.0f);
    Hawk::Math::Mat4x4 W = Hawk::Math::RotateX(Hawk::Math::Radians(-90.0f)) * Hawk::Math::Scale(scaleVector);
    Hawk::Math::Mat4x4 N = Hawk::Math::Inverse(Hawk::Math::Transpose(W));

    Hawk::Math::Mat4x4 VP = P * V;
    Hawk::Math::Mat4x4 NV = V * N;
    Hawk::Math::Mat4x4 WVP = P * V * W;
    m_WVP = WVP;

    {
        DX::MapHelper<FrameBuffer> map(m_pImmediateContext, m_pConstantBufferFrame, D3D11_MAP_WRITE_DISCARD, 0);

        map->ProjectionMatrix = P;
        map->ViewMatrix = V;
        map->WorldMatrix = W;
        map->NormalMatrix = N;

        map->InvProjectionMatrix = Hawk::Math::Inverse(P);
        map->InvViewMatrix = Hawk::Math::Inverse(V);
        map->InvWorldMatrix = Hawk::Math::Inverse(W);
        map->InvNormalMatrix = Hawk::Math::Inverse(N);

        map->ViewProjectionMatrix = VP;
        map->NormalViewMatrix = NV;
        map->WorldViewProjectionMatrix = WVP;

        map->InvViewProjectionMatrix = Hawk::Math::Inverse(VP);
        map->InvNormalViewMatrix = Hawk::Math::Inverse(NV);
        map->InvWorldViewProjectionMatrix = Hawk::Math::Inverse(WVP);

        map->BoundingBoxMin = m_BoundingBoxMin;
        map->BoundingBoxMax = m_BoundingBoxMax;

        map->StepSize = Hawk::Math::Distance(map->BoundingBoxMin, map->BoundingBoxMax) / m_StepCount;

        map->Density = m_Density;
        map->FrameIndex = m_FrameIndex;
        map->Exposure = m_Exposure;

        map->FrameOffset = Hawk::Math::Vec2(m_RandomDistribution(m_RandomGenerator), m_RandomDistribution(m_RandomGenerator));
        map->RenderTargetDim = Hawk::Math::Vec2(static_cast<F32>(m_ApplicationDesc.Width), static_cast<F32>(m_ApplicationDesc.Height));
        map->InvRenderTargetDim = Hawk::Math::Vec2(1.0f, 1.0f) / map->RenderTargetDim;

        // map->GridLineCnt = 2; // 每个tick都需要两条线
        map->GridLineInfo.x = static_cast<F32>(xTicks.size()); // x轴的ticks数量
        map->GridLineInfo.y = static_cast<F32>(yTicks.size()); // y轴的ticks数量
        map->GridLineInfo.z = static_cast<F32>(zTicks.size()); // z轴的ticks数量
        map->GridLineInfoStart.x = xTicks[0];
        map->GridLineInfoStart.y = yTicks[0];
        map->GridLineInfoStart.z = zTicks[0];
        map->GridLineInfoStep.x = xTicks[1] - xTicks[0];
        map->GridLineInfoStep.y = yTicks[1] - yTicks[0];
        map->GridLineInfoStep.z = zTicks[1] - zTicks[0];
    }
}

void ApplicationVolumeRender::TextureBlit(DX::ComPtr<ID3D11ShaderResourceView> pSrc, DX::ComPtr<ID3D11RenderTargetView> pDst)
{

    ID3D11ShaderResourceView *ppSRVClear[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(m_ApplicationDesc.Width), static_cast<float>(m_ApplicationDesc.Height), 0.0f, 1.0f};
    D3D11_RECT scissor = {0, 0, static_cast<int32_t>(m_ApplicationDesc.Width), static_cast<int32_t>(m_ApplicationDesc.Height)};

    { // 清深度缓冲

        D3D11_DEPTH_STENCIL_DESC dsDesc;
        ZeroMemory(&dsDesc, sizeof(dsDesc));
        dsDesc.DepthEnable = TRUE;
        dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
        dsDesc.StencilEnable = FALSE;

        ID3D11DepthStencilState *pDSState;
        m_pDevice->CreateDepthStencilState(&dsDesc, &pDSState);
        m_pImmediateContext->OMSetDepthStencilState(pDSState, 1);
    }

    { // 清颜色缓冲

        float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        m_pImmediateContext->ClearRenderTargetView(pDst.Get(), clearColor);
    }

    m_pImmediateContext->ClearDepthStencilView(m_pDSV.Get(), D3D11_CLEAR_DEPTH, 1.0f, 0);
    m_pImmediateContext->OMSetRenderTargets(1, pDst.GetAddressOf(), m_pDSV.Get());
    // m_pImmediateContext->OMSetRenderTargets(1, pDst.GetAddressOf(), nullptr);
    m_pImmediateContext->RSSetScissorRects(1, &scissor);
    m_pImmediateContext->RSSetViewports(1, &viewport);

    // Bind PSO and Resources
    m_PSOBlit.Apply(m_pImmediateContext);
    m_pImmediateContext->PSSetShaderResources(0, 1, pSrc.GetAddressOf());
    m_pImmediateContext->PSSetSamplers(0, 1, m_pSamplerPoint.GetAddressOf());

    // Execute
    m_pImmediateContext->Draw(6, 0);

    // Unbind RTV's
    m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);
    m_pImmediateContext->RSSetScissorRects(0, nullptr);
    m_pImmediateContext->RSSetViewports(0, nullptr);

    // Unbind PSO and unbind Resources
    m_PSODefault.Apply(m_pImmediateContext);
    m_pImmediateContext->PSSetSamplers(0, 0, nullptr);
    m_pImmediateContext->PSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
}

void ApplicationVolumeRender::DrawGridLine(DX::ComPtr<ID3D11RenderTargetView> pDst)
{
    ID3D11ShaderResourceView *ppSRVClear[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(m_ApplicationDesc.Width), static_cast<float>(m_ApplicationDesc.Height), 0.0f, 1.0f};
    D3D11_RECT scissor = {0, 0, static_cast<int32_t>(m_ApplicationDesc.Width), static_cast<int32_t>(m_ApplicationDesc.Height)};

    // 设置渲染目标、裁剪矩形和视口
    m_pImmediateContext->OMSetRenderTargets(1, pDst.GetAddressOf(), m_pDSV.Get());

    m_pImmediateContext->RSSetViewports(1, &viewport);

    // 绑定PSO
    m_PSOGridLine.Apply(m_pImmediateContext);

    // 执行
    m_pImmediateContext->Draw(m_GridLineVertexCnt, 0);

    // 解绑RTV
    m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);
    m_pImmediateContext->RSSetViewports(0, nullptr);

    // 解绑PSO以及资源
    m_PSODefault.Apply(m_pImmediateContext);
    m_pImmediateContext->PSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
}

void ApplicationVolumeRender::RenderFrame(DX::ComPtr<ID3D11RenderTargetView> pRTV)
{
    if (m_IsFirstFrameAfterVolumeDataLoad)
    {
        std::cout << "First Frame After Volume Data Load." << std::endl;
        m_IsFirstFrameAfterVolumeDataLoad = false;
    }

    ID3D11UnorderedAccessView *ppUAVClear[] = {nullptr, nullptr, nullptr, nullptr};
    ID3D11ShaderResourceView *ppSRVClear[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    const auto threadGroupsX = static_cast<uint32_t>(std::ceil(m_ApplicationDesc.Width / 8.0f));
    const auto threadGroupsY = static_cast<uint32_t>(std::ceil(m_ApplicationDesc.Height / 8.0f));

    m_pImmediateContext->VSSetConstantBuffers(0, 1, m_pConstantBufferFrame.GetAddressOf());
    m_pImmediateContext->GSSetConstantBuffers(0, 1, m_pConstantBufferFrame.GetAddressOf());
    m_pImmediateContext->PSSetConstantBuffers(0, 1, m_pConstantBufferFrame.GetAddressOf());
    m_pImmediateContext->CSSetConstantBuffers(0, 1, m_pConstantBufferFrame.GetAddressOf());

    if (m_FrameIndex < m_SampleDispersion)
    {
        ID3D11UnorderedAccessView *ppUAVResources[] = {m_pUAVDispersionTiles.Get()};
        constexpr uint32_t pCounters[] = {0}; // 通常伴随UAV可以设置一个计数器，用于原子操作

        m_pAnnotation->BeginEvent(L"Render Pass: Reset computed tiles");
        m_PSOResetTiles.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, pCounters);
        m_pImmediateContext->Dispatch(threadGroupsX, threadGroupsY, 1);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pAnnotation->EndEvent();
    }
    else
    {
        ID3D11ShaderResourceView *ppSRVResources[] = {m_pSRVToneMap.Get(), m_pSRVDepth.Get()};
        ID3D11UnorderedAccessView *ppUAVResources[] = {m_pUAVDispersionTiles.Get()};
        constexpr uint32_t pCounters[] = {0};

        m_pAnnotation->BeginEvent(L"Render Pass: Generate computed tiles");
        m_PSOComputeTiles.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, pCounters);
        m_pImmediateContext->Dispatch(threadGroupsX, threadGroupsY, 1);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_pAnnotation->EndEvent();
    }

    constexpr float clearColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
    m_pAnnotation->BeginEvent(L"Render Pass: Clear buffers [Color, Normal, Depth]");
    m_pImmediateContext->ClearUnorderedAccessViewFloat(m_pUAVDiffuse.Get(), clearColor);
    m_pImmediateContext->ClearUnorderedAccessViewFloat(m_pUAVNormal.Get(), clearColor);
    m_pImmediateContext->ClearUnorderedAccessViewFloat(m_pUAVDepth.Get(), clearColor);
    m_pImmediateContext->ClearUnorderedAccessViewFloat(m_pUAVRadiance.Get(), clearColor);
    m_pAnnotation->EndEvent();

    // 获取Tiles计数器的值
    m_pAnnotation->BeginEvent(L"Render Pass: Copy counters of tiles");
    m_pImmediateContext->CopyStructureCount(m_pDispatchIndirectBufferArgs.Get(), 0, m_pUAVDispersionTiles.Get());
    m_pImmediateContext->CopyStructureCount(m_pDrawInstancedIndirectBufferArgs.Get(), 0, m_pUAVDispersionTiles.Get());
    m_pAnnotation->EndEvent();

    { // 生成光线
        ID3D11SamplerState *ppSamplers[] = {
            m_pSamplerPoint.Get(),
            m_pSamplerLinear.Get(),
            m_pSamplerAnisotropic.Get()};

        ID3D11ShaderResourceView *ppSRVResources[] = {
            m_pSRVVolumeIntensity[m_MipLevel].Get(),
            m_pSRVGradient.Get(),
            m_pSRVDiffuseTF.Get(),
            m_pSRVSpecularTF.Get(),
            m_pSRVRoughnessTF.Get(),
            m_pSRVOpacityTF.Get(),
            m_pSRVDispersionTiles.Get()};

        ID3D11UnorderedAccessView *ppUAVResources[] = {
            m_pUAVDiffuse.Get(),
            m_pUAVSpecular.Get(),
            m_pUAVNormal.Get(),
            m_pUAVDepth.Get()};

        m_pAnnotation->BeginEvent(L"Render pass: Generate Rays");
        m_PSOGeneratePrimaryRays.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplers), ppSamplers);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, nullptr);
        m_pImmediateContext->DispatchIndirect(m_pDispatchIndirectBufferArgs.Get(), 0);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_pAnnotation->EndEvent();
    }

    { // 计算辐射度
        ID3D11SamplerState *ppSamplers[] = {
            m_pSamplerPoint.Get(),
            m_pSamplerLinear.Get(),
            m_pSamplerAnisotropic.Get()};

        ID3D11ShaderResourceView *ppSRVResources[] = {
            m_pSRVVolumeIntensity[m_MipLevel].Get(),
            m_pSRVOpacityTF.Get(),
            m_pSRVDiffuse.Get(),
            m_pSRVSpecular.Get(),
            m_pSRVNormal.Get(),
            m_pSRVDepth.Get(),
            m_pSRVEnvironment.Get(),
            m_pSRVDispersionTiles.Get()};

        ID3D11UnorderedAccessView *ppUAVResources[] = {
            m_pUAVRadiance.Get()};

        m_pAnnotation->BeginEvent(L"Render pass: Compute Radiance");
        m_PSOComputeDiffuseLight.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetSamplers(0, _countof(ppSamplers), ppSamplers);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, nullptr);
        m_pImmediateContext->DispatchIndirect(m_pDispatchIndirectBufferArgs.Get(), 0);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_pAnnotation->EndEvent();
    }

    { // 累积辐射度
        ID3D11ShaderResourceView *ppSRVResources[] = {m_pSRVRadiance.Get(), m_pSRVDispersionTiles.Get()};
        ID3D11UnorderedAccessView *ppUAVResources[] = {m_pUAVColorSum.Get()};

        m_pAnnotation->BeginEvent(L"Render Pass: Accumulate");
        m_PSOAccumulate.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, nullptr);
        m_pImmediateContext->DispatchIndirect(m_pDispatchIndirectBufferArgs.Get(), 0);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_pAnnotation->EndEvent();
    }

    { // 色调映射
        ID3D11ShaderResourceView *ppSRVResources[] = {m_pSRVColorSum.Get(), m_pSRVDispersionTiles.Get()};
        ID3D11UnorderedAccessView *ppUAVResources[] = {m_pUAVToneMap.Get()};

        m_pAnnotation->BeginEvent(L"Render Pass: Tone Map");
        m_PSOToneMap.Apply(m_pImmediateContext);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVResources), ppUAVResources, nullptr);
        m_pImmediateContext->DispatchIndirect(m_pDispatchIndirectBufferArgs.Get(), 0);
        m_pImmediateContext->CSSetUnorderedAccessViews(0, _countof(ppUAVClear), ppUAVClear, nullptr);
        m_pImmediateContext->CSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_pAnnotation->EndEvent();
    }

    // 贴到后台缓冲区
    m_pAnnotation->BeginEvent(L"Render Pass: TextureBlit [Tone Map] -> [Back Buffer]");
    this->TextureBlit(m_pSRVToneMap, pRTV);
    m_pAnnotation->EndEvent();

    // 渲染到后台缓冲区
    this->DrawGridLine(pRTV);

    // 渲染tiles线
    if (m_IsDrawDebugTiles)
    {
        ID3D11ShaderResourceView *ppSRVResources[] = {m_pSRVDispersionTiles.Get()};
        D3D11_VIEWPORT viewport = {0.0f, 0.0f, static_cast<float>(m_ApplicationDesc.Width), static_cast<float>(m_ApplicationDesc.Height), 0.0f, 1.0f};

        m_pAnnotation->BeginEvent(L"Render Pass: Debug -> [Generated tiles]");
        m_pImmediateContext->OMSetRenderTargets(1, pRTV.GetAddressOf(), nullptr);
        m_pImmediateContext->RSSetViewports(1, &viewport);

        m_PSODegugTiles.Apply(m_pImmediateContext);
        m_pImmediateContext->VSSetShaderResources(0, _countof(ppSRVResources), ppSRVResources);
        m_pImmediateContext->DrawInstancedIndirect(m_pDrawInstancedIndirectBufferArgs.Get(), 0);

        m_pImmediateContext->VSSetShaderResources(0, _countof(ppSRVClear), ppSRVClear);
        m_PSODefault.Apply(m_pImmediateContext);
        m_pImmediateContext->RSSetViewports(0, nullptr);
        m_pImmediateContext->OMSetRenderTargets(0, nullptr, nullptr);
        m_pAnnotation->EndEvent();
    }

    m_FrameIndex++;
}

void ApplicationVolumeRender::RenderGUI(DX::ComPtr<ID3D11RenderTargetView> pRTV)
{
    assert(ImGui::GetCurrentContext() != nullptr && "Missing dear imgui context. Refer to examples app!");

    ImGui::ShowDemoWindow();

    ImGui::Begin("Settings");

    static bool isShowAppMetrics = false;
    static bool isShowAppAbout = false;

    { // debug相机信息相关
        Hawk::Math::Vec4 position_o = {0.0f, 0.0f, 0.0f, 1.0f};
        Hawk::Math::Vec4 position_x = {1.0f, 0.0f, 0.0f, 1.0f};
        Hawk::Math::Vec4 position_y = {0.0f, 1.0f, 0.0f, 1.0f};
        Hawk::Math::Vec4 position_z = {0.0f, 0.0f, 1.0f, 1.0f};

        position_o = m_WVP * position_o;
        position_x = m_WVP * position_x;
        position_y = m_WVP * position_y;
        position_z = m_WVP * position_z;

        ImGui::Text("O projected to: %.2f %.2f %.2f %.2f", position_o.x, position_o.y, position_o.z, position_o.w);
        ImGui::Text("X projected to: %.2f %.2f %.2f %.2f", position_x.x, position_x.y, position_x.z, position_x.w);
        ImGui::Text("Y projected to: %.2f %.2f %.2f %.2f", position_y.x, position_y.y, position_y.z, position_y.w);
        ImGui::Text("Z projected to: %.2f %.2f %.2f %.2f", position_z.x, position_z.y, position_z.z, position_z.w);
    }

    if (ImGui::Button("Load Volume"))
    {
        std::wstring filter = L"Volume Files\0*.tif\0";
        auto select_file = utils::SelectFile(filter.c_str());
        std::wcout << select_file << std::endl;

        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        m_VolumeDataPath = conv.to_bytes(select_file);

        m_IsVolumeDataLoaded = true;
    }

    if (!m_VolumeDataPath.empty())
    {
        ImGui::SameLine();
        ImGui::Text("Volume Path: %s", m_VolumeDataPath.c_str());
    }

    ImGui::Text("Frame Rate: %.1f FPS", ImGui::GetIO().Framerate);

    if (isShowAppMetrics)
        ImGui::ShowMetricsWindow(&isShowAppMetrics);

    if (isShowAppAbout)
        ImGui::ShowAboutWindow(&isShowAppAbout);

    if (ImGui::CollapsingHeader("TransferFunction", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static double X_LIMIT_MAX = 65536.0;
        static double X_LIMIT_MIN = 0.0;
        ImPlot::SetNextPlotLimitsX(X_LIMIT_MIN, X_LIMIT_MAX);
        ImPlot::SetNextPlotLimitsY(0.0f, 1.025f);
        if (ImPlot::BeginPlot("Opacity", 0, 0, ImVec2(-1, 175), 0,
                              ImPlotAxisFlags_LockMin | ImPlotAxisFlags_LockMax,
                              ImPlotAxisFlags_LockMin | ImPlotAxisFlags_LockMax))
        {

            { // 绘制直方图
                static std::vector<double> histogramX, histogramY;
                if (m_IsHistogramUpdated)
                {
                    histogramX.clear();
                    histogramY.clear();

                    std::vector<double> histogramNormalized = m_HistogramData;

                    double max = *std::max_element(histogramNormalized.begin(), histogramNormalized.end()); // 找最大值

                    std::transform(histogramNormalized.begin(), histogramNormalized.end(), histogramNormalized.begin(), [max](double value) // 归一化
                                   { return value / max; });

                    for (size_t i = 0; i < histogramNormalized.size(); i++)
                    {
                        histogramX.push_back(i);
                        histogramY.push_back(histogramNormalized[i]);
                    }

                    m_IsHistogramUpdated = false;
                }

                if (histogramX.size() > 0)
                {
                    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.15f);
                    ImPlot::PlotLine("Histogram", histogramX.data(), histogramY.data(), histogramX.size());
                    ImPlot::PopStyleVar();
                }
            }

            std::vector<ImVec2> opacity;

            opacity.resize(m_SamplingCount);

            for (uint32_t index = 0; index < m_SamplingCount; index++)
            {
                const float x = (index / static_cast<float>(m_SamplingCount - 1));
                const float y = m_OpacityTransferFunc.Evaluate(x);
                opacity[index] = ImVec2(x * (m_OpacityTransferFunc.PLF.RangeMax - m_OpacityTransferFunc.PLF.RangeMin) + m_OpacityTransferFunc.PLF.RangeMin, y);
            }

            ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
            ImPlot::PlotShaded("Opacity", &opacity[0].x, &opacity[0].y, std::size(opacity), 0, 0, sizeof(ImVec2));
            ImPlot::PopStyleVar();

            ImPlot::PlotLine("Opacity", &opacity[0].x, &opacity[0].y, std::size(opacity), 0, sizeof(ImVec2));

            ImPlot::EndPlot();
        }

        { // 透明度编辑表格
            ImGuiTableFlags flags2 = ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Borders;
            static const int COLUMNS_COUNT = 4;
            if (ImGui::BeginTable("table_opacity", COLUMNS_COUNT, flags2))
            {
                ImGui::TableSetupColumn("ID");
                ImGui::TableSetupColumn("Intensity");
                ImGui::TableSetupColumn("Opacity");

                // [2.1] Right-click on the TableHeadersRow() line to open the default table context menu.
                ImGui::TableHeadersRow();
                for (int row = 0; row < m_OpacityTransferFunc.PLF.Count; row++)
                {
                    ImGui::TableNextRow();
                    for (int column = 0; column < COLUMNS_COUNT; column++)
                    {
                        // Submit dummy contents
                        ImGui::TableSetColumnIndex(column);

                        if (column == 0)
                        {
                            ImGui::Text("%d", row + 1);
                        }
                        else if (column == 1)
                        {
                            static int position = m_OpacityTransferFunc.PLF.Position[row];
                            position = m_OpacityTransferFunc.PLF.Position[row];

                            int vMin = m_OpacityTransferFunc.PLF.RangeMin;
                            int vMax = m_OpacityTransferFunc.PLF.RangeMax;

                            if (row > 1)
                                vMin = m_OpacityTransferFunc.PLF.Position[row - 1];

                            if (row < m_OpacityTransferFunc.PLF.Count - 1)
                                vMax = m_OpacityTransferFunc.PLF.Position[row + 1];

                            ImGui::PushID(row * COLUMNS_COUNT + column);

                            if (ImGui::SliderInt("##pos", &position, vMin, vMax))
                            {
                                m_OpacityTransferFunc.PLF.Position[row] = position;
                                m_IsRecreateOpacityTexture = true;
                            }

                            ImGui::PopID();
                        }
                        else if (column == 2)
                        {
                            float opacity = m_OpacityTransferFunc.PLF.Value[row];
                            ImGui::PushID(row * COLUMNS_COUNT + column);
                            if (ImGui::SliderFloat("##opa", &opacity, 0, 1, "%.2f"))
                            {
                                m_OpacityTransferFunc.PLF.Value[row] = opacity;
                                m_IsRecreateOpacityTexture = true;
                            }
                            ImGui::PopID();
                        }
                        else if (column == COLUMNS_COUNT - 1)
                        {
                            ImGui::PushID(row * COLUMNS_COUNT + column);
                            ImGui::SmallButton("Delete");
                            ImGui::PopID();
                        }
                    }
                }

                int hovered_column = -1;
                for (int column = 0; column < COLUMNS_COUNT + 1; column++)
                {
                    ImGui::PushID(column);
                    if (ImGui::TableGetColumnFlags(column) & ImGuiTableColumnFlags_IsHovered)
                        hovered_column = column;
                    if (hovered_column == column && !ImGui::IsAnyItemHovered() && ImGui::IsMouseReleased(1))
                        ImGui::OpenPopup("MyPopup");
                    if (ImGui::BeginPopup("MyPopup"))
                    {
                        if (column == COLUMNS_COUNT)
                            ImGui::Text("This is a custom popup for unused space after the last column.");
                        else
                            ImGui::Text("This is a custom popup for Column %d", column);
                        if (ImGui::Button("Close"))
                            ImGui::CloseCurrentPopup();
                        ImGui::EndPopup();
                    }
                    ImGui::PopID();
                }

                ImGui::EndTable();
            }

            if (ImGui::Button("Add"))
            {
            }
        }

        { // 颜色点编辑表格
            ImGuiTableFlags flags2 = ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Borders;
            static const int COLUMNS_COUNT = 4;
            if (ImGui::BeginTable("table_colos", COLUMNS_COUNT, flags2))
            {
                ImGui::TableSetupColumn("ID");
                ImGui::TableSetupColumn("Intensity");
                ImGui::TableSetupColumn("Diffuse");
                ImGui::TableSetupColumn("Specular");

                ImGui::TableHeadersRow();
                for (int row = 0; row < m_DiffuseTransferFunc.PLF[0].Count; row++)
                {
                    ImGui::TableNextRow();
                    for (int column = 0; column < COLUMNS_COUNT; column++)
                    {
                        // Submit dummy contents
                        ImGui::TableSetColumnIndex(column);

                        if (column == 0)
                        {
                            ImGui::Text("%d", row + 1);
                        }
                        else if (column == 1)
                        {
                            static int position = m_DiffuseTransferFunc.PLF[0].Position[row];
                            position = m_DiffuseTransferFunc.PLF[0].Position[row];

                            int vMin = m_DiffuseTransferFunc.PLF[0].RangeMin;
                            int vMax = m_DiffuseTransferFunc.PLF[0].RangeMax;

                            if (row > 1)
                                vMin = m_DiffuseTransferFunc.PLF[0].Position[row - 1];

                            if (row < m_DiffuseTransferFunc.PLF[0].Count - 1)
                                vMax = m_DiffuseTransferFunc.PLF[0].Position[row + 1];

                            ImGui::PushID(row * COLUMNS_COUNT + column);

                            if (ImGui::SliderInt("##pos", &position, vMin, vMax))
                            {
                                m_DiffuseTransferFunc.PLF[0].Position[row] = position; // red
                                m_DiffuseTransferFunc.PLF[1].Position[row] = position; // green
                                m_DiffuseTransferFunc.PLF[2].Position[row] = position; // blue
                                m_IsRecreateDiffuseTexture = true;
                            }

                            ImGui::PopID();
                        }
                        else if (column == 2)
                        {
                            double r = m_DiffuseTransferFunc.PLF[0].Value[row];
                            double g = m_DiffuseTransferFunc.PLF[1].Value[row];
                            double b = m_DiffuseTransferFunc.PLF[2].Value[row];

                            ImGui::Text("%.2f,%.2f,%.2f", r, g, b);
                        }
                        else if (column == 3)
                        {
                            double r = m_SpecularTransferFunc.PLF[0].Value[row];
                            double g = m_SpecularTransferFunc.PLF[1].Value[row];
                            double b = m_SpecularTransferFunc.PLF[2].Value[row];

                            ImGui::Text("%.2f,%.2f,%.2f", r, g, b);
                        }
                    }
                }

                ImGui::EndTable();
            }

            if (ImGui::Button("Add##a"))
            {
            }
        }
    }

    if (ImGui::CollapsingHeader("Camera"))
    {
        ImGui::SliderFloat("Rotate sensitivity", &m_RotateSensitivity, 0.1f, 10.0f);
        ImGui::SliderFloat("Zoom sensitivity", &m_ZoomSensitivity, 0.1f, 10.0f);
    }

    if (ImGui::CollapsingHeader("Volume"))
    {
        ImGui::SliderFloat("Density", &m_Density, 0.1f, 100.0f);
        ImGui::SliderInt("Step count", reinterpret_cast<int32_t *>(&m_StepCount), 1, 512);
        m_FrameIndex = ImGui::SliderInt("Mip Level", reinterpret_cast<int32_t *>(&m_MipLevel), 0, m_DimensionMipLevels - 1) ? 0 : m_FrameIndex;
    }

    if (ImGui::CollapsingHeader("Post-Processing"))
        ImGui::SliderFloat("Exposure", &m_Exposure, 4.0f, 100.0f);

    if (ImGui::CollapsingHeader("Debug"))
        ImGui::Checkbox("Show computed tiles", &m_IsDrawDebugTiles);

    ImGui::Checkbox("Show metrics", &isShowAppMetrics);
    ImGui::Checkbox("Show about", &isShowAppAbout);

    ImGui::End();
}

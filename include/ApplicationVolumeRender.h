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

#pragma once

#include "Application.h"
#include "TransferFunction.h"

#include <Hawk/Components/Camera.hpp>
#include <Hawk/Math/Functions.hpp>

class ApplicationVolumeRender final : public Application
{
public:
    using Base = Application;

    ApplicationVolumeRender(ApplicationDesc const &desc);

private:
    void InitializeVolumeTexture();

    void UpdateVolumeTexture(const std::vector<uint16_t> &intensity, int width, int height, int depth);

    void InitializeTransferFunction();

    void InitializeSamplerStates();

    void InitializeShaders();

    void InitializeRenderTextures();

    void InitializeBuffers();

    void InitializeTileBuffer();

    void InitializeEnvironmentMap();

    void Resize(int32_t width, int32_t height) override;

    void EventMouseWheel(float delta) override;

    void EventMouseMove(float x, float y) override;

    void Update(float deltaTime) override;

    void RenderFrame(DX::ComPtr<ID3D11RenderTargetView> pRTV) override;

    void RenderGUI(DX::ComPtr<ID3D11RenderTargetView> pRTV) override;

    void TextureBlit(DX::ComPtr<ID3D11ShaderResourceView> pSrc, DX::ComPtr<ID3D11RenderTargetView> pDst);

    void DrawGridLine(DX::ComPtr<ID3D11RenderTargetView> pDst);

private:
    using D3D11ArrayUnorderedAccessView = std::vector<DX::ComPtr<ID3D11UnorderedAccessView>>;
    using D3D11ArrayShadeResourceView = std::vector<DX::ComPtr<ID3D11ShaderResourceView>>;

    D3D11ArrayShadeResourceView m_pSRVVolumeIntensity;
    D3D11ArrayUnorderedAccessView m_pUAVVolumeIntensity;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVGradient;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVGradient;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVDiffuseTF;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVSpecularTF;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVRoughnessTF;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVOpacityTF;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVEnvironment;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVRadiance;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVRadiance;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVDiffuse;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVSpecular;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVNormal;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVDepth;
    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVColorSum;

    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVDiffuse;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVSpecular;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVNormal;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVDepth;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVColorSum;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVToneMap;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVToneMap;

    DX::ComPtr<ID3D11ShaderResourceView> m_pSRVDispersionTiles;
    DX::ComPtr<ID3D11UnorderedAccessView> m_pUAVDispersionTiles;

    DX::GraphicsPSO m_PSODefault = {};
    DX::GraphicsPSO m_PSOBlit = {};
    DX::GraphicsPSO m_PSODegugTiles = {};
    DX::GraphicsPSO m_PSOGridLine = {};

    DX::ComputePSO m_PSOGeneratePrimaryRays = {};
    DX::ComputePSO m_PSOComputeDiffuseLight = {};

    DX::ComputePSO m_PSOAccumulate = {};
    DX::ComputePSO m_PSOComputeTiles = {};
    DX::ComputePSO m_PSOResetTiles = {};
    DX::ComputePSO m_PSOToneMap = {};
    DX::ComputePSO m_PSOGenerateMipLevel = {};
    DX::ComputePSO m_PSOComputeGradient = {};

    DX::ComPtr<ID3D11SamplerState> m_pSamplerPoint;
    DX::ComPtr<ID3D11SamplerState> m_pSamplerLinear;
    DX::ComPtr<ID3D11SamplerState> m_pSamplerAnisotropic;

    DX::ComPtr<ID3D11Buffer> m_pConstantBufferFrame;
    DX::ComPtr<ID3D11Buffer> m_pDispatchIndirectBufferArgs;
    DX::ComPtr<ID3D11Buffer> m_pDrawInstancedIndirectBufferArgs;
    DX::ComPtr<ID3D11Buffer> m_pGridLineVertexBuffer; // 用于渲染网格线

    DX::ComPtr<ID3D11InputLayout> m_pGridLineInputLayout; // 用于渲染网格线的着色器的输入布局

    ColorTransferFunction1D m_DiffuseTransferFunc;
    ColorTransferFunction1D m_SpecularTransferFunc;
    ColorTransferFunction1D m_EmissionTransferFunc;
    ScalarTransferFunction1D m_RoughnessTransferFunc;
    ScalarTransferFunction1D m_OpacityTransferFunc;

    Hawk::Components::Camera m_Camera = {};

    Hawk::Math::Vec3 m_BoundingBoxMin = Hawk::Math::Vec3(-0.5f, -0.5f, -0.5f);
    Hawk::Math::Vec3 m_BoundingBoxMax = Hawk::Math::Vec3(+0.5f, +0.5f, +0.5f);

    float m_DeltaTime = 0.0f;
    float m_RotateSensitivity = 2.25f;
    float m_ZoomSensitivity = 3.5f;
    float m_Density = 100.0f;
    float m_Exposure = 12.0f;
    float m_Zoom = 1.0f;
    uint32_t m_MipLevel = 0;
    uint32_t m_StepCount = 180;
    uint32_t m_FrameIndex = 0;
    uint32_t m_SampleDispersion = 8;
    uint32_t m_SamplingCount = 256;

    bool m_IsReloadShader = false;
    bool m_IsReloadTransferFunc = false;
    bool m_IsDrawDebugTiles = false;

    uint16_t m_DimensionX = 0;
    uint16_t m_DimensionY = 0;
    uint16_t m_DimensionZ = 0;
    uint16_t m_DimensionMipLevels = 0;

    std::random_device m_RandomDevice;
    std::mt19937 m_RandomGenerator;
    std::uniform_real_distribution<float> m_RandomDistribution;

    std::string m_VolumeDataPath;
    bool m_IsVolumeDataLoaded = false;
    bool m_IsRecreateOpacityTexture = false;
    bool m_IsRecreateDiffuseTexture = false;

    std::vector<double> m_HistogramData;
    bool m_IsHistogramUpdated = false;

    bool m_IsFirstFrameAfterVolumeDataLoad = false;

    Hawk::Math::Mat4x4 m_WVP;

    uint32_t m_GridLineVertexCnt = 2;
};

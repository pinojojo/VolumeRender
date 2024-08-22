#pragma once

/**
 * @brief 硬编码的着色器源代码，避免了文件读写操作
 *
 */

namespace ShaderSource
{
    const char *Accumulation = R"(
    cbuffer ConstantBuffer : register(b0)
    {
        matrix world;
        matrix view;
        matrix projection;
    }

    struct VS_INPUT
    {
        float4 position : POSITION;
        float4 color : COLOR;
    };

    struct PS_INPUT
    {
        float4 position : SV_POSITION;
        float4 color : COLOR;
    };

    PS_INPUT main(VS_INPUT input)
    {
        PS_INPUT output = (PS_INPUT)0;
        output.position = mul(input.position, world);
        output.position = mul(output.position, view);
        output.position = mul(output.position, projection);
        output.color = input.color;
        return output;
    }
    )";

    const char *pixelShader = R"(
    struct PS_INPUT
    {
        float4 position : SV_POSITION;
        float4 color : COLOR;
    };

    float4 main(PS_INPUT input) : SV_TARGET
    {
        return input.color;
    }
    )";
} // namespace ShaderSource
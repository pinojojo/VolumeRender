#include "Common.hlsl"

// 顶点结构体，包含位置和颜色
struct VertexOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

// ID代表第几个要渲染的线段
void RenderLineVS(uint id : SV_VertexID, out VertexOutput output)
{
    float3 unicoord;
    float4 color;

    if (id == 0 || id == 2 || id == 4)
    {
        unicoord = float3(0, 0, 0);

        if (id == 0)
            color = float4(1.0, 0.0, 0.0, 1.0);
        else if (id == 2)
            color = float4(0.0, 1.0, 0.0, 1.0);
        else if (id == 4)
            color = float4(0.0, 0.0, 1.0, 1.0);
    }
    else if (id == 1)
    {
        unicoord = float3(1, 0, 0);
        color = float4(1.0, 0.0, 0.0, 1.0);
    }
    else if (id == 3)
    {
        unicoord = float3(0, 1, 0);
        color = float4(0.0, 1.0, 0.0, 1.0);
    }
    else if (id == 5)
    {
        unicoord = float3(0, 0, 1);
        color = float4(0.0, 0.0, 1.0, 1.0);
    }

    output.position = mul(float4(unicoord, 1.0), FrameBuffer.ViewProjectionMatrix);
    output.color = color;
}

float4 RenderLinePS(VertexOutput input) : SV_TARGET0
{
    return input.color;
}

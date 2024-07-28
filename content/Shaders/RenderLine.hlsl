#include "Common.hlsl"

struct VS_Input
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct GS_INPUT
{
    float4 position : SV_POSITION;   
    float4 color : COLOR;      
};


struct PS_Input
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

GS_INPUT RenderLineVS(VS_Input input)
{
    GS_INPUT output;

    float4 worldPosition = float4(input.position, 1.0f);

    output.position = mul(worldPosition, FrameBuffer.ViewProjectionMatrix);
    output.color = input.color;
    
    return output;
}

[maxvertexcount(2)]
void RenderLineGS(point GS_INPUT input[1], inout LineStream<PS_Input> OutputStream)
{
    PS_Input output;

    // 直接传递第一个顶点
    output.position = input[0].position;
    output.color = input[0].color;
    OutputStream.Append(output);

    // 直接传递第二个顶点
    output.position = input[0].position + float4(1.0f, 0.0f, 0.0f, 0.0f);
    output.color = input[0].color;
    OutputStream.Append(output);

    OutputStream.RestartStrip();  // 重启图元组装，准备下一对顶点（如果有的话）
}

float4 RenderLinePS(PS_Input input) : SV_TARGET
{
    return input.color;
}

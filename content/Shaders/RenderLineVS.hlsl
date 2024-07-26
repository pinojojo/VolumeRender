#include "Common.hlsl"

struct VSInput
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

PSInput main(VSInput input)
{
    PSInput output;

    float4 worldPosition = float4(input.position, 1.0f);
    float4 viewPosition = mul(worldPosition, FrameBuffer.ViewMatrix);
    ouptut.position = mul(viewPosition, FrameBuffer.ProjectionMatrix);
    output.color = input.color;
    return output;
}

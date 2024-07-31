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
    // 确定XYZ三个方向的可见性
    float3 visible;

    float4 xProj = mul(FrameBuffer.WorldViewProjectionMatrix, float4(1.0, 0.0, 0.0, 1.0));
    float4 yProj = mul(FrameBuffer.WorldViewProjectionMatrix, float4(0.0, 1.0, 0.0, 1.0));
    float4 zProj = mul(FrameBuffer.WorldViewProjectionMatrix, float4(0.0, 0.0, 1.0, 1.0));

    visible.x = xProj.z > 0.5 ? 0.5 : -0.5;
    visible.y = yProj.z > 0.5 ? 0.5 : -0.5;
    visible.z = zProj.z > 0.5 ? 0.5 : -0.5;

    int3 ticksCount;
    ticksCount.x = FrameBuffer.GridLineInfo.x;
    ticksCount.y = FrameBuffer.GridLineInfo.y;
    ticksCount.z = FrameBuffer.GridLineInfo.z;

    int order = id % 2;   // 0代表起点，1代表终点
    int lineID = id / 2;  // 一个线段两个顶点
    int groupID = id / 4; // 两个线段为一组(组对应tick)

    int axisID = 0;
    if (groupID < ticksCount.x)
    {
        axisID = 0;
    }
    else if (groupID < ticksCount.x + ticksCount.y)
    {
        axisID = 1;
        groupID -= ticksCount.x;
    }
    else
    {
        axisID = 2;
        groupID -= ticksCount.x + ticksCount.y;
    }

    float3 pos;
    float4 color;
    if (axisID == 0)
    {
        if (lineID % 2 == 0)
        {
            pos.x = FrameBuffer.GridLineInfoStart.x + FrameBuffer.GridLineInfoStep.x * groupID;
            pos.y = 1.0 * (order - 0.5);
            pos.z = visible.z;
        }
        else
        {
            pos.x = FrameBuffer.GridLineInfoStart.x + FrameBuffer.GridLineInfoStep.x * groupID;
            pos.y = visible.y;
            pos.z = 1.0 * (order - 0.5);
        }

        color = float4(1.0, 0.0, 0.0, 1.0);
    }
    else if (axisID == 1)
    {
        if (lineID % 2 == 0)
        {
            pos.x = visible.x;
            pos.y = FrameBuffer.GridLineInfoStart.y + FrameBuffer.GridLineInfoStep.y * groupID;
            pos.z = 1.0 * (order - 0.5);
        }
        else
        {
            pos.x = 1.0 * (order - 0.5);
            pos.y = FrameBuffer.GridLineInfoStart.y + FrameBuffer.GridLineInfoStep.y * groupID;
            pos.z = visible.z;
        }

        color = float4(0.0, 1.0, 0.0, 1.0);
    }
    else
    {
        if (lineID % 2 == 0)
        {
            pos.x = 1.0 * (order - 0.5);
            pos.y = visible.y;
            pos.z = FrameBuffer.GridLineInfoStart.z + FrameBuffer.GridLineInfoStep.z * groupID;
        }
        else
        {
            pos.x = visible.x;
            pos.y = 1.0 * (order - 0.5);
            pos.z = FrameBuffer.GridLineInfoStart.z + FrameBuffer.GridLineInfoStep.z * groupID;
        }

        color = float4(0.0, 0.0, 1.0, 1.0);
    }

    output.position = mul(FrameBuffer.WorldViewProjectionMatrix, float4(pos, 1.0));
    output.color = color;
}

float4 RenderLinePS(VertexOutput input) : SV_TARGET0
{
    return input.color;
}

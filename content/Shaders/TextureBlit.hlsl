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

Texture2D<float4> TextureSrc : register(t0);
SamplerState SamplerPoint : register(s0);

struct VertexOutput
{
    float4 position : SV_Position;
    float2 texcoord : TEXCOORD;
    float depth : TEXCOORD1; // 用于传递深度值
};

void BlitVS(uint id : SV_VertexID, out VertexOutput output)
{
    float2 texcoord = float2((id << 1) & 2, id & 2);
    output.texcoord = texcoord;
    float4 position = float4(texcoord * float2(2, -2) + float2(-1, 1), 0.5, 1);
    output.position = position;

    output.depth = 0.1;
}

struct PixelOutput
{
    float4 color : SV_Target;
    float depth : SV_Depth;
};

PixelOutput BlitPS(VertexOutput input)
{
    PixelOutput output;

    float4 color = TextureSrc.Sample(SamplerPoint, input.texcoord);
    float colorIntensity = color.r * 0.3 + color.g * 0.59 + color.b * 0.11;
    color.a = colorIntensity;

    output.depth = input.depth; // 输出手动设置的深度值
    if (colorIntensity < 0.01)
    {
        output.depth = 1.0; // 如果颜色值过小，设置深度值为1.0
    }

    output.color = color;

    return output;
}

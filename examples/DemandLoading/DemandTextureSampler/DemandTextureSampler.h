// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <OptiXToolkit/DemandLoading/DeviceContext.h>

#define INCLUDE_NTC_TEXTURES 1
const int MAX_TEXTURE_SAMPLES = 512;

enum TextureFunction
{
    TF_tex2D = 0,
    TF_tex2DLod,
    TF_tex2DGrad,
    TF_tex2DGradUdim,
    TF_tex2DGradUdimBlend,
    TF_tex2DCubic,
    TF_tex2DCubicDerivatives,
    TF_tex2DCubicUdim,
    TF_tex2DCubicUdimDerivatives,
    TF_ntcTex2DGrad,
    TF_ntcTex2DGradUdim,
};

struct TextureSample
{
    // Inputs
    int textureId;
    int textureFunction;
    float u, v;
    float2 ddx;
    float2 ddy;
    float lod;
    float2 jitter;

    // Outputs
    bool resident;
    float4 result;    // texture value
    float4 dresultds; // s derivative
    float4 dresultdt; // t derivative
    float out[16];    // values for neural textures
};

struct DemandTextureSamplerLaunchParams
{
    demandLoading::DeviceContext demandTextureContext;
    int numSamples;
    TextureSample* samples;
};


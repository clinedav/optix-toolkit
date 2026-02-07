// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <optix.h>

#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/DemandLoading/Texture2DExtended.h>
#include <OptixToolkit/DemandLoading/Texture2DCubic.h>

#ifdef INCLUDE_NTC_TEXTURES
#include <OptiXToolkit/NeuralTextures/Texture2DNeural.h>
#endif

#include "DemandTextureSampler.h"

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__demandTextureSampler()
{
    int idx = optixGetLaunchIndex().x;
    if( idx >= params.numSamples )
        return;
    
    const demandLoading::DeviceContext dtContext = params.demandTextureContext;
    TextureSample& s = params.samples[idx]; 

    // From Texture2D.h
    if( s.textureFunction == TF_tex2D )
    {
        s.result = tex2D( dtContext, s.textureId, s.u, s.v, &(s.resident), s.jitter );
    }
    else if( s.textureFunction == TF_tex2DLod )
    {
        s.result = tex2DLod( dtContext, s.textureId, s.u, s.v, s.lod, &(s.resident), s.jitter );
    }
    else if( s.textureFunction == TF_tex2DGrad )
    {
        s.result = tex2DGrad( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, &(s.resident), s.jitter );
    }

    // From Texture2DExtended.h
    else if( s.textureFunction == TF_tex2DUdim )
    {
        s.result = tex2DGradUdim( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, &(s.resident), s.jitter )
    }
    else if( s.textureFunction == TF_tex2DGradUdimBlend )
    {
        s.result = tex2DGradUdimBlend( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, &(s.resident), s.jitter )
    }

    // From Texture2DCubic.h
    else if( s.textureFunction == TF_tex2DCubic )
    {
        s.resident = tex2DCubic( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, &s.result, nullptr, nullptr, s.jitter );
    }
    else if( s.textureFunction == TF_tex2DCubicDerivatives )
    {
        s.resident = tex2DCubic( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, nullptr, &s.dresultds, &s.dresultdt, s.jitter );
    }
    else if( s.textureFunction == TF_tex2DCubicUdim )
    {
        s.resident = tex2DCubicUdim( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, &s.result, nullptr, nullptr, s.jitter );
    }
    else if( s.textureFunction == TF_tex2DCubicUdimDerivatives )
    {
        s.resident = tex2DCubicUdim( dContext, s.textureId, s.u, s.v, s.ddx, s.ddy, nullptr, &s.dresultds, &s.dresultdt, s.jitter );
    }

#ifdef INCLUDE_NTC_TEXTURES
    // From Texture2DNeural.h
    else if( s.textureFunction == TF_ntcTex2DGrad )
    {
        T_VEC_OUT_FLOAT out;
        s.resident = ntcTex2DGrad<T_VEC_OUT_FLOAT>( out, dtContext, textureId, s.u, s.v, s.ddx, s.ddy, s.jitter );
        for( int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; i++ )
            s.out[i] = out[i];
    }
    else if( s.textureFunction == TF_ntcTex2DGradUdim )
    {
        T_VEC_OUT_FLOAT out;
        s.resident = ntcTex2DGradUdim<T_VEC_OUT_FLOAT>( out, dtContext, textureId, s.u, s.v, s.ddx, s.ddy, s.jitter );
        for( int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; i++ )
            s.out[i] = out[i];
    }
#endif

}

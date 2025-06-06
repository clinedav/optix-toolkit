// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <optix.h>
#include <optix_device.h>

#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include "SplatTypes.h"
#include "SplatBenchParams.h"
#include "SplatIntersections.h"

extern "C" {
__constant__ OTKAppLaunchParams params;
}

struct SplatBenchRayPayload
{
    int numTests;
    int numSuccessfulTests;
    UVRay uvRay;
    HalfUVRay huvRay;
    IsectList isects;
};

enum RenderModes
{
    FULL_RENDER = 1,
    NUM_ISECT_TESTS = 2
};

OTK_DEVICE const float EPS = 0.0001f;
OTK_DEVICE const float INF = 1e16f;

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

static __forceinline__ __device__ SplatBenchParams* getSplatBenchParams()
{
    return reinterpret_cast<SplatBenchParams*>( &params.extraData );
}

static __forceinline__ __device__ void makeEyeRay( uint2 px, float2 xi, float3& origin, float3& direction )
{
    xi = tentFilter( xi ) + float2{0.5f, 0.5f};
    makeEyeRayPinhole( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
}

static __forceinline__ __device__ void accumulateColor( uint2 px, float4 accumColor )
{
    unsigned int imageIdx = px.y * params.image_dim.x + px.x;
    if( params.subframe != 0 )
        accumColor += params.accum_buffer[imageIdx];
    params.accum_buffer[imageIdx] = accumColor;

    accumColor *= ( 1.0f / accumColor.w );
    params.result_buffer[imageIdx] = make_color( accumColor );

    //unsigned int imageIdx = px.y * params.image_dim.x + px.x;
    //params.result_buffer[imageIdx] = make_color( accumColor );
    //return;
}






template<class TYPE>
static __forceinline__ __device__ void SWAP( TYPE& a, TYPE& b )
{
    TYPE temp = a;
    a = b;
    b = temp;
}

template<class TYPE>
static __forceinline__ __device__ void shell( TYPE* data, int size, int step )
{
    for( int i = step; i < size; ++i )
    {
        int j = i;
        while( j >= step && data[j].tval < data[j - step].tval)
        {
            SWAP<TYPE>( data[j], data[j - step] );
            j -= step;
        }
    }
}

template<class TYPE>
static __forceinline__ __device__ void shellSort( TYPE* data, int size )
{
    //for( int step = size / 2; step >= 23; step = step / 2 )
    //    shell<TYPE>( data, size, step );
    shell<TYPE>( data, size, 10 );
    //shell<TYPE>( data, size, 4 );
    shell<TYPE>( data, size, 1 );
}

//------------------------------------------------------------------------------
// OptiX Programs - Fixed size array with multiple ray casts
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__splatBench()
{
    // Get pixel location
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) ) return;
 
    // Make eye ray
    float3 origin, direction;
    unsigned int rseed = srand( px.x, px.y, params.subframe );
    float2 pixelJitter = (params.subframe == 0) ? float2{0.5f, 0.5f} : float2{rnd(rseed), rnd(rseed)};
    makeEyeRay( px, pixelJitter, origin, direction );

    // Set up per ray data
    SplatBenchRayPayload prd;
    prd.uvRay.init( origin, direction );

    // Trace rays as needed to gather all intersections
    float3 color = {0.0f, 0.0f, 0.0f};
    float alpha = 1.0f;
    float tmin = EPS;
    prd.numTests = 0;
    prd.numSuccessfulTests = 0;

    //for( int i=0; i<4; ++i )
    //{
        prd.isects.alpha = 1.0f;
        prd.isects.numIsects = 0;
        traceRay( params.traversable_handle, origin, direction, tmin, INF, OPTIX_RAY_FLAG_NONE, &prd );
        if( prd.isects.numIsects > 0 )
        {
            //shellSort( (Isect*)prd.isects.isects, (int)prd.isects.numIsects );
            color += integrateColor( prd.isects, alpha );
            tmin = prd.isects.isects[prd.isects.numIsects-1].tval + EPS;
        }
        //if( prd.isects.numIsects < ISECT_LIST_SIZE ) break;
    //} 

    // Display number of intersection tests
    if( params.render_mode == NUM_ISECT_TESTS )
        color = 0.001f * float3{(float)prd.numTests, (float)prd.numSuccessfulTests, (float)0.0f};

    // Accumulate final color
    accumulateColor( px, float4{color.x, color.y, color.z, 1.0f} );
}

extern "C" __global__ void __anyhit__transparency()
{
    optixTerminateRay();
}


extern "C" __global__ void __intersection__sphereSplat()
{
    SplatBenchParams* pbParams = getSplatBenchParams();
    SplatBenchRayPayload* prd = reinterpret_cast<SplatBenchRayPayload*>( getRayPayload() );
    prd->numTests++;

    Isect isect;
    HalfSphereSplat* splat = pbParams->s_splats;
    splat += optixGetPrimitiveIndex();
    if( intersectSphereSplat(prd->uvRay, *splat, isect) )
    {
        prd->numSuccessfulTests++;
        addIsect( prd->isects, isect );
    }
}

extern "C" __global__ void __intersection__ellipsoidSplat()
{
    SplatBenchParams* sbParams = getSplatBenchParams();
    SplatBenchRayPayload* prd = reinterpret_cast<SplatBenchRayPayload*>( getRayPayload() );

    Isect isect;
    EllipsoidSplat* splat = sbParams->e_splats;
    splat += optixGetPrimitiveIndex();
    prd->numTests++;
    if( intersectEllipsoidSplat(prd->uvRay, *splat, isect, sbParams->splatScale) )
    {
        prd->numSuccessfulTests++;
        addIsect( prd->isects, isect );
    }
    
    /*
    for( int i=1; i<=32; ++i)
    {
        prd->numTests++;
        splat += 1;
        if( intersectEllipsoidSplat(prd->uvRay, *splat, isect, sbParams->splatScale) )
        {
            prd->numSuccessfulTests++;
            addIsect( prd->isects, isect );
        }
    }
    */
}

/*
extern "C" __global__ void __intersection__sphereSplatBatch()
{
    const SplatBatchHitGroupData* hgd = reinterpret_cast<HalfSphereSplatBatchHitGroupData*>( optixGetSbtDataPointer() );
    SplatBenchRayPayload* prd = reinterpret_cast<SplatBenchRayPayload*>( getRayPayload() );
    const unsigned int groupIdx = optixGetPrimitiveIndex();

    Isect isect;
    const unsigned int startIdx = hgd->startIdx[groupIdx];
    const unsigned int endIdx = hgd->startIdx[groupIdx+1]; // Need sentinel value
    #pragma unroll
    for( unsigned int idx = startIdx; idx < endIdx; ++idx )
    {
        HalfSphereSplat* splat = &hgd->sphereSplats[idx];
        if( intersectSphereSplat(prd->uvRay, *splat, isect) )
        {
            addIsect( prd->isects, isect );
        }
    }
}
*/

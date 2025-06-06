// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/stochastic_filtering.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <OptiXToolkit/OTKAppBase/OTKAppLaunchParams.h>
#include <OptiXToolkit/OTKAppBase/OTKAppOptixPrograms.h>

#include "TransparencyViewerParams.h"
#include "DeviceVec.h"

OTK_DEVICE const float EPS = 0.0001f;
OTK_DEVICE const float INF = 1e16f;

struct Isect
{
    float tval;
    uchar4 color;
};

__device__ __forceinline__ bool operator < (const Isect& a, const Isect& b )
{
    return a.tval < b.tval;
}

//------------------------------------------------------------------------------
// Render modes
//------------------------------------------------------------------------------

enum RenderModes
{
    FULL_RENDER = 1,
    NORMALS,
    GEOMETRIC_NORMALS,
    TEX_COORDS,
    DDX_LENGTH, 
    CURVATURE,
    DIFFUSE_TEXTURE
};

enum RayTypes
{
    COUNT_ISECTS = 0,
    GATHER_ISECTS
};

//------------------------------------------------------------------------------
// Params - globally visible struct
//------------------------------------------------------------------------------

extern "C" {
__constant__ OTKAppLaunchParams params;
}

//------------------------------------------------------------------------------
// Utility functions
//------------------------------------------------------------------------------

static __forceinline__ __device__ TransparencyViewerParams* getTransparencyViewerParams()
{
    return reinterpret_cast<TransparencyViewerParams*>( &params.extraData );
}

static __forceinline__ __device__ void makeEyeRay( uint2 px, float2 xi, float3& origin, float3& direction )
{
    xi = tentFilter( xi ) + float2{0.5f, 0.5f};
    makeEyeRayPinhole( params.camera, params.image_dim, float2{px.x + xi.x, px.y + xi.y}, origin, direction );
}

static __forceinline__ __device__ Isect getIsect()
{
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float tmax = optixGetRayTmax();
    float3 p = origin + tmax * direction;
    float pdist = length(p);
    uchar4 color = uchar4{(uchar)(255*pdist/2.0f), (uchar)(255*(1.0f-pdist/2.0f)), 50, 8};
    return Isect{tmax, color};
}

template<class ARRAYTYPE>
static __forceinline__ __device__ float3 integrateColor( ARRAYTYPE& isects, int numIsects, float& alpha )
{
    float3 color = float3{0.0f, 0.0f, 0.0f};
    #pragma unroll
    for( int i = 0; i < numIsects; ++i )
    {
        const uchar4 c = isects[i].color;
        color += alpha * (c.w/255.0f) * (1.0f/255.0f) * float3{(float)c.x, (float)c.y, (float)c.z};
        alpha *= (1.0f - c.w/255.0f);
    }
    return color;
}

static __forceinline__ __device__ void accumulateColor( uint2 px, float4 accumColor )
{
    unsigned int imageIdx = px.y * params.image_dim.x + px.x;
    if( params.subframe != 0 )
        accumColor += params.accum_buffer[imageIdx];
    params.accum_buffer[imageIdx] = accumColor;

    accumColor *= ( 1.0f / accumColor.w );
    params.result_buffer[imageIdx] = make_color( accumColor );
}

//------------------------------------------------------------------------------
// Particle intersections
//------------------------------------------------------------------------------

struct Ray
{
    float3 O, U, V, D;
};

struct SphereParticle
{
    float3 C;
    float r;
    float3 color;
    float alpha;
};

static __forceinline__ __device__ bool intersectSphereParticle( Ray r, SphereParticle p, float3& color, float& alpha )
{
    const float ALPHA_THRESHOLD = 0.05f;

    float tca = dot(p.C - r.O, r.D);
    float3 P = (r.O + r.D * tca) - p.C;
    float d2 = dot(P, P) / (p.r * p.r);
    
    //float a = dot(p.C - r.O, r.U);
    //float b = dot(p.C - r.O, r.V);
    //float d2 = (a*a + b*b) / (p.r * p.r);

    alpha = (1.0f - d2) * p.alpha;
    color = alpha * p.color;
    return (alpha > ALPHA_THRESHOLD);
}

struct EllipsoidParticle
{
    float3 Q;
    float3 A, B, C;
    float3 color;
    float alpha;
};

//------------------------------------------------------------------------------
// OptiX Programs - DeviceRingBuffer
//------------------------------------------------------------------------------

#ifdef USE_DEVICE_RING_BUFFER

struct TransparencyRayPayload
{
    int capacity;
    int numIsects;
    Isect* isects;
};

static __forceinline__ __device__ DeviceRingBuffer* getDeviceRingBuffer()
{
    return getTransparencyViewerParams()->ringBuffer;
}

extern "C" __global__ void __raygen__transparency()
{
    // Get pixel location
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) ) return;
    
    // Make eye ray
    float3 origin, direction;
    unsigned int rseed = srand( px.x, px.y, params.subframe );
    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, origin, direction );

    // Set up per ray data
    DeviceRingBuffer* ringBuffer = getDeviceRingBuffer();
    TransparencyRayPayload prd{};
    prd.isects = (Isect*) ringBuffer->alloc( START_ISECT_CAPACITY_FOR_RING_BUFFER * sizeof(Isect) );
    prd.numIsects = 0;
    prd.capacity = START_ISECT_CAPACITY_FOR_RING_BUFFER;

    // Trace initial ray to count intersections
    traceRay( params.traversable_handle, origin, direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &prd );

    // Sort intersections and compute final color
    shellSort<Isect*, Isect>( prd.isects, prd.numIsects );
    float alpha = 1.0f;
    float3 color = integrateColor<Isect*>( prd.isects, prd.numIsects, alpha );
    accumulateColor( px, float4{color.x, color.y, color.z, 1.0f} );
}

extern "C" __global__ void __anyhit__transparency()
{
    TransparencyRayPayload* prd = (TransparencyRayPayload*)getRayPayload();

    if( prd->numIsects >= prd->capacity )
    {
        DeviceRingBuffer* ringBuffer = getDeviceRingBuffer();
        Isect* newIsects = (Isect*) ringBuffer->alloc( prd->capacity * 2 * sizeof(Isect) );
        #pragma unroll
        for( int i=0; i < prd->numIsects; ++i )
            newIsects[i] = prd->isects[i];
        prd->isects = newIsects;
        prd->capacity = prd->capacity * 2;
    }
    prd->isects[prd->numIsects] = getIsect();
    prd->numIsects++;
    optixIgnoreIntersection();
}

#endif // USE_DEVICE_RING_BUFFER

//------------------------------------------------------------------------------
// OptiX Programs - DeviceFixedPool
//------------------------------------------------------------------------------

#ifdef USE_DEVICE_FIXED_POOL

#define ISECT_VEC DeviceVec<Isect, FIXED_POOL_ITEM_SIZE / sizeof(Isect), FIXED_POOL_NUM_POINTERS>

struct TransparencyRayPayload
{
    ISECT_VEC isects; 
};

static __forceinline__ __device__ otk::DeviceFixedPool* getIsectFixedPool()
{
    return getTransparencyViewerParams()->fixedPool;
}

extern "C" __global__ void __raygen__transparency()
{
    // Get pixel location
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) ) return;
    
    // Cast ray
    float3 origin, direction;
    unsigned int rseed = srand( px.x, px.y, params.subframe );
    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, origin, direction );
    TransparencyRayPayload prd{};
    traceRay( params.traversable_handle, origin, direction, EPS, INF, OPTIX_RAY_FLAG_NONE, &prd );

    // Sort intersections and compute final color
    shellSort<ISECT_VEC, Isect>( prd.isects, prd.isects.size );
    float alpha = 1.0f;
    float3 color = integrateColor<ISECT_VEC>( prd.isects, prd.isects.size, alpha );
    accumulateColor( px, float4{color.x, color.y, color.z, 1.0f} );
    prd.isects.clear( *getIsectFixedPool() );
}

extern "C" __global__ void __anyhit__transparency()
{
    TransparencyRayPayload* prd = (TransparencyRayPayload*)getRayPayload();
    Isect isect = getIsect();
    prd->isects.add( isect, *getIsectFixedPool() );
    optixIgnoreIntersection();
}

#endif // USE_DEVICE_FIXED_POOL

//------------------------------------------------------------------------------
// OptiX Programs - Fixed size array with multiple ray casts
//------------------------------------------------------------------------------

#ifdef USE_FIXED_ARRAY

struct TransparencyRayPayload
{
    int numIsects;
    Isect isects[FIXED_ARRAY_SIZE];
};

extern "C" __global__ void __raygen__transparency()
{
    // Get pixel location
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) ) return;
 
    // Make eye ray
    float3 origin, direction;
    unsigned int rseed = srand( px.x, px.y, params.subframe );
    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, origin, direction );

    // Set up per ray data
    TransparencyRayPayload prd;
    float3 color = {0.0f, 0.0f, 0.0f};
    float alpha = 1.0f;
    float tmin = EPS;

    // Trace rays as needed to gather all intersections
    do {
        prd.numIsects = 0;
        traceRay( params.traversable_handle, origin, direction, tmin, INF, OPTIX_RAY_FLAG_NONE, &prd );
        if( prd.numIsects > 0 )
        {
            color += integrateColor( prd.isects, prd.numIsects, alpha );
            tmin = prd.isects[prd.numIsects-1].tval + EPS;
        }
    } while( prd.numIsects >= FIXED_ARRAY_SIZE );

    // Accumulate final color
    accumulateColor( px, float4{color.x, color.y, color.z, 1.0f} );
}

extern "C" __global__ void __anyhit__transparency()
{
    TransparencyRayPayload* prd = (TransparencyRayPayload*)getRayPayload();
    Isect isect = getIsect();

    int idx = prd->numIsects - 1;
    while( idx >= 0 )
    {
        if( prd->isects[idx].tval <= isect.tval )
            break;
        if( idx < FIXED_ARRAY_SIZE - 1 )
            prd->isects[idx + 1] = prd->isects[idx];
        idx--;
    }

    if( idx < FIXED_ARRAY_SIZE - 1 )
        prd->isects[idx + 1] = isect;
    if( prd->numIsects < FIXED_ARRAY_SIZE )
        prd->numIsects++;

    optixIgnoreIntersection();
}

#endif // USE_FIXED_ARRAY

//------------------------------------------------------------------------------
// OptiX Programs - Ring Bbffer (manually) interleaved
//------------------------------------------------------------------------------

#ifdef USE_DEVICE_RING_BUFFER_INTERLEAVED

struct TransparencyRayPayload
{
    int numIsects;
    Isect* isects;
};

template<class ARRAYTYPE>
static __forceinline__ __device__ float3 integrateColor32( ARRAYTYPE& isects, int numIsects, float& alpha )
{
    float3 color = float3{0.0f, 0.0f, 0.0f};
    #pragma unroll
    for( int i = 0; i < numIsects; ++i )
    {
        const uchar4 c = isects[i*WARP_SIZE].color;
        color += alpha * (c.w/255.0f) * (1.0f/255.0f) * float3{(float)c.x, (float)c.y, (float)c.z};
        alpha *= (1.0f - c.w/255.0f);
    }
    return color;
}

extern "C" __global__ void __raygen__transparency()
{
    // Get pixel location
    uint2 px = getPixelIndex( params.num_devices, params.device_idx );
    if( !pixelInBounds( px, params.image_dim ) ) return;
    
    // Make eye ray
    float3 origin, direction;
    unsigned int rseed = srand( px.x, px.y, params.subframe );
    makeEyeRay( px, float2{rnd(rseed), rnd(rseed)}, origin, direction );

    // Set up per ray data
    TransparencyRayPayload prd;
    float3 color = {0.0f, 0.0f, 0.0f};
    float alpha = 1.0f;
    float tmin = EPS;
    prd.isects = (Isect*)getTransparencyViewerParams()->ringBuffer->alloc( FIXED_ARRAY_SIZE * sizeof(Isect) );

    // Trace rays as needed to gather all intersections
    do {
        prd.numIsects = 0;
        traceRay( params.traversable_handle, origin, direction, tmin, INF, OPTIX_RAY_FLAG_NONE, &prd );
        if( prd.numIsects > 0 )
        {
            //color += float3{0.2f, 0.2f, 0.2f};
            color += integrateColor32( prd.isects, prd.numIsects, alpha );
            tmin = prd.isects[(prd.numIsects-1)*32].tval + EPS;
        }
    } while( prd.numIsects >= FIXED_ARRAY_SIZE );

    // Accumulate final color
    accumulateColor( px, float4{color.x, color.y, color.z, 1.0f} );
}

extern "C" __global__ void __anyhit__transparency()
{
    TransparencyRayPayload* prd = (TransparencyRayPayload*)getRayPayload();
    Isect isect = getIsect();

    int idx = prd->numIsects - 1;
    while( idx >= 0 )
    {
        if( prd->isects[idx*WARP_SIZE].tval <= isect.tval )
            break;
        if( idx < FIXED_ARRAY_SIZE - 1 )
            prd->isects[(idx + 1)*WARP_SIZE] = prd->isects[idx*WARP_SIZE];
        idx--;
    }

    if( idx < FIXED_ARRAY_SIZE - 1 )
        prd->isects[(idx + 1)*WARP_SIZE] = isect;
    if( prd->numIsects < FIXED_ARRAY_SIZE )
        prd->numIsects++;

    optixIgnoreIntersection();
}

#endif // USE_DEVICE_RING_BUFFER_INTERLEAVED
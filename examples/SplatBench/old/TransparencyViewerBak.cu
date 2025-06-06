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
    FULL_RENDER = 1
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
// Sphere Particles
//------------------------------------------------------------------------------

struct half3
{
    half x, y, z;
};

struct Ray
{
    float3 O, D, U, V;
};

// 32 bytes
struct SphereParticle
{
    float3 C;
    float r;
    float3 color;
    float alpha;
};

// 16 bytes
/*
struct HalfSphereParticle
{
    half4 C; // center + radius
    half4 color; // color + alpha
};
*/

// 8 bytes
struct PackedSphereParticle
{
    uchar4 C; // center + radius
    uchar4 color; // color + alpha
};

static __forceinline__ __device__ bool intersectSphereParticle( Ray r, SphereParticle p, Isect& isect )
{
    float tca = dot(p.C - r.O, r.D);
    float3 P = (r.O + r.D * tca) - p.C;
    float d2 = dot(P, P) / (p.r * p.r);
    
    if( d2 < 1.0f )
    {
        float alpha = (1.0f - d2) * p.alpha;
        float3 color = p.color;
        isect.tval = tca;
        isect.color = uchar4{ (uchar)(color.x*255), (uchar)(color.y*255), (uchar)(color.z*255), (uchar)(alpha*255) };
        return true;
    }
    return false;
}

static __forceinline__ __device__ bool intersectHalfSphereParticle( Ray r, HalfSphereParticle p, Isect& isect )
{
    float3 pC = float3{(float)p.C.x, (float)p.C.y, (float)p.C.z};
    float invr2 = float(p.C.w);

    float a = dot(pC, r.U) - dot(r.O, r.U);
    float b = dot(pC, r.V) - dot(r.O, r.V);
    float d2 = (a*a + b*b) * invr2;

    if( d2 < 1.0f )
    {
        float tca = dot(pC - r.O, r.D);
        float alpha = (1.0f - d2) * (float)p.color.w;
        float3 color = float3{(float)p.color.x, (float)p.color.y, (float)p.color.z};
        isect.tval = tca;
        isect.color = uchar4{ (uchar)(color.x*255), (uchar)(color.y*255), (uchar)(color.z*255), (uchar)(alpha*255) };
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------
// Half sphere particles (CoopVec)
//------------------------------------------------------------------------------
/*
static __forceinline__ __device__ 
unsigned int intersectHalfSphereParticles( char* particleBase, int particleBatchOffset, HalfRay r, half ROdotRU, half ROdotRV )
{
    const int NPARTICLES = 32;
    const int centerOffset = 0;
    const int invR2Offset = 6 * NPARTICLES;
    #define T_HALF OPTIX_COOP_VEC_ELEM_TYPE_HALF
    #define ROW_MAJOR OPTIX_COOP_VEC_ROW_MAJOR

    //                                      outType    inType    inElem  matLayout  transpose outSize     inSize matElem biasElem
    #define COOP_MAT_MUL optixCoopVecMatMul<T_VEC_OUT, T_VEC_IN, T_HALF, ROW_MAJOR, false,    NPARTICLES, 3,     T_HALF, T_HALF>

    using T_VEC_IN = OptixCoopVec<T_IN, N_IN>;

    using T_VEC_IN = OptixCoopVec<half, 3>;
    using T_VEC_OUT = OptixCoopVec<half, NPARTICLES>;

    T_VEC_IN U_vec; 
    U_vec[0]=ray.U.x; U_vec[1]=ray.U.y; U_vec[2]=ray.U.z; // Does this need to have 4 components?
    T_VEC_IN V_vec; 
    V_vec[0]=ray.V.x; V_vec[1]=ray.V.y; V_vec[2]=ray.V.z;
    T_VEC_OUT ROdotRU_vec( ROdotRU );
    T_VEC_OUT ROdotRV_vec( RVdotRU );
    T_VEC_OUT invR2_vec = optixCoopVecLoad<T_VEC_OUT>( particleBase + particleBatchOffset + invR2Offset );

    T_VEC_OUT A_vec = COOP_MAT_MUL( U_vec, particleBase, particleBatchOffset, 0, 0 );
    A_vec = optixCoopVecSubtract<T_VEC_OUT>( A_vec, ROdotRU_vec );
    A_vec = optixCoopVecMul<T_VEC_OUT>( A_vec, A_vec );

    T_VEC_OUT B_vec = COOP_MAT_MUL( V_vec, particleBase, particleBatchOffset, 0, 0 );
    B_vec = optixCoopVecSubtract<T_VEC_OUT>( B_vec, ROdotRV_vec );
    B_vec = optixCoopVecMul<T_VEC_OUT>( B_vec, B_vec );

    A_vec = optixCoopVecAdd<T_VEC_OUT>( A_vec, B_vec );
    A_vec = optixCoopVecMul<T_VEC_OUT>( A_vec, invR2_vec );

    unsigned int rval = 0;
    #pragma unroll
    for( int i=0; i<NPARTICLES; ++i )
    {
        rval += ((unsigned int)(A_vec[i] < (half)1.0f)) << i;
    }
    return rval;

    //A = [Centers] * U - dotRO_RU
    //A = A * A
    //B = [Centers] * V - dotRO_RV
    //B = B * B
    //AB = A + B
    //AB = AB * [invR2]
}
*/

//------------------------------------------------------------------------------
// Ellipsoid Particles
//------------------------------------------------------------------------------

// 64 bytes
struct EllipsoidParticle
{
    float3 Q; // center
    float3 A, B, C;
    float3 color;
    float alpha;
};

// 32 bytes
struct HalfEllipsoidParticle
{
    half3 Q, A, B, C;
    half4 color;
};

// 16 bytes
struct PackedEllipsoidParticle
{
    // Center
    uint x : 11;
    uint y : 11;
    uint z : 10;

    // Packed A orientation
    int theta : 11;
    int phi : 10;
    int alpha : 11;
    
    // Packed A, B, C lengths
    uint a : 11;
    uint b : 11;
    uint c : 10;

    uchar4 color;
};

static __forceinline__ __device__ bool intersectEllipsoidParticle( Ray ray, EllipsoidParticle ep, Isect& isect )
{
    float3 O = (ray.O - ep.Q);
    O = float3{dot(O, ep.A), dot(O, ep.B), dot(O, ep.C)};

    float3 D = ray.D;
    D = float3{dot(D, ep.A), dot(D, ep.B), dot(D, ep.C)};

    float tca = -dot(O, D) / dot(D, D);
    float3 Pca = O + tca * D;

    if( dot(Pca, Pca) < 1.0f )
    {
        float alpha = (1.0f - dot(Pca, Pca)) * ep.alpha;
        float3 color = ep.color;
        isect.tval = tca;
        isect.color = uchar4{(uchar)(color.x*255), (uchar)(color.y*255), (uchar)(color.z*255), (uchar)(alpha*255)};
        return true;
    }
    return false;
}

//------------------------------------------------------------------------------
// OptiX Programs - Fixed size array with multiple ray casts
//------------------------------------------------------------------------------

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
    TransparencyViewerParams* tvp = getTransparencyViewerParams();

    Ray ray = Ray{};
    ray.O = optixGetWorldRayOrigin();
    ray.D = optixGetWorldRayDirection();
    ray.U = normalize( cross(ray.D, float3{1.0f, 1.0f, 1.0f}) );
    ray.V = normalize( cross(ray.U, ray.D) );

    int numSpheres = 7*7*7;
    for( int i=0; i<numSpheres; ++i )
    {
        Isect isect;
        const HalfSphereParticle sp = tvp->halfSphereParticles[i];
        bool success = intersectHalfSphereParticle( ray, sp, isect );
        if( success )
        {
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
        }
    }

    optixIgnoreIntersection();
}

//------------------------------------------------------------------------------
// OptiX Programs - Each thread handles 2x2 pixels
//------------------------------------------------------------------------------
/*
struct ColorF 
{
    float4 color[4];
};

struct IsectF
{
    float tval;
    uchar4 color[4];
};

struct TransparencyRayPayloadF
{
    int numIsects;
    Ray rays[4];
    IsectF isects[FIXED_ARRAY_SIZE];
};

static __forceinline__ __device__ void makeEyeRayF( uint2 px, uint2 imageDim, Ray& ray )
{
    makeEyeRayPinhole( params.camera, imageDim, float2{px.x + 0.5f, px.y + 0.5f}, ray.O, ray.D );
    ray.U = normalize( cross(ray.D, float3{1.0f, 1.0f, 1.0f}) );
    ray.V = normalize( cross(ray.U, ray.D) );
}

static __forceinline__ __device__ void integrateSingleColor( float4& color, uchar4 incolor )
{
    float t = (color.w * incolor.w) / (255.0f*255.0f);
    color.x += incolor.x * t;
    color.y += incolor.y * t;
    color.z += incolor.z * t;
    color.w *= (1.0f - incolor.w/255.0f);
}

static __forceinline__ __device__ void integrateColorF( IsectF* isects, int numIsects, ColorF& c4 )
{
    for( int i = 0; i < numIsects; ++i )
    {
        integrateSingleColor( c4.color[0], isects[i].color[0] );
        integrateSingleColor( c4.color[1], isects[i].color[1] );
        integrateSingleColor( c4.color[2], isects[i].color[2] );
        integrateSingleColor( c4.color[3], isects[i].color[3] );
    }
}

static __forceinline__ __device__ void accumulateColorF( uint2 px, float4 accumColor )
{
    unsigned int imageIdx = px.y * params.image_dim.x + px.x;
    if( params.subframe != 0 )
        accumColor += params.accum_buffer[imageIdx];
    params.accum_buffer[imageIdx] = accumColor;

    accumColor *= ( 1.0f / accumColor.w );
    params.result_buffer[imageIdx] = make_color( accumColor );
}

extern "C" __global__ void __raygen__transparency()
{
    // Get pixel location
    uint2 launchpx = getPixelIndex( params.num_devices, params.device_idx );
    uint2 px = launchpx * 2;
    if( !pixelInBounds( px, params.image_dim ) ) return;
 
    // Make eye rays
    TransparencyRayPayloadF prd;
    makeEyeRayF( px, params.image_dim, prd.rays[0] );
    makeEyeRayF( px + uint2{1,0}, params.image_dim, prd.rays[1] );
    makeEyeRayF( px + uint2{0,1}, params.image_dim, prd.rays[2] );
    makeEyeRayF( px + uint2{1,1}, params.image_dim, prd.rays[3] );

    // Set up per ray data
    ColorF c4{};
    c4.color[0].w = c4.color[1].w = c4.color[2].w = c4.color[3].w = 1.0f;
    float tmin = EPS;

    // Trace rays as needed to gather all intersections
    do {
        prd.numIsects = 0;
        traceRay( params.traversable_handle, prd.rays[0].O, prd.rays[0].D, tmin, INF, OPTIX_RAY_FLAG_NONE, &prd );
        if( prd.numIsects > 0 )
        {
            integrateColorF( prd.isects, prd.numIsects, c4 );
            tmin = prd.isects[prd.numIsects-1].tval + EPS;
        }
    } while( prd.numIsects >= FIXED_ARRAY_SIZE );

    // Accumulate final color
    accumulateColorF( px, float4{c4.color[0].x, c4.color[0].y, c4.color[0].z, 1.0f});
    accumulateColorF( px + uint2{1,0}, float4{c4.color[1].x, c4.color[1].y, c4.color[1].z, 1.0f});
    accumulateColorF( px + uint2{0,1}, float4{c4.color[2].x, c4.color[2].y, c4.color[2].z, 1.0f});
    accumulateColorF( px + uint2{1,1}, float4{c4.color[3].x, c4.color[3].y, c4.color[3].z, 1.0f});
}

extern "C" __global__ void __anyhit__transparency()
{
    TransparencyRayPayloadF* prd = (TransparencyRayPayloadF*)getRayPayload();
    TransparencyViewerParams* tvp = getTransparencyViewerParams();

    int numSpheres = 7*7*7;
    for( int i=0; i<numSpheres; ++i )
    {
        Isect isect;
        const HalfSphereParticle sp = tvp->halfSphereParticles[i];
        bool success = intersectHalfSphereParticle( prd->rays[0], sp, isect );
        if( success )
        {
            IsectF isectF;
            isectF.tval = isect.tval;
            isectF.color[0] = isect.color;
            intersectHalfSphereParticle( prd->rays[1], sp, isect );
            isectF.color[1] = isect.color;
            intersectHalfSphereParticle( prd->rays[2], sp, isect );
            isectF.color[2] = isect.color;
            intersectHalfSphereParticle( prd->rays[3], sp, isect );
            isectF.color[3] = isect.color;

            int idx = prd->numIsects - 1;
            while( idx >= 0 )
            {
                if( prd->isects[idx].tval <= isectF.tval )
                    break;
                if( idx < FIXED_ARRAY_SIZE - 1 )
                    prd->isects[idx + 1] = prd->isects[idx];
                idx--;
            }

            if( idx < FIXED_ARRAY_SIZE - 1 )
                prd->isects[idx + 1] = isectF;
            if( prd->numIsects < FIXED_ARRAY_SIZE )
                prd->numIsects++;
        }
    }

    optixIgnoreIntersection();
}
*/
//------------------------------------------------------------------------------
// Boneyard
//------------------------------------------------------------------------------

/*
for( int i=-3; i<=3; ++i )
{
    float3 clr = float3{ (i+3)/6.0f, 1.0f-(i+3)/6.0f, 0.5f };
    for( int j=-3; j<=3; ++j )
    {
        for( int k=-3; k<=3; ++k )
        {
            SphereParticle sp;
            sp.C = float3{(float)i, (float)j, (float)k};
            sp.r = 0.5f * (j+4) / 6.0f;
            sp.color = clr;
            sp.alpha = 0.1f;
            Isect isect{};
            bool success = intersectSphereParticle( ray, sp, isect );
            
            EllipsoidParticle ep;
            ep.Q = float3{(float)i, (float)j, (float)k};
            ep.A = float3{6.0f / (0.5f * (j+4)), 6.0f / (0.5f * (j+4)), 0.0f};
            ep.B = float3{2.0f, -2.0f, 0.0f};
            ep.C = float3{0.0f, 0.0f, 2.0f};
            ep.color = clr;
            ep.alpha = 0.1f;
            Isect isect{};
            bool success = intersectEllipsoidParticle( ray, ep, isect );

            HalfSphereParticle sp;
            float invr2 = 1.0f / (0.5f * (j+4) / 6.0f);
            invr2 *= invr2;
            sp.C = half4{(half)i, (half)j, (half)k, invr2};
            sp.color = half4{(half)clr.x, (half)clr.y, (half)clr.z, (half)0.1f};
            Isect isect{};
            bool success = intersectHalfSphereParticle( ray, sp, isect );

            if( success )
            {
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
            }
        }
    }
}
*/

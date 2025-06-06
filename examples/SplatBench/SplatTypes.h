// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#ifdef __CUDACC__
#define HDINLINE __forceinline__ __device__
#define DINLINE __forceinline__ __device__
#else
#define HDINLINE inline
#endif

#define GAUSSIAN_SQUARED 1
//#define RAY_TRACING_ORDER 1

const float ALPHA_MIN = 0.01f;
const float INV_255   = 1.0f / 255.0f;

//------------------------------------------------------------------------------
// half3
//------------------------------------------------------------------------------

struct half3
{
    half x, y, z;
    HDINLINE half3() {}
    HDINLINE half3( half _x, half _y, half _z ) { x = _x; y = _y; z = _z; }
    HDINLINE half3( float _x, float _y, float _z ) { x = (half)_x; y = (half)_y; z = (half)_z; }
    HDINLINE half3( float3 a ) { x = (half)a.x; y = (half)a.y; z = (half)a.z; }
};

HDINLINE half hdot(half3 a, half3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
HDINLINE half hdot(half4 a, half3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

//HDINLINE float dot(float3 a, half3 b) { return a.x*(float)b.x + a.y*(float)b.y + a.z*(float)b.z; }
//HDINLINE float dot(half3 a, float3 b) { return dot(b,a); }

HDINLINE half3 operator - (half4 a, half3 b) { return half3{a.x-b.x, a.y-b.y, a.z-b.z}; }
HDINLINE half3 operator - (half3 a, half3 b) { return half3{a.x-b.x, a.y-b.y, a.z-b.z}; }

HDINLINE float3 make_float3(half3 a) { return float3{(float)a.x, (float)a.y, (float)a.z}; }
HDINLINE float3 make_float3(half4 a) { return float3{(float)a.x, (float)a.y, (float)a.z}; }
HDINLINE float3 make_float3(uchar4 a) { return float3{(float)a.x, (float)a.y, (float)a.z}; }

//------------------------------------------------------------------------------
// Quaternion functions
//------------------------------------------------------------------------------

HDINLINE void getBasisFromNormalizedQuaternion( const float4& Q, float3& A, float3& B, float3& C )
{
    float x=Q.x, y=Q.y, z=Q.z, w=Q.w;
    A = float3{1.0f-2.0f*(y*y+z*z), 2.0f*(x*y+z*w), 2.0f*(x*z-y*w)};
    B = float3{2.0f*(x*y-z*w), 1.0f-2.0f*(x*x+z*z), 2.0f*(y*z+x*w)};
    C = float3{2.0f*(x*z+y*w), 2.0f*(y*z-x*w), 1.0f-2.0f*(x*x+y*y)};
}

HDINLINE float4 SRGBAtoRGBA( const float4 c )
{
    float r = powf( clamp(c.x, 0.0f, 1.0f), 2.2f );
    float g = powf( clamp(c.y, 0.0f, 1.0f), 2.2f );
    float b = powf( clamp(c.z, 0.0f, 1.0f), 2.2f );
    float a = c.w;
    return float4{r, g, b, a};
}

//------------------------------------------------------------------------------
// Intersection and ray payload
//------------------------------------------------------------------------------

struct Isect
{
    float tval;
    uchar4 color;
    
    HDINLINE float getAlpha() const { return color.w * INV_255; }
};

#define ISECT_LIST_SIZE 64

struct IsectList
{
    float alpha;
    int numIsects;
    Isect isects[ISECT_LIST_SIZE];
};

HDINLINE void addIsect( IsectList& ilist, const Isect isect )
{
    // If (nearly) opaque and the new isect is behind the list, discard it
    int idx = ilist.numIsects - 1;
    if( (ilist.alpha <= ALPHA_MIN) && (idx >= 0) && (ilist.isects[idx].tval < isect.tval) )
        return;

    while( idx >= 0 )
    {
        if( ilist.isects[idx].tval <= isect.tval )
            break;
        if( idx < ISECT_LIST_SIZE - 1 )
            ilist.isects[idx + 1] = ilist.isects[idx];
        idx--;
    }
    
    if( idx < ISECT_LIST_SIZE - 1 )
        ilist.isects[idx + 1] = isect;
    if( ilist.numIsects < ISECT_LIST_SIZE )
        ilist.numIsects++;
    
    // Udate the list alpha
    idx = ilist.numIsects - 1;
    ilist.alpha *= (1.0f - isect.getAlpha()*0.99f);
    if( ilist.alpha > ALPHA_MIN ) 
        return;

    // Remove splats that won't be seen becuase the splats before them are opaque
    if( idx <= 0 || ilist.alpha > ALPHA_MIN * (1.0f - ilist.isects[idx].getAlpha()*0.99f) )
        return;
    ilist.numIsects--;
    ilist.alpha /= (1.0f - ilist.isects[idx].getAlpha()*0.99f);
    idx--;

    if( idx <= 0 || ilist.alpha > ALPHA_MIN * (1.0f - ilist.isects[idx].getAlpha()*0.99f) )
        return;
    ilist.numIsects--;
    ilist.alpha /= (1.0f - ilist.isects[idx].getAlpha()*0.99f);
}

HDINLINE float3 integrateColor( IsectList& ilist, float alpha )
{
    float3 color = float3{0.0f, 0.0f, 0.0f};
    //#pragma unroll
    for( int i = 0; i < ilist.numIsects; ++i )
    {
        const Isect& isect = ilist.isects[i];
        const float isectAlpha = isect.getAlpha();
        color += ( alpha * isectAlpha ) * make_float3( isect.color );
        alpha *= ( 1.0f - isectAlpha );
    }
    return color * INV_255;
}

//------------------------------------------------------------------------------
// UV rays - rays with perpendicular axes and precomputed dot products
//------------------------------------------------------------------------------

struct UVRay
{
    float3 O, D, U, V;
    float OdotU, OdotV;

    HDINLINE UVRay() {}
    HDINLINE UVRay( float3 O_, float3 D_ ) { init(O_, D_); }
    HDINLINE void init( float3 O_, float3 D_ )
    {
        O = O_;
        D = D_;
        U = normalize( cross(D, float3{1.0f, 1.0f, 1.0f}) );
        V = normalize( cross(U, D) );
        OdotU = dot(O, U);
        OdotV = dot(O, V);
    }
};

struct HalfUVRay
{
    half3 O, D, U, V;
    float OdotU, OdotV;
};

//------------------------------------------------------------------------------
// Sphere Splats
//------------------------------------------------------------------------------

struct SphereSplat
{
    float4 center; // center, 1/r
    uchar4 color;  // rgba 
};

struct HalfSphereSplat
{
    half4 center; // center, 1/r
    uchar4 color; // rgba
};

struct ByteSphereSplat
{
    uchar4 center; // center, 1/r
    uchar4 color;  // rgba
};

//------------------------------------------------------------------------------
// Sphere Splat Group
//------------------------------------------------------------------------------

#define SPHERE_SPLATS_PER_GROUP 32

struct HalfSphereSplatGroup
{
    half x[SPHERE_SPLATS_PER_GROUP];
    half y[SPHERE_SPLATS_PER_GROUP];
    half z[SPHERE_SPLATS_PER_GROUP];
    half invr2[SPHERE_SPLATS_PER_GROUP];
    uchar4 color[SPHERE_SPLATS_PER_GROUP];

    HDINLINE void setSplat( HalfSphereSplat p, int k )
    {
        x[k] = p.center.x; 
        y[k] = p.center.y; 
        z[k] = p.center.z;
        invr2[k] = p.center.w * p.center.w;
        color[k] = p.color;
    }

    HDINLINE HalfSphereSplat getSplat( int k )
    {
        return HalfSphereSplat{ half4{x[k], y[k], z[k], invr2[k]}, color[k] };
    }
};

//------------------------------------------------------------------------------
// Ellipsoid splats
//------------------------------------------------------------------------------


struct GSEllipsoidSplat
{
    float3 center;
    float3 scale;
    float4 rot;
    float4 color;
};

struct EllipsoidSplat
{
    float3 Q, A, B, C; // Center and axes
    uchar4 color;
};

struct HalfEllipsoidSplat
{
    half3 Q, A, B, C;
    uchar4 color;
};

struct uint34
{
    uint x : 11;
    uint y : 10;
    uint z : 11;
};

struct int34
{
    int x : 11;
    int y : 10;
    int z : 11;
};

struct PIntEllipsoidSplat
{
    uint34 Q;
    int34 A, B, C;
    uchar4 color;
};

struct PackedEllipsoidSplat
{
    // Center
    uint x : 11;
    uint y : 10;
    uint z : 11;

    // Packed quaternion
    uint rx : 2;
    int r0 : 10;
    int r1 : 10;
    int r2 : 10;
    
    // Packed A, B, C lengths
    uint a : 11;
    uint b : 10;
    uint c : 11;

    uchar4 color;
};

/*
struct TinyEllipsoid10
{
    x,y,z: 999: 
    r0,r1,r2: 666: 
    a,b,c: 555: 
    r,g,b,a: 5555: 
}
*/

HDINLINE OptixAabb getAabb( EllipsoidSplat& ep )
{
    float a = sqrtf( log((ep.color.w+ALPHA_MIN*255.0f)/(ALPHA_MIN*255.0f)) );
#ifdef GAUSSIAN_SQUARED
    a = sqrtf( a );
#endif
    
    float3 Q = ep.Q;
    float3 A = ep.A * ( a / dot( ep.A, ep.A ) );
    float3 B = ep.B * ( a / dot( ep.B, ep.B ) );
    float3 C = ep.C * ( a / dot( ep.C, ep.C ) );
    
    float dx = sqrtf( A.x*A.x + B.x*B.x + C.x*C.x );
    float dy = sqrtf( A.y*A.y + B.y*B.y + C.y*C.y );
    float dz = sqrtf( A.z*A.z + B.z*B.z + C.z*C.z );
    return OptixAabb{ Q.x-dx, Q.y-dy, Q.z-dz, Q.x+dx, Q.y+dy, Q.z+dz };
}

//------------------------------------------------------------------------------
// LSS
//------------------------------------------------------------------------------

struct LSSSplat
{
    half rot, c; // a and b from LSS data
    uchar4 color;
    // Alternate method. Steal low order bits from lss radii for rot and c.
};


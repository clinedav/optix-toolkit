// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "SplatTypes.h"

//------------------------------------------------------------------------------
// Sphere splat intersections
//------------------------------------------------------------------------------

static __forceinline__ __device__ bool intersectSphereSplat( UVRay r, HalfSphereSplat p, Isect& isect )
{
    const float3 pC = make_float3( p.center );
    const float invr = float( p.center.w );

    float a = dot(pC, r.U) - r.OdotU;
    float b = dot(pC, r.V) - r.OdotV;
    float d2 = ((a*a + b*b) * invr) * invr;
    if( d2 >= 1.0f ) 
        return false;

    isect.tval = dot(pC - r.O, r.D);
    if( isect.tval <= 0.0f ) // FIXME: Shouldn't this be tmin instead of 0?
        return false;
    isect.color = p.color;
    // Is 9.0 right? This makes 3 std deviations. 
    isect.color.w = (unsigned char)( expf(-0.5f*9.0f*d2) * (float)p.color.w );
    return isect.color.w > 0; //true;
}

//------------------------------------------------------------------------------
// Ellipsoid splat intersections
//------------------------------------------------------------------------------

static __forceinline__ __device__ bool intersectEllipsoidSplat( UVRay ray, EllipsoidSplat& p, Isect& isect, float splatScale )
{
    float3 O = (ray.O - p.Q);
    O = float3{dot(O, p.A), dot(O, p.B), dot(O, p.C)} * splatScale;
    float3 D = ray.D;
    D = float3{dot(D, p.A), dot(D, p.B), dot(D, p.C)} * splatScale;

    float tca = -dot(O, D) / dot(D, D);
    if( tca < 0.0f )
        return false;

    float3 Pca = O + tca * D;
    float d2 = dot(Pca, Pca);
#ifdef GAUSSIAN_SQUARED
    d2 *= d2;
#endif
    if( d2 > 4.6f )
        return false;

    int alpha = (unsigned char)( expf(-d2) * (float)p.color.w );
    if( alpha <= 1 )
        return false;

#ifdef RAY_TRACING_ORDER
    isect.tval = tca;
#else // Gaussian Splat sorted order
    isect.tval = dot(ray.O-p.Q, ray.O-p.Q); 
#endif

    isect.color = p.color;
    isect.color.w = alpha;
    return true;
}


static __forceinline__ __device__ bool intersectHalfEllipsoidSplat( UVRay ray, HalfEllipsoidSplat& p, Isect& isect )
{
    const float3 Q = make_float3(p.Q);
    const float3 A = make_float3(p.A);
    const float3 B = make_float3(p.B);
    const float3 C = make_float3(p.C);

    float3 O = (ray.O - Q);
    O = float3{dot(O, A), dot(O, B), dot(O, C)};

    float3 D = ray.D;
    D = float3{dot(D, A), dot(D, B), dot(D, C)};

    float tca = -dot(O, D) / dot(D, D);
    if( tca < 0.0f )
        return false;

    float3 Pca = O + tca * D;
    float d2 = dot(Pca, Pca);
    if( d2 > 1.0 )
        return false;

    //int alpha = (int)((1.0f - d2) * p.color.w);
    // Is 9.0 right? This makes 3 std deviations. 
    int alpha = (unsigned char)( expf(-0.5f*9.0f*d2) * (float)p.color.w );
    if( alpha <= 0 )
        return false;

    isect.tval = tca;
    isect.color = p.color;
    isect.color.w = alpha;
    return true;
}

//------------------------------------------------------------------------------
// Sphere group intersections
//------------------------------------------------------------------------------
/*
DINLINE void intersectHalfSphereSplatBatch( CUdeviceptr splatBase, HalfSphereSplatGroup* splats, 
                                               HalfUVRay ray, int startSplat, TransparencyRayPayload* prd )
{
    #define BSPLATS SPLATS_PER_BATCH
    #define T_HALF OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
    #define MAT_LAYOUT OPTIX_COOP_VEC_MATRIX_LAYOUT_COLUMN_MAJOR

    //                                      outType    inType    inElem  matLayout   transpose outSize     inSize matElem biasElem
    #define COOP_MAT_MUL optixCoopVecMatMul<T_VEC_OUT, T_VEC_IN, T_HALF, MAT_LAYOUT, false,    BSPLATS, 3,     T_HALF, T_HALF>

    using T_VEC_IN = OptixCoopVec<half, 3>;
    using T_VEC_OUT = OptixCoopVec<half, BSPLATS>;

    //A = [Centers] * U - dotRO_RU
    //A = A * A
    //B = [Centers] * V - dotRO_RV
    //B = B * B
    //AB = A + B
    //AB = AB * [invR2]

    // Load splat centers and splat radii (1/r^2)
    T_VEC_OUT X_vec = optixCoopVecLoad<T_VEC_OUT>( splatBase + splatBatchOffset );
    T_VEC_OUT Y_vec = optixCoopVecLoad<T_VEC_OUT>( splatBase + splatBatchOffset + BSPLATS * 2 );
    T_VEC_OUT Z_vec = optixCoopVecLoad<T_VEC_OUT>( splatBase + splatBatchOffset + BSPLATS * 4 );
    T_VEC_OUT invR2_vec = optixCoopVecLoad<T_VEC_OUT>( splatBase + invR2Offset );

    // U vector
    T_VEC_OUT UX_vec(ray.U.x);
    T_VEC_OUT UY_vec(ray.U.y);
    T_VEC_OUT UZ_vec(ray.U.z);
    T_VEC_OUT ROdotRU_vec( ray.ROdotRU );

    // V vector
    T_VEC_OUT VX_vec(ray.V.x);
    T_VEC_OUT VY_vec(ray.V.y);
    T_VEC_OUT VZ_vec(ray.V.z);
    T_VEC_OUT ROdotRV_vec( ray.ROdotRV );
    
    // A = ([Centers] * U - dotRO_RU)^2
    T_VEC_OUT A_vec = optixCoopVecMul<T_VEC_OUT>( X_vec, UX_vec );
    A_vec = optixCoopVecFFMA( Y_vec, UY_vec, A_vec );
    A_vec = optixCoopVecFFMA( Z_vec, UZ_vec, A_vec );
    A_vec = optixCoopVecSub( A_vec, ROdotRU_vec );
    A_vec = optixCoopVecMul( A_vec, A_vec );

    // A = A + ([Centers] * V - dotRO_RV)^2
    T_VEC_OUT B_vec = optixCoopVecMul( X_vec, VX_vec );
    B_vec = optixCoopVecFFMA( Y_vec, VY_vec, B_vec );
    B_vec = optixCoopVecFFMA( Z_vec, VZ_vec, B_vec );
    B_vec = optixCoopVecSub( B_vec, ROdotRV_vec );
    A_vec = optixCoopVecFFMA( B_vec, B_vec, A_vec );

    // A = A * (1/r^2)
    A_vec = optixCoopVecMul( A_vec, invR2_vec );
*/
    /*
    // Set up U, V, (1/r^2) vectors
    T_VEC_IN U_vec; 
    U_vec[0]=ray.U.x; U_vec[1]=ray.U.y; U_vec[2]=ray.U.z;
    T_VEC_IN V_vec; 
    V_vec[0]=ray.V.x; V_vec[1]=ray.V.y; V_vec[2]=ray.V.z;
    T_VEC_OUT ROdotRU_vec( ray.ROdotRU );
    T_VEC_OUT ROdotRV_vec( ray.ROdotRV );
    T_VEC_OUT invR2_vec = optixCoopVecLoad<T_VEC_OUT>( splatBase + invR2Offset );

    // A = ([Centers] * U - dotRO_RU)^2
    T_VEC_OUT A_vec = COOP_MAT_MUL( U_vec, splatBase, splatBatchOffset, 0, 0 );
    A_vec = optixCoopVecSub<T_VEC_OUT>( A_vec, ROdotRU_vec );
    A_vec = optixCoopVecMul<T_VEC_OUT>( A_vec, A_vec );
    
    // A = A + ([Centers] * V - dotRO_RV)^2
    T_VEC_OUT B_vec = COOP_MAT_MUL( V_vec, splatBase, splatBatchOffset, 0, 0 );
    B_vec = optixCoopVecSub<T_VEC_OUT>( B_vec, ROdotRV_vec );
    A_vec = optixCoopVecFFMA( B_vec, B_vec, A_vec );

    // A = A * (1/r^2)
    A_vec = optixCoopVecMul<T_VEC_OUT>( A_vec, invR2_vec );
    */
/*
    // Early out for if we miss all splats
    unsigned int imask = 0;
    #pragma unroll
    for( int i=0; i<BSPLATS; ++i )
        imask += ((A_vec[i] < (half)1.0f) << i);
    if( imask == 0 )
        return;
    
    half* pbuff = (half*)splatBase;
    #pragma unroll
    for( int i=0; i<BSPLATS; ++i )
    //while( imask )
    {
        //int i = __ffs(imask) - 1;
        //imask = imask ^ (1<<i);

        if( A_vec[i] >= (half)1.0f ) // Ray missed the splat
            continue;
        half3 pC = float3{pbuff[i], pbuff[i+BSPLATS], pbuff[i+2*BSPLATS]};
        half tca = dot(pC - ray.O, ray.D); // Could be dot(pC, r.D) - ROdotRD
        if( tca < (half)0.0f ) // Splat behind ray
            continue;
        half d2 = A_vec[i];

        Isect isect{};

        half4* colors = (half4*)(splatBase + NUM_SPLATS * 8);
        half4 pcolor = colors[startSplat + i];
        //uchar4 pcolor = uchar4{200, 255, 200, 25}; // FIXME
        float alpha = (1.0f - (float)d2) * (float)(pcolor.w);
        isect.tval = tca;
        isect.color = uchar4{ (uchar)(pcolor.x*(half)255), (uchar)(pcolor.y*(half)255), (uchar)(pcolor.z*(half)255), (uchar)(alpha*255) };
        
        addIsect( prd, isect );
    }
}
*/


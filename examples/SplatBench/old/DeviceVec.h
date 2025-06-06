// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

template<class TYPE>
static __forceinline__ __device__ void SWAP( TYPE& a, TYPE& b )
{
    TYPE temp = a;
    a = b;
    b = temp;
}

template<class ARRAYTYPE, class TYPE>
static __forceinline__ __device__ void shell( ARRAYTYPE& data, int size, int step )
{
    for( int i = step; i < size; ++i )
    {
        int j = i;
        while( j >= step && data[j] < data[j - step] )
        {
            SWAP<TYPE>( data[j], data[j - step] );
            j -= step;
        }
    }
}

template<class ARRAYTYPE, class TYPE>
static __forceinline__ __device__ void shellSort( ARRAYTYPE& data, int size )
{
    for( int step = size / 2; step >= 23; step = step / 2 )
        shell<ARRAYTYPE, TYPE>( data, size, step );
    shell<ARRAYTYPE, TYPE>( data, size, 10 );
    shell<ARRAYTYPE, TYPE>( data, size, 4 );
    shell<ARRAYTYPE, TYPE>( data, size, 1 );
}

//--------------------------------------------------------------------------------

template <class TYPE, int ITEMS_PER_CHUNK, int MAX_POINTERS>
class DeviceVec
{
  public:

    //-------------------------------------------------- non-inplace variant
    /*
    __device__ __forceinline__ DeviceVec() { capacity = size = 0; }
    __device__ __forceinline__ int chunkId( int idx ) { return idx / ITEMS_PER_CHUNK; }
    __device__ __forceinline__ int chunkOffset( int idx ) { return idx & (ITEMS_PER_CHUNK-1); }
    __device__ __forceinline__ bool full() { return size >= MAX_POINTERS * ITEMS_PER_CHUNK; }
    __device__ __forceinline__ TYPE& operator[]( int idx ) { return A[chunkId(idx)][chunkOffset(idx)]; }

    __device__ __forceinline__ void add( TYPE& x, DeviceFixedPool& pool ) 
    { 
        if( size >= capacity )
        {
            if( full() ) return;
            A[chunkId(size)] = (TYPE*) pool.alloc();
            capacity += ITEMS_PER_CHUNK;
        }
        (*this)[size] = x;
        ++size;
    }

    __device__ __forceinline__ void clear( DeviceFixedPool& pool )
    {
        if( capacity == 0 ) 
            return;
        for( int idx = chunkId(capacity-1); idx >= 0; --idx )
        {
            pool.free( (char*) A[idx] );
        }
        capacity = 0;
        size = 0;
    }
    */

    //-------------------------------------------------- inplace variant
    __device__ __forceinline__ DeviceVec() 
    {
        inplace = true;
        capacity = ( MAX_POINTERS * sizeof(void*) ) / sizeof( TYPE );
        capacity = min( capacity, ITEMS_PER_CHUNK );
        size = 0;
    }

    __device__ __forceinline__ int chunkId( int idx ) { return idx / ITEMS_PER_CHUNK; }
    __device__ __forceinline__ int chunkOffset( int idx ) { return idx & (ITEMS_PER_CHUNK-1); }
    __device__ __forceinline__ bool full() { return size >= MAX_POINTERS * ITEMS_PER_CHUNK; }

    __device__ __forceinline__ TYPE& operator[]( int idx )
    {
        if( inplace ) 
            return ( (TYPE*)A )[idx];
        return A[chunkId(idx)][chunkOffset(idx)];
    }

    __device__ __forceinline__ void add( TYPE& x, DeviceFixedPool& pool ) 
    {
        if( inplace && size >= capacity )
        {
            TYPE* P = (TYPE*)pool.alloc();
            #pragma unroll
            for( int i=0; i<size; ++i )
            {
                P[i] = ( (TYPE*)A )[i];
            }
            A[0] = P;
            capacity = ITEMS_PER_CHUNK;
            inplace = false;
        }

        if( size >= capacity )
        {
            if( full() ) return;
            A[chunkId(size)] = (TYPE*)pool.alloc();
            capacity += ITEMS_PER_CHUNK;
        }
        
        (*this)[size] = x;
        ++size;
    }

    __device__ __forceinline__ void clear( DeviceFixedPool& pool )
    {
        if( !inplace )
        {
            for( int idx = chunkId(capacity-1); idx >= 0; --idx )
            {
                pool.free( (char*)A[idx] );
            }
        }
        capacity = 0;
        size = 0;
    }

    TYPE* A[MAX_POINTERS];
    unsigned int capacity;
    unsigned int size;
    bool inplace;
};

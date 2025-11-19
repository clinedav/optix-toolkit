// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "NeuralTextureSource.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace imageSource {

// Helper function to get a color based on mip level
static float4 getColorForMipLevel( unsigned int mipLevel )
{
    // Create distinct colors for different mip levels
    const float4 colors[] = {
        { 1.0f, 0.0f, 0.0f, 1.0f },  // Red - mip 0
        { 0.0f, 1.0f, 0.0f, 1.0f },  // Green - mip 1
        { 0.0f, 0.0f, 1.0f, 1.0f },  // Blue - mip 2
        { 1.0f, 1.0f, 0.0f, 1.0f },  // Yellow - mip 3
        { 1.0f, 0.0f, 1.0f, 1.0f },  // Magenta - mip 4
        { 0.0f, 1.0f, 1.0f, 1.0f },  // Cyan - mip 5
        { 1.0f, 0.5f, 0.0f, 1.0f },  // Orange - mip 6
        { 0.5f, 0.0f, 1.0f, 1.0f },  // Purple - mip 7
    };
    const int numColors = sizeof( colors ) / sizeof( colors[0] );
    return colors[mipLevel % numColors];
}

NeuralTextureSource::NeuralTextureSource( unsigned int width, unsigned int height, const std::string& modelPath )
    : m_modelPath( modelPath )
    , m_isOpen( false )
{
    m_info.width        = width;
    m_info.height       = height;
    m_info.format       = CU_AD_FORMAT_FLOAT; // float4
    m_info.numChannels  = 4;
    m_info.numMipLevels = imageSource::calculateNumMipLevels( width, height );
    m_info.isValid      = true;
    m_info.isTiled      = true;
}

void NeuralTextureSource::open( imageSource::TextureInfo* info )
{
    m_isOpen = true;
    if( info != nullptr )
    {
        *info = m_info;
    }
}

void NeuralTextureSource::close()
{
    m_isOpen = false;
}

bool NeuralTextureSource::readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream )
{
    // Get color for this mip level
    float4 color = getColorForMipLevel( mipLevel );
    
    // Calculate total number of pixels
    size_t numPixels = static_cast<size_t>( tile.width ) * static_cast<size_t>( tile.height );
    
    // Allocate host buffer and fill with solid color
    std::vector<float4> hostData( numPixels, color );
    
    // Copy to device memory asynchronously
    cudaMemcpyAsync( dest, hostData.data(), numPixels * sizeof( float4 ), cudaMemcpyHostToDevice, stream );
    
    return true;
}

bool NeuralTextureSource::readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    // Get color for this mip level
    float4 color = getColorForMipLevel( mipLevel );
    
    // Calculate total number of pixels
    size_t numPixels = static_cast<size_t>( expectedWidth ) * static_cast<size_t>( expectedHeight );
    
    // Allocate host buffer and fill with solid color
    std::vector<float4> hostData( numPixels, color );
    
    // Copy to device memory asynchronously
    cudaMemcpyAsync( dest, hostData.data(), numPixels * sizeof( float4 ), cudaMemcpyHostToDevice, stream );
    
    return true;
}

bool NeuralTextureSource::readMipTail( char* dest,
                                       unsigned int mipTailFirstLevel,
                                       unsigned int numMipLevels,
                                       const uint2* mipLevelDims,
                                       CUstream stream )
{
    // Fill each mip level in the tail with its corresponding color
    size_t offset = 0;
    for( unsigned int i = 0; i < numMipLevels; ++i )
    {
        unsigned int mipLevel = mipTailFirstLevel + i;
        unsigned int width    = mipLevelDims[i].x;
        unsigned int height   = mipLevelDims[i].y;
        
        // Get color for this mip level
        float4 color = getColorForMipLevel( mipLevel );
        
        // Calculate total number of pixels
        size_t numPixels = static_cast<size_t>( width ) * static_cast<size_t>( height );
        
        // Allocate host buffer and fill with solid color
        std::vector<float4> hostData( numPixels, color );
        
        // Copy to device memory asynchronously
        cudaMemcpyAsync( dest + offset, hostData.data(), numPixels * sizeof( float4 ), cudaMemcpyHostToDevice, stream );
        
        offset += numPixels * sizeof( float4 );
    }
    return true;
}

bool NeuralTextureSource::readBaseColor( float4& dest )
{
    // Return the color for the highest mip level (base color)
    // For a 1x1 texture, we can use the last mip level's color
    dest = getColorForMipLevel( m_info.numMipLevels - 1 );
    return true;
}

}  // namespace imageSource


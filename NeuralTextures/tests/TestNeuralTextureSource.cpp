// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "NeuralTextureSource.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

using namespace imageSource;

class TestNeuralTextureSource : public testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA context for tests
        cudaFree(0);
    }
};

TEST_F( TestNeuralTextureSource, Constructor )
{
    NeuralTextureSource source( 512, 512, "dummy_model.pth" );
    
    EXPECT_FALSE( source.isOpen() );
}

TEST_F( TestNeuralTextureSource, OpenClose )
{
    NeuralTextureSource source( 256, 256, "test_model.pth" );
    
    EXPECT_FALSE( source.isOpen() );
    
    TextureInfo info;
    source.open( &info );
    
    EXPECT_TRUE( source.isOpen() );
    EXPECT_EQ( 256U, info.width );
    EXPECT_EQ( 256U, info.height );
    EXPECT_EQ( CU_AD_FORMAT_FLOAT, info.format );
    EXPECT_EQ( 4U, info.numChannels );
    EXPECT_TRUE( info.isValid );
    EXPECT_TRUE( info.isTiled );
    
    source.close();
    EXPECT_FALSE( source.isOpen() );
}

TEST_F( TestNeuralTextureSource, GetInfo )
{
    const unsigned int width = 1024;
    const unsigned int height = 512;
    NeuralTextureSource source( width, height, "model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    const TextureInfo& retrievedInfo = source.getInfo();
    EXPECT_EQ( width, retrievedInfo.width );
    EXPECT_EQ( height, retrievedInfo.height );
    EXPECT_EQ( CU_AD_FORMAT_FLOAT, retrievedInfo.format );
    EXPECT_EQ( 4U, retrievedInfo.numChannels );
}

TEST_F( TestNeuralTextureSource, ReadTile )
{
    NeuralTextureSource source( 512, 512, "test_model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    const unsigned int mipLevel = 0;
    const unsigned int tileWidth = 64;
    const unsigned int tileHeight = 64;
    
    // Allocate device memory for tile
    float4* d_tile;
    const size_t tileSize = tileWidth * tileHeight * sizeof(float4);
    ASSERT_EQ( cudaMalloc( &d_tile, tileSize ), cudaSuccess );
    
    // Read tile
    ASSERT_TRUE( source.readTile( reinterpret_cast<char*>(d_tile), mipLevel, 
                                  { 0, 0, tileWidth, tileHeight }, nullptr ) );
    
    // Copy back to host to verify
    std::vector<float4> h_tile( tileWidth * tileHeight );
    ASSERT_EQ( cudaMemcpy( h_tile.data(), d_tile, tileSize, cudaMemcpyDeviceToHost ), cudaSuccess );
    
    // Verify that tile has data (mip level 0 should be red)
    EXPECT_FLOAT_EQ( 1.0f, h_tile[0].x );  // Red channel
    EXPECT_FLOAT_EQ( 0.0f, h_tile[0].y );  // Green channel
    EXPECT_FLOAT_EQ( 0.0f, h_tile[0].z );  // Blue channel
    EXPECT_FLOAT_EQ( 1.0f, h_tile[0].w );  // Alpha channel
    
    cudaFree( d_tile );
}

TEST_F( TestNeuralTextureSource, ReadMipLevel )
{
    NeuralTextureSource source( 128, 128, "test_model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    const unsigned int mipLevel = 1;
    const unsigned int levelWidth = 64;
    const unsigned int levelHeight = 64;
    
    // Allocate device memory
    float4* d_mipLevel;
    const size_t levelSize = levelWidth * levelHeight * sizeof(float4);
    ASSERT_EQ( cudaMalloc( &d_mipLevel, levelSize ), cudaSuccess );
    
    // Read mip level
    ASSERT_TRUE( source.readMipLevel( reinterpret_cast<char*>(d_mipLevel), mipLevel, 
                                      levelWidth, levelHeight, nullptr ) );
    
    // Copy back to host to verify
    std::vector<float4> h_mipLevel( levelWidth * levelHeight );
    ASSERT_EQ( cudaMemcpy( h_mipLevel.data(), d_mipLevel, levelSize, cudaMemcpyDeviceToHost ), cudaSuccess );
    
    // Verify that mip level 1 should be green
    EXPECT_FLOAT_EQ( 0.0f, h_mipLevel[0].x );  // Red channel
    EXPECT_FLOAT_EQ( 1.0f, h_mipLevel[0].y );  // Green channel
    EXPECT_FLOAT_EQ( 0.0f, h_mipLevel[0].z );  // Blue channel
    EXPECT_FLOAT_EQ( 1.0f, h_mipLevel[0].w );  // Alpha channel
    
    cudaFree( d_mipLevel );
}

TEST_F( TestNeuralTextureSource, ReadMipTail )
{
    NeuralTextureSource source( 128, 128, "test_model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    const unsigned int mipTailFirstLevel = 3;
    const unsigned int numMipLevels = 3;
    
    // Define dimensions for mip tail levels
    std::vector<uint2> mipLevelDims = {
        { 16, 16 },  // mip level 3
        { 8, 8 },    // mip level 4
        { 4, 4 }     // mip level 5
    };
    
    // Calculate total size needed
    size_t totalSize = 0;
    for( const auto& dims : mipLevelDims )
    {
        totalSize += dims.x * dims.y * sizeof(float4);
    }
    
    // Allocate device memory
    char* d_mipTail;
    ASSERT_EQ( cudaMalloc( &d_mipTail, totalSize ), cudaSuccess );
    
    // Read mip tail
    ASSERT_TRUE( source.readMipTail( d_mipTail, mipTailFirstLevel, numMipLevels, 
                                     mipLevelDims.data(), nullptr ) );
    
    // Copy back to host to verify first mip level in tail
    std::vector<float4> h_firstMip( mipLevelDims[0].x * mipLevelDims[0].y );
    ASSERT_EQ( cudaMemcpy( h_firstMip.data(), d_mipTail, 
                          mipLevelDims[0].x * mipLevelDims[0].y * sizeof(float4), 
                          cudaMemcpyDeviceToHost ), cudaSuccess );
    
    // Verify that mip level 3 should be yellow
    EXPECT_FLOAT_EQ( 1.0f, h_firstMip[0].x );  // Red channel
    EXPECT_FLOAT_EQ( 1.0f, h_firstMip[0].y );  // Green channel
    EXPECT_FLOAT_EQ( 0.0f, h_firstMip[0].z );  // Blue channel
    
    cudaFree( d_mipTail );
}

TEST_F( TestNeuralTextureSource, ReadBaseColor )
{
    NeuralTextureSource source( 256, 256, "test_model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    float4 baseColor;
    ASSERT_TRUE( source.readBaseColor( baseColor ) );
    
    // Base color should be set (exact value depends on numMipLevels)
    // Just verify we got a valid color with proper format
    EXPECT_GE( baseColor.x, 0.0f );
    EXPECT_LE( baseColor.x, 1.0f );
    EXPECT_GE( baseColor.y, 0.0f );
    EXPECT_LE( baseColor.y, 1.0f );
    EXPECT_GE( baseColor.z, 0.0f );
    EXPECT_LE( baseColor.z, 1.0f );
    EXPECT_FLOAT_EQ( 1.0f, baseColor.w );
}

TEST_F( TestNeuralTextureSource, MipLevelColors )
{
    NeuralTextureSource source( 512, 512, "test_model.pth" );
    
    TextureInfo info;
    source.open( &info );
    
    // Test that different mip levels produce different colors
    const unsigned int tileSize = 32;
    float4* d_tile;
    ASSERT_EQ( cudaMalloc( &d_tile, tileSize * tileSize * sizeof(float4) ), cudaSuccess );
    
    std::vector<float4> mip0( tileSize * tileSize );
    std::vector<float4> mip1( tileSize * tileSize );
    
    // Read mip level 0
    ASSERT_TRUE( source.readTile( reinterpret_cast<char*>(d_tile), 0, 
                                  { 0, 0, tileSize, tileSize }, nullptr ) );
    ASSERT_EQ( cudaMemcpy( mip0.data(), d_tile, tileSize * tileSize * sizeof(float4), 
                          cudaMemcpyDeviceToHost ), cudaSuccess );
    
    // Read mip level 1
    ASSERT_TRUE( source.readTile( reinterpret_cast<char*>(d_tile), 1, 
                                  { 0, 0, tileSize, tileSize }, nullptr ) );
    ASSERT_EQ( cudaMemcpy( mip1.data(), d_tile, tileSize * tileSize * sizeof(float4), 
                          cudaMemcpyDeviceToHost ), cudaSuccess );
    
    // Verify colors are different between mip levels
    EXPECT_NE( mip0[0].x, mip1[0].x ) << "Mip level 0 and 1 should have different colors";
    
    cudaFree( d_tile );
}

TEST_F( TestNeuralTextureSource, GetFillType )
{
    NeuralTextureSource source( 256, 256, "test_model.pth" );
    
    EXPECT_EQ( CU_MEMORYTYPE_DEVICE, source.getFillType() );
}

TEST_F( TestNeuralTextureSource, DifferentDimensions )
{
    // Test with various dimensions
    std::vector<std::pair<unsigned int, unsigned int>> dimensions = {
        {64, 64},
        {128, 256},
        {512, 512},
        {1024, 512},
        {2048, 2048}
    };
    
    for( const auto& dim : dimensions )
    {
        NeuralTextureSource source( dim.first, dim.second, "test_model.pth" );
        
        TextureInfo info;
        source.open( &info );
        
        EXPECT_EQ( dim.first, info.width ) << "Width mismatch for " << dim.first << "x" << dim.second;
        EXPECT_EQ( dim.second, info.height ) << "Height mismatch for " << dim.first << "x" << dim.second;
        EXPECT_GT( info.numMipLevels, 0U ) << "Should have at least one mip level";
    }
}


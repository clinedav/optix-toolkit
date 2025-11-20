/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <fstream>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include "NtcImageReader.h"

#define OPTIX_CHK( call ) if( call != OPTIX_SUCCESS ) return false
#define CUDA_CHK( call ) if( call != cudaSuccess ) return false

bool NtcImageReader::loadFile( const char* fileName )
{
    const uint32_t NTEX_MAGIC_NUMBER = 0x5845544E; // "NTEX"
    const uint32_t NTEX_SUPPORTED_VERSION = 0x100;

    std::ifstream file( fileName, std::ios::binary );
    if( !file.is_open() )
        return false;

    // Read file header
    NtcFileHeader header{};
    file.read( reinterpret_cast<char*>( &header ), sizeof( NtcFileHeader ) );
    if( header.magicNumber != NTEX_MAGIC_NUMBER || header.version != NTEX_SUPPORTED_VERSION )
        return false;

    // Read json text
    std::vector<char> jsonText( header.jsonSize + 1, '\0' );
    file.seekg( header.jsonOffset );
    file.read( jsonText.data(), header.jsonSize );

    // Parse json document and fill in texture description
    rapidjson::Document jsonDoc;
    jsonDoc.Parse( jsonText.data() );
    if ( jsonDoc.HasParseError() )
        return false;
    if( !parseTextureSetDescription( jsonDoc ) )
        return false;

    // Figure out how much of the data chunk to read
    bool loadLatents = true;
    int dataChunkReadSize = header.dataSize;
    if( !loadLatents )
    {
        int latentOffsetViewIdx = jsonDoc["latents"][0]["view"].GetInt();
        int latentOffset = jsonDoc["views"][latentOffsetViewIdx]["offset"].GetInt();
        dataChunkReadSize = latentOffset;
    }
    
    // Read data chunk, parse latents and network weights
    m_hDataChunk.resize( dataChunkReadSize, '\0' );

    file.seekg( header.dataOffset );
    file.read( m_hDataChunk.data(), dataChunkReadSize );
    if( !parseLatentsDescription( jsonDoc ) )
        return false;
    if( !parseNetworkDescription( jsonDoc ) )
        return false;

    printTextureSetDescription();
    return true;
}


bool NtcImageReader::parseTextureSetDescription( rapidjson::Document& doc )
{
    try
    {
        NtcTextureSetConstants& tsc = m_textureSet.constants;

        // General characteristics
        tsc.imageMips = doc["numColorMips"].GetInt();
        tsc.imageWidth = doc["colorMips"][0]["width"].GetInt();
        tsc.imageHeight = doc["colorMips"][0]["height"].GetInt();

        tsc.validChannelMask = 0;
        tsc.channelColorSpaces = 0; // FIXME: actually read these

        m_textureSet.numTextures = doc["textures"].Size();
        m_textureSet.numChannels = doc["numChannels"].GetInt();

        // Latents info
        m_textureSet.latentFeatures = doc["latentShape"]["numFeatures"].GetInt();
        m_textureSet.latentWidth = doc["latents"][0]["width"].GetInt();
        m_textureSet.latentHeight = doc["latents"][0]["height"].GetInt();

        // Color mips
        for( int colorMipLevel = 0; colorMipLevel < tsc.imageMips; ++colorMipLevel )
        {
            NtcColorMipConstants &colorMip = tsc.colorMips[colorMipLevel];
            colorMip.neuralMip = doc["colorMips"][colorMipLevel]["latentMip"].GetInt();
            colorMip.positionLod = doc["colorMips"][colorMipLevel]["positionLod"].GetFloat();
            colorMip.positionScale = doc["colorMips"][colorMipLevel]["positionScale"].GetFloat();
        }
    }
    catch(...)
    {
        return false;
    }

    return true;
}


bool NtcImageReader::parseLatentsDescription( rapidjson::Document& doc )
{
    try 
    {
        // The latent data is held in m_hDataChunk. Read the offsets and sizes
        m_hLatentMipOffsets.resize( doc["latents"].Size(), 0 );
        m_hLatentMipSizes.resize( doc["latents"].Size(), 0 );
        m_textureSet.numLatentMips = static_cast<int>( doc["latents"].Size() );
        for( unsigned int level = 0; level < doc["latents"].Size(); ++level )
        {
            int dataViewIdx = doc["latents"][level]["view"].GetInt();
            m_hLatentMipOffsets[level] = doc["views"][dataViewIdx]["offset"].GetInt();
            m_hLatentMipSizes[level] = doc["views"][dataViewIdx]["storedSize"].GetInt();
        }
    }
    catch(...)
    {
        return false;
    }

    return true;
}


bool NtcImageReader::parseNetworkDescription( rapidjson::Document& doc )
{
    try
    {
        // Find a proper network (3 or 4 layers, FloatE4M3 weights...)
        unsigned int networkIdx = 0;
        for( networkIdx = 0; networkIdx < doc["mlpVersions"].Size(); ++networkIdx )
        {
            rapidjson::Value& network = doc["mlpVersions"][networkIdx];
            if( ( network["layers"].Size() == 4 || network["layers"].Size() == 3 ) &&
                network["layers"][0]["weightType"].GetString() == std::string("FloatE4M3") )
            {
                break;
            }
        }
        if( networkIdx >= doc["mlpVersions"].Size() )
            return false;
        rapidjson::Value& network = doc["mlpVersions"][networkIdx];

        // Read the network layer descriptions, and copy data to m_hNetworkData
        // so it is in one contiguous array.

        const int maxNetworkSizeInBytes = 16384;
        m_hNetworkData.resize( maxNetworkSizeInBytes );
        m_hNetwork.resize( network["layers"].Size() );
        int offset = 0;

        // Read the network matrix weights first so network layers are all together
        for( unsigned int layerId = 0; layerId < network["layers"].Size(); ++layerId )
        {
            rapidjson::Value& networkLayer = network["layers"][layerId];
            NtcNetworkLayer& layer = m_hNetwork[layerId];

            layer.inputChannels = networkLayer["inputChannels"].GetInt();
            layer.outputChannels = networkLayer["outputChannels"].GetInt();
            
            if( networkLayer.HasMember( "weightView" ) )
            {
                int viewIdx = networkLayer["weightView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.weightSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.weightOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.weightSize );
                offset += layer.weightSize;
            }

            if( networkLayer.HasMember( "weightType" ) )
                layer.weightType = networkLayer["weightType"].GetString();
            if( networkLayer.HasMember( "scaleType" ) )
                layer.scaleType = networkLayer["scaleType"].GetString();
            if( networkLayer.HasMember( "biasType" ) )
                layer.biasType = networkLayer["biasType"].GetString();
        }

        // Read the scales and biases
        for( unsigned int layerId = 0; layerId < network["layers"].Size(); ++layerId )
        {
            rapidjson::Value& networkLayer = network["layers"][layerId];
            NtcNetworkLayer& layer = m_hNetwork[layerId];

            if( networkLayer.HasMember( "scaleView" ) )
            {
                int viewIdx = networkLayer["scaleView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.scaleSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.scaleOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.scaleSize );
                offset += layer.scaleSize;
            }
            if( networkLayer.HasMember( "biasView" ) )
            {
                int viewIdx = networkLayer["biasView"].GetInt();
                int srcOffset = doc["views"][viewIdx]["offset"].GetInt();
                layer.biasSize = doc["views"][viewIdx]["storedSize"].GetInt();
                layer.biasOffset = offset;
                memcpy( &m_hNetworkData[offset], &m_hDataChunk[srcOffset], layer.biasSize );
                offset += layer.biasSize;
            }
        }

        m_hNetworkData.resize( offset );
    }
    catch(...)
    {
        return false;
    }

    return true;
}


bool NtcImageReader::readLatentRectUshort( ushort* dest, int mipLevel, int xstart, int ystart, int width, int height )
{
    int numLatentTextures = m_textureSet.latentFeatures / 4;
    int destPixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;

    int mipWidth = m_textureSet.latentWidth >> mipLevel;
    int mipHeight = m_textureSet.latentHeight >> mipLevel;
    int latentOffset = m_hLatentMipOffsets[mipLevel];

    ushort* src = (ushort*) &m_hDataChunk[latentOffset];
    width = std::min( width, mipWidth - xstart );
    height = std::min( height, mipHeight - ystart );

    for( int y = 0; y < height; ++y )
    {
        for( int x = 0; x < width; ++x )
        {
            ushort* pixelDest = &dest[( y * width + x ) * destPixelStride];
            for( int c = 0; c < numLatentTextures; ++c )
            {
                int srcLayerOffset = mipWidth * mipHeight * c;
                int srcPixelOffset = ( y + ystart ) * mipWidth + ( x + xstart );
                ushort* pixelSrc = &src[srcLayerOffset + srcPixelOffset];
                pixelDest[c] = *pixelSrc;
            }
        }
    }
    return true;
}


bool NtcImageReader::makeLatentTexture()
{
    // Allocate mipmapped CUDA array
    int numLatentTextures = m_textureSet.latentFeatures / 4;
    int pixelStride = (numLatentTextures != 3) ? numLatentTextures : 4;
    int numMips = m_hLatentMipOffsets.size();
    cudaChannelFormatDesc channelDesc;
    if( numLatentTextures == 1 )
        channelDesc = cudaCreateChannelDesc<ushort>();
    else if( numLatentTextures == 2 )
        channelDesc = cudaCreateChannelDesc<ushort2>();
    else if( numLatentTextures >= 3 )
        channelDesc = cudaCreateChannelDesc<ushort4>();

    cudaExtent extent = make_cudaExtent( m_textureSet.latentWidth, m_textureSet.latentHeight, 0 );
    cudaMipmappedArray_t mipmappedArray;
    cudaMallocMipmappedArray( &mipmappedArray, &channelDesc, extent, numMips );
    
    // Fill each mip level
    for ( int mipLevel = 0; mipLevel < numMips; mipLevel++ )
    {
        // No data for this mip level
        if( mipLevel > 0 && m_hLatentMipOffsets[mipLevel] == 0 )
            continue;

        // Get the source (latentSrc) and destination (levelArray)
        //std::vector<ushort> latentSrc;
        //convertLatentMipLevelToCombinedUshort( mipLevel, latentSrc );
        int mipWidth = m_textureSet.latentWidth >> mipLevel;
        int mipHeight = m_textureSet.latentHeight >> mipLevel;
        std::vector<ushort> latentSrc( mipWidth * mipHeight * pixelStride, 0 );
        readLatentRectUshort( latentSrc.data(), mipLevel, 0, 0, mipWidth, mipHeight );
        cudaArray_t levelArray;
        cudaGetMipmappedArrayLevel( &levelArray, mipmappedArray, mipLevel );
        
        // Copy data to this mip level
        cudaMemcpy2DToArray(
            levelArray,
            0, 0,  // offset in destination
            latentSrc.data(),  // source pointer with offset
            mipWidth * pixelStride * sizeof(ushort),  // source pitch
            mipWidth * pixelStride * sizeof(ushort),  // width in bytes
            mipHeight,  // height
            cudaMemcpyHostToDevice
        );
    }
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = mipmappedArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = (float)( numMips - 1 );
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.maxAnisotropy = 16;
    
    cudaTextureObject_t texObj;
    cudaCreateTextureObject( &texObj, &resDesc, &texDesc, nullptr );
    m_textureSet.latentTexture = texObj;

    return m_textureSet.latentTexture != 0;
}


bool NtcImageReader::prepareDeviceNetwork( OptixDeviceContext optixContext )
{
    if( m_hNetwork.size() != NTC_MLP_LAYERS )
        return false;
    if( m_textureSet.d_mlpWeights != nullptr ) // already prepared
        return true;

    void* d_networkData = nullptr;
    CUDA_CHK( cudaMalloc( (void**)&d_networkData, m_hNetworkData.size() ) );
    CUDA_CHK( cudaMemcpy( d_networkData, m_hNetworkData.data(), m_hNetworkData.size(), cudaMemcpyHostToDevice ) ); 

    bool rval = convertNetworkToOptixInferencingOptimal( optixContext, d_networkData );
    CUDA_CHK( cudaFree( d_networkData ) );
    return rval;
}


bool NtcImageReader::convertNetworkToOptixInferencingOptimal( OptixDeviceContext optixContext, void* d_srcNetworkData )
{
    const int numLayers = NTC_MLP_LAYERS;
    int networkSizeInBytes = (int)m_hNetworkData.size();
    
    std::vector<OptixCoopVecMatrixDescription> srcLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );
    std::vector<OptixCoopVecMatrixDescription> dstLayerDesc( numLayers, OptixCoopVecMatrixDescription{} );

    OptixCoopVecMatrixLayout srcMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    OptixCoopVecMatrixLayout dstMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;
    
    NtcTextureSetConstants& tsc = m_textureSet.constants;
    const int optStride = 0;

    // Compute layer sizes
    for( int i = 0; i < numLayers; ++i )
    {
        const unsigned int N = m_hNetwork[i].outputChannels;
        const unsigned int K = m_hNetwork[i].inputChannels;

        size_t srcMatrixDataSize = 0;
        size_t dstMatrixDataSize = 0;

        OptixCoopVecElemType layerType = (i < numLayers - 1) ? OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3 : OPTIX_COOP_VEC_ELEM_TYPE_INT8;

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            optixContext,
            N,
            K,
            layerType,
            srcMatrixLayout,
            optStride,
            &srcMatrixDataSize
            ) );

        OPTIX_CHK( optixCoopVecMatrixComputeSize(
            optixContext,
            N,
            K,
            layerType,
            dstMatrixLayout,
            optStride,
            &dstMatrixDataSize
            ) );
        
        OptixCoopVecMatrixDescription& srcLayer = srcLayerDesc[i];
        OptixCoopVecMatrixDescription& dstLayer = dstLayerDesc[i];
        srcLayer.N = dstLayer.N = N;
        srcLayer.K = dstLayer.K = K;
        srcLayer.offsetInBytes  = m_hNetwork[i].weightOffset;
        dstLayer.offsetInBytes  = i == 0 ? 0 : dstLayerDesc[i - 1].offsetInBytes + dstLayerDesc[i - 1].sizeInBytes;
        srcLayer.elementType    = layerType;
        dstLayer.elementType    = layerType;
        srcLayer.layout         = srcMatrixLayout;
        dstLayer.layout         = dstMatrixLayout;
        srcLayer.rowColumnStrideInBytes = optStride;
        dstLayer.rowColumnStrideInBytes = optStride;
        srcLayer.sizeInBytes    = static_cast<unsigned int>( srcMatrixDataSize );
        dstLayer.sizeInBytes    = static_cast<unsigned int>( dstMatrixDataSize );

        // Put network data offsets in the texture set constants
        tsc.networkWeightOffsets[i] = dstLayer.offsetInBytes;
    }

    OptixNetworkDescription inputNetworkDescription = { srcLayerDesc.data(), static_cast<unsigned int>( srcLayerDesc.size() ) };
    OptixNetworkDescription outputNetworkDescription = { dstLayerDesc.data(), static_cast<unsigned int>( dstLayerDesc.size() ) };

    size_t dst_mats_size = dstLayerDesc.back().offsetInBytes + dstLayerDesc.back().sizeInBytes;  // trick to sum all dstLayer sizes
    size_t src_mats_size = srcLayerDesc.back().offsetInBytes + srcLayerDesc.back().sizeInBytes;  // trick to sum all srcLayer sizes
    size_t src_other_stuff_size = networkSizeInBytes - src_mats_size;
    size_t dst_total_size       = dst_mats_size + src_other_stuff_size;

    CUdeviceptr d_dstMatrix;
    CUDA_CHK( cudaMalloc( (void**)&d_dstMatrix, dst_total_size ) );

    const int numNetworks = 1;
    OPTIX_CHK( optixCoopVecMatrixConvert(
        optixContext,
        CUstream{0},
        numNetworks,
        &inputNetworkDescription,
        (CUdeviceptr)d_srcNetworkData,
        optStride,
        &outputNetworkDescription,
        d_dstMatrix,
        optStride) );

    // Put scale and bias offsets in texture set constants
    tsc.networkScaleOffsets[0] = dst_mats_size;
    tsc.networkBiasOffsets[0] = tsc.networkScaleOffsets[0] + m_hNetwork[0].scaleSize;
    for( int i = 1; i < numLayers; ++i )
    {
        tsc.networkScaleOffsets[i] = tsc.networkBiasOffsets[i-1] + m_hNetwork[i-1].biasSize;
        tsc.networkBiasOffsets[i] = tsc.networkScaleOffsets[i] + m_hNetwork[i].scaleSize;
    }

    // copy the other stuff after the mats arrays from src to dest
    CUDA_CHK( cudaMemcpy(
        (void*)(d_dstMatrix + dst_mats_size),
        (void*)((CUdeviceptr)d_srcNetworkData + src_mats_size),
        src_other_stuff_size,
        cudaMemcpyDeviceToDevice ) );

    m_textureSet.d_mlpWeights = (uint8_t*)d_dstMatrix;
    return true;
}


void NtcImageReader::printTextureSetDescription()
{
    NtcTextureSetConstants& tsc = m_textureSet.constants;
    printf( "width:%d, height:%d, mips:%d\n", tsc.imageWidth, tsc.imageHeight, tsc.imageMips );
    printf( "validChannelMask:%x, channelColorSpaces:%x\n", tsc.validChannelMask, tsc.channelColorSpaces );
    
    for( unsigned int i = 0; i < m_hNetwork.size(); ++i )
    {
        NtcNetworkLayer& layer = m_hNetwork[i];
        printf( "Layer %d: inputs:%d, outputs:%d, weights:%s,%d,%d, scale:%s,%d,%d, bias:%s,%d,%d\n",
            i, layer.inputChannels, layer.outputChannels,
            layer.weightType.c_str(), layer.weightOffset, layer.weightSize,
            layer.scaleType.c_str(), layer.scaleOffset, layer.scaleSize,
            layer.biasType.c_str(), layer.biasOffset, layer.biasSize );
    }

    for( int i=0; i<tsc.imageMips; ++i )
    {
        NtcColorMipConstants mip = tsc.colorMips[i];
        printf( "mip[%d]: neuralMip:%d, positionLod:%1.3f, positionScale:%1.3f\n", 
            i, mip.neuralMip, mip.positionLod, mip.positionScale );
    }

    printf("numTextures:%d, numChannels:%d\n", m_textureSet.numTextures, m_textureSet.numChannels );

    printf("latentFeatures:%d, latentWidth:%d, latentHeight:%d\n", 
        m_textureSet.latentFeatures, m_textureSet.latentWidth, m_textureSet.latentHeight );
    
    for( unsigned int i=0; i < m_hLatentMipOffsets.size(); ++i )
    {
        printf( "latentMip[%d]: offset:%d\n", i, m_hLatentMipOffsets[i] );
    }
}

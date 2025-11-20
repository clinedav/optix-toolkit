// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cuda_runtime.h>
#include <vector>

#include <OptiXToolkit/Error/ErrorCheck.h>

#include "NeuralTextureSource.h"

namespace imageSource {

NeuralTextureSource::NeuralTextureSource( const std::string& filename )
    : m_filename( filename )
    , m_isOpen( false )
{
}

void NeuralTextureSource::open( imageSource::TextureInfo* info )
{
    if( !m_isOpen )
    {
        std::string errString = "Could not open NTC image file " + m_filename;
        bool success = m_imageReader.loadFile( m_filename.c_str() );
        OTK_ERROR_CHECK_MSG( !success, errString.c_str() );
        NtcTextureSet nts = m_imageReader.getTextureSet();

        m_info.width        = nts.latentWidth;
        m_info.height       = nts.latentHeight;
        m_info.format       = CU_AD_FORMAT_UNSIGNED_INT16;
        m_info.numChannels  = ( nts.latentFeatures != 12 ) ? nts.latentFeatures / 4 : 4;
        m_info.numMipLevels = nts.numLatentMips;
        m_info.isValid      = true;
        m_info.isTiled      = true;
    }

    m_isOpen = true;
    if( info != nullptr )
    {
        *info = m_info;
    }
}

bool NeuralTextureSource::readTile( char* dest, unsigned int latentMipLevel, const imageSource::Tile& tile, CUstream stream )
{
    (void) stream;
    return m_imageReader.readLatentRectUshort( (ushort*)dest, latentMipLevel, tile.x * tile.width, tile.y * tile.height, tile.width, tile.height );
}

bool NeuralTextureSource::readMipLevel( char* dest, unsigned int latentMipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream )
{
    (void) stream;
    int mipWidth = m_info.width >> latentMipLevel;
    int mipHeight = m_info.height >> latentMipLevel;
    return m_imageReader.readLatentRectUshort( (ushort*)dest, latentMipLevel, 0, 0, mipWidth, mipHeight );
}

}  // namespace imageSource


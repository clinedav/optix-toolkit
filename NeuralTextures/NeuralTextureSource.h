// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>

#include <string>

namespace imageSource {

/// NeuralTextureSource generates textures using neural network inference
class NeuralTextureSource : public imageSource::ImageSourceBase
{
  public:
    /// Create a neural texture source with the specified dimensions and model path
    NeuralTextureSource( unsigned int width, unsigned int height, const std::string& modelPath );

    /// The destructor is virtual.
    ~NeuralTextureSource() override = default;

    /// The open method initializes the given image info struct.
    void open( imageSource::TextureInfo* info ) override;

    /// The close operation.
    void close() override;

    /// Check if image is currently open.
    bool isOpen() const override { return m_isOpen; }

    /// Get the image info.  Valid only after calling open().
    const imageSource::TextureInfo& getInfo() const override { return m_info; }

    /// Return the mode in which the image fills part of itself
    CUmemorytype getFillType() const override { return CU_MEMORYTYPE_DEVICE; }

    /// Read the specified tile or mip level, returning the data in dest.  dest must be large enough
    /// to hold the tile.  Pixels outside the bounds of the mip level will be filled in with black.
    bool readTile( char* dest, unsigned int mipLevel, const imageSource::Tile& tile, CUstream stream ) override;

    /// Read the specified mipLevel.  Returns true for success.
    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int expectedWidth, unsigned int expectedHeight, CUstream stream ) override;

    /// Read the mip tail into a single buffer
    bool readMipTail( char* dest,
                      unsigned int mipTailFirstLevel,
                      unsigned int numMipLevels,
                      const uint2* mipLevelDims,
                      CUstream stream ) override;

    /// Read the base color of the image (1x1 mip level) as a float4. Returns true on success.
    bool readBaseColor( float4& dest ) override;

  private:
    imageSource::TextureInfo m_info;
    std::string m_modelPath;
    bool m_isOpen;
};

}  // namespace imageSource


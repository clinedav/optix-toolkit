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
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <rapidjson/document.h>

#include "NtcTextureSet.h"

class NtcImageReader
{
  public:
    bool loadFile( const char* fileName );
    bool makeLatentTexture();
    bool prepareDeviceNetwork( OptixDeviceContext optixContext );
    const NtcTextureSet& getTextureSet() { return m_textureSet; }
    bool readLatentRectUshort( ushort* dest, int mipLevel, int xstart, int ystart, int width, int height );

  private:

    struct NtcFileHeader
    {
        uint32_t magicNumber;
        uint32_t version;
        uint64_t jsonOffset;
        uint64_t jsonSize;
        uint64_t dataOffset;
        uint64_t dataSize;
    };
    
    struct NtcNetworkLayer
    {
        int inputChannels = -1;
        int outputChannels = -1;
        int weightOffset = -1;
        int weightSize = 0;
        int scaleOffset = -1;
        int scaleSize = 0;
        int biasOffset = -1;
        int biasSize = 0;
        std::string weightType;
        std::string scaleType;
        std::string biasType;
    };

    NtcTextureSet m_textureSet{};
    std::vector<char> m_hDataChunk;
    std::vector<int> m_hLatentMipOffsets;
    std::vector<int> m_hLatentMipSizes;
    std::vector<NtcNetworkLayer> m_hNetwork;
    std::vector<char> m_hNetworkData;

    bool parseTextureSetDescription( rapidjson::Document& doc );
    bool parseLatentsDescription( rapidjson::Document& doc );
    bool parseNetworkDescription( rapidjson::Document& doc );
    
    bool convertNetworkToOptixInferencingOptimal( OptixDeviceContext optixContext, void* d_srcNetworkData );

    void printTextureSetDescription();
};
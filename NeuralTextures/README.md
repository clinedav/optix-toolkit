# Neural Textures

This module provides a neural network-based texture generation system for the OptiX Toolkit.

## NeuralTextureSource

`NeuralTextureSource` is a derived class of `ImageSource` that generates textures using neural network inference.

### Features

- Implements the full `ImageSource` interface
- Supports mipmapped textures with multiple levels
- Designed for GPU-based texture generation using CUDA
- Can load neural network models from file paths

### Usage

```cpp
#include "NeuralTextureSource.h"

// Create a neural texture source
auto neuralTexture = std::make_shared<imageSource::NeuralTextureSource>(
    2048,                    // width
    2048,                    // height
    "path/to/model.pt"      // neural network model path
);

// Open the texture
imageSource::TextureInfo info;
neuralTexture->open(&info);

// Read texture data
char* buffer = ...; // allocate buffer
neuralTexture->readMipLevel(buffer, 0, info.width, info.height, stream);

// Close when done
neuralTexture->close();
```

### Implementation Notes

The current implementation provides a skeleton for neural texture generation. The following methods need to be implemented based on your specific neural network framework:

1. **`open()`**: Load the neural network model from the specified path
2. **`close()`**: Release neural network resources
3. **`generateNeuralTexture()`**: Run neural network inference to generate texture data

### TODO

- [ ] Integrate neural network inference framework (e.g., TensorRT, PyTorch C++)
- [ ] Implement model loading in `open()`
- [ ] Implement actual texture generation in `generateNeuralTexture()`
- [ ] Add support for different texture formats beyond float4
- [ ] Optimize memory usage for large textures
- [ ] Add caching for generated mip levels

### Dependencies

- OptiX Toolkit ImageSource library
- OptiX Toolkit Error library
- CUDA Driver API

### Building

The NeuralTextures module is built as part of the OptiX Toolkit. To include it in your build:

```bash
cmake -DOTK_LIBRARIES="ALL" ..
# or specifically include NeuralTextures
cmake -DOTK_LIBRARIES="DemandLoading;NeuralTextures" ..
```

### License

SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause



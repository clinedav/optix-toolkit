# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include( FetchGtest )
include( GoogleTest )

otk_add_executable( testNeuralTextureSource
  TestNeuralTextureSource.cpp
)

target_link_libraries( testNeuralTextureSource PUBLIC
    NeuralTextures
    OptiXToolkit::ImageSource
    GTest::gmock_main
    CUDA::cudart
)

set_target_properties( testNeuralTextureSource PROPERTIES 
  CXX_STANDARD 14  # Required by latest gtest
  FOLDER NeuralTextures/Tests
)

# Register test cases with CTest.
gtest_discover_tests(testNeuralTextureSource PROPERTIES LABELS NeuralTextures)


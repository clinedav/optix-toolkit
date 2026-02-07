# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 11.1 REQUIRED )
endif()

embed_cuda(
  CONST HEADER DemandTextureSampler.h
  OUTPUT_TARGET
    demandTextureSamplerKernel
  FOLDER
    Examples/DemandLoading
  LIBRARIES
    OptiXToolkit::DemandLoading
    OptiXToolkit::NeuralTextures
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    DemandTextureSampler.cu
)

# Embed CMAKE_CURRENT_SOURCE_DIR in getSourceDir()
configure_file( SourceDir.h.in include/SourceDir.h @ONLY )

otk_add_executable( demandTextureSampler
  DemandTextureSampler.cpp
  DemandTextureSampler.h
  )

target_link_libraries( demandTextureSampler
  demandTextureSamplerKernel
  OptiXToolkit::DemandLoading
  OptiXToolkit::NeuralTextures
  OptiXToolkit::Gui
  OptiXToolkit::Util
  CUDA::cudart
  )

target_include_directories( demandTextureSampler PUBLIC 
  ${CMAKE_CURRENT_BINARY_DIR}/include )

set_target_properties( demandTextureSampler PROPERTIES
  FOLDER Examples/DemandLoading
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:demandTextureSampler> )
  add_custom_command( TARGET demandTextureSampler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:demandTextureSampler> $<TARGET_FILE_DIR:demandTextureSampler>
    COMMAND_EXPAND_LISTS )
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

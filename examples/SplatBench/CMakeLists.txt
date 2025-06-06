# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)
include(BuildConfig)
include(FetchGlfw)
include(FetchGlad)
include(embed_cuda)

embed_cuda(
  CONST HEADER SplatBenchKernelCuda.h
  OUTPUT_TARGET
    splatBenchKernel
  FOLDER
    Examples
  LIBRARIES
    OptiXToolkit::OTKAppBase
    OptiXToolkit::ShaderUtil
  SOURCES
    SplatBench.cu
    SplatIntersections.h
    SplatTypes.h
    SplatBenchParams.h
)

otk_add_executable( splatBench
  SplatBench.cpp
  SplatIntersections.h
  SplatTypes.h
  SplatBenchParams.h
  SplatLoader.h
  )

target_link_libraries( splatBench
  PUBLIC
  splatBenchKernel
  OptiXToolkit::OTKAppBase
  OptiXToolkit::Gui
  OptiXToolkit::ImageSources
  OptiXToolkit::Util
  )

set_target_properties( splatBench PROPERTIES
  FOLDER Examples
  INSTALL_RPATH ${OptiXToolkit_DIR}/../../OptiXToolkit )

# Copy shared libraries that the built executable depends on.
if( $<TARGET_RUNTIME_DLLS:splatBench> )
  add_custom_command( TARGET splatBench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:splatBench> $<TARGET_FILE_DIR:splatBench>
    COMMAND_EXPAND_LISTS )
endif()

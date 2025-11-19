# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(BuildConfig)

otk_add_library( NeuralTextures
  NeuralTextureSource.cpp
)

set_property(TARGET NeuralTextures PROPERTY FOLDER NeuralTextures)

add_library(OptiXToolkit::NeuralTextures ALIAS NeuralTextures)

target_sources(NeuralTextures
  PUBLIC 
  FILE_SET HEADERS 
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
  FILES
  NeuralTextureSource.h
)

target_include_directories( NeuralTextures
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

target_link_libraries( NeuralTextures
  PUBLIC
  OptiXToolkit::ImageSource
  CUDA::cuda_driver
  PRIVATE
  OptiXToolkit::Error
  )

set_target_properties(NeuralTextures PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

install(TARGETS NeuralTextures
  EXPORT NeuralTexturesTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/OptiXToolkit
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/OptiXToolkit/NeuralTextures
  )

install(EXPORT NeuralTexturesTargets
  FILE NeuralTexturesTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OptiXToolkit
  NAMESPACE OptiXToolkit::
  )

if( BUILD_TESTING )
  add_subdirectory( tests )
endif()


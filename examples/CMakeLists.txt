# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

#########################################################
# Welcome to the OptiX Toolkit (OTK)

# If you have any questions, we encourage you to post on the OptiX forums:
# https://devtalk.nvidia.com/default/board/90/

# CMake helper files are located in the CMake subdirectory.
list(APPEND CMAKE_MODULE_PATH 
  ${CMAKE_CURRENT_SOURCE_DIR}/CMake
  )

# Using the latest CMake is highly recommended, to ensure up-to-date CUDA language support.
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
include(Policies)
include(SetCxxStandard)

project(OptiXToolkitExamples LANGUAGES C CXX CUDA)
set_cxx_standard(11)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET OptiX::OptiX)
  # Set OptiX_INSTALL_DIR to the root of the OptiX SDK when configuring CMake.
  set(OptiX_INSTALL_DIR "OptiX_INSTALL_DIR-NOTFOUND" CACHE PATH "Path to OptiX installed location.")
  find_package(OptiX 7.3 REQUIRED)
endif()

add_subdirectory(Gui)
add_subdirectory(pbrtApi)
add_subdirectory(pbrtParser)
add_subdirectory(PbrtSceneLoader)
add_subdirectory(Simple)
add_subdirectory(Util)
add_subdirectory(tests)

# Skip examples if the libraries they depend upon were not built.
if(TARGET OptiXToolkit::CuOmmBaking)
  add_subdirectory(CuOmmBaking)
endif()
if(TARGET OptiXToolkit::DemandLoading)
  add_subdirectory(DemandLoading)
endif()

add_subdirectory(OTKAppBase)


// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "SplatTypes.h"

//#define USE_COOP_VEC 1
const int SPLATS_PER_BATCH = 64;

struct SplatBenchParams
{
    EllipsoidSplat* e_splats;
    HalfEllipsoidSplat* he_splats;
    HalfSphereSplat* s_splats;
    unsigned int* groupStartIdx;

    float splatScale;
};

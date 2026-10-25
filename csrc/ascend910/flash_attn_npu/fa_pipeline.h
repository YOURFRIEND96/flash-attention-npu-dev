#pragma once

#include <cstdint>

// Host allocation, AICPU metadata and device ring buffers must use one depth.
namespace fa_pipeline {
constexpr uint32_t PRE_LAUNCH = 2;
constexpr uint32_t SLOTS = PRE_LAUNCH + 1;
} // namespace fa_pipeline

#pragma once

#include "ghostrider/algorithms.cuh"

#include <array>

namespace ghostrider {

constexpr std::size_t kPipelineLength = 19;
using PipelineSchedule = std::array<AlgorithmId, kPipelineLength>;

PipelineSchedule default_schedule();
Digest256 run_pipeline(const Digest256& seed, const PipelineSchedule& schedule);

}  // namespace ghostrider

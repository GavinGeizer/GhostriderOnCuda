# Architecture Overview

## Current prototype modules
- `include/ghostrider/algorithms.cuh`: full GhostRider stage enum + dispatch API.
- `include/ghostrider/pipeline.cuh`: fixed pipeline schedule type and runner API.
- `cuda/algorithms.cu`: CUDA placeholder stage-mix kernel and per-stage parameter table.
- `src/pipeline.cu`: host orchestration for stage chaining.
- `src/main.cu`: executable smoke test wiring header+nonce -> digest.

## Intended evolution
1. Replace stage parameter placeholders with per-algorithm spec implementations.
2. Add schedule derivation logic from block/header entropy.
3. Introduce batched nonce search kernels and target checks.
4. Add correctness and performance test suites.

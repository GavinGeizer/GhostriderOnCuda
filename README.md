# GhostriderOnCuda

Early CUDA prototype for GhostRider-oriented hashing pipeline work.

## Current state
This repository contains **algorithm skeletons** for the full GhostRider stage set:

- X16R-derived family placeholders:
  Blake, BMW, Groestl, JH, Keccak, Skein, Luffa, CubeHash, Shavite, SIMD, Echo, Hamsi, Fugue, Shabal, Whirlpool, SHA-512.
- CryptoNight-family placeholders:
  CryptoNightR, CryptoNightFast, CryptoNightLite.

These are scaffolding kernels for interface/pipeline bring-up and are **not yet cryptographically correct**.

## Build (prototype)
```bash
cmake -S . -B build
cmake --build build -j
./build/ghostrider_prototype
```

## Next milestones
1. Replace placeholder stage mixing with spec-correct per-algorithm implementations.
2. Add schedule derivation rules from header/seed entropy.
3. Add nonce-range parallel search and target checking.
4. Add vector tests for every stage.

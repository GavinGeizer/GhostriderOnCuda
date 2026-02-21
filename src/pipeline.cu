#include "ghostrider/pipeline.cuh"

namespace ghostrider {

PipelineSchedule default_schedule() {
    return {
        AlgorithmId::Blake256,
        AlgorithmId::Bmw256,
        AlgorithmId::Groestl256,
        AlgorithmId::Jh256,
        AlgorithmId::Keccak256,
        AlgorithmId::Skein256,
        AlgorithmId::Luffa256,
        AlgorithmId::CubeHash256,
        AlgorithmId::Shavite256,
        AlgorithmId::Simd256,
        AlgorithmId::Echo256,
        AlgorithmId::Hamsi256,
        AlgorithmId::Fugue256,
        AlgorithmId::Shabal256,
        AlgorithmId::Whirlpool,
        AlgorithmId::Sha512,
        AlgorithmId::CryptoNightR,
        AlgorithmId::CryptoNightFast,
        AlgorithmId::CryptoNightLite,
    };
}

Digest256 run_pipeline(const Digest256& seed, const PipelineSchedule& schedule) {
    Digest256 state = seed;
    for (AlgorithmId id : schedule) {
        state = run_algorithm(id, state);
    }
    return state;
}

}  // namespace ghostrider

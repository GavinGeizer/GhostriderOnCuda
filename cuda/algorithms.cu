#include "ghostrider/algorithms.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ghostrider {
namespace {

#define CUDA_CHECK(stmt)                                                                 \
    do {                                                                                 \
        cudaError_t err = (stmt);                                                        \
        if (err != cudaSuccess) {                                                        \
            throw std::runtime_error(cudaGetErrorString(err));                           \
        }                                                                                \
    } while (0)

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t r) {
    return (x << r) | (x >> (32u - r));
}

__global__ void stage_mix_kernel(const uint32_t* in,
                                 uint32_t* out,
                                 uint32_t stage_seed,
                                 uint32_t rot_a,
                                 uint32_t rot_b) {
    const int idx = threadIdx.x;
    if (idx >= 8) return;

    uint32_t v = in[idx] ^ (stage_seed + static_cast<uint32_t>((idx + 1) * 0x9E3779B9u));
    v += rotl32(in[(idx + 1) & 7], rot_a);
    v ^= rotl32(in[(idx + 4) & 7], rot_b);
    out[idx] = v;
}

struct StageParams {
    uint32_t seed;
    uint32_t rot_a;
    uint32_t rot_b;
};

StageParams stage_params(AlgorithmId algorithm) {
    switch (algorithm) {
        case AlgorithmId::Blake256:
            return {0x243F6A88u, 16u, 7u};
        case AlgorithmId::Bmw256:
            return {0x85A308D3u, 11u, 17u};
        case AlgorithmId::Groestl256:
            return {0x13198A2Eu, 9u, 13u};
        case AlgorithmId::Jh256:
            return {0x03707344u, 7u, 21u};
        case AlgorithmId::Keccak256:
            return {0xA4093822u, 5u, 14u};
        case AlgorithmId::Skein256:
            return {0x299F31D0u, 12u, 8u};
        case AlgorithmId::Luffa256:
            return {0x082EFA98u, 10u, 6u};
        case AlgorithmId::CubeHash256:
            return {0xEC4E6C89u, 8u, 15u};
        case AlgorithmId::Shavite256:
            return {0x452821E6u, 13u, 9u};
        case AlgorithmId::Simd256:
            return {0x38D01377u, 17u, 5u};
        case AlgorithmId::Echo256:
            return {0xBE5466CFu, 6u, 11u};
        case AlgorithmId::Hamsi256:
            return {0x34E90C6Cu, 19u, 7u};
        case AlgorithmId::Fugue256:
            return {0xC0AC29B7u, 4u, 12u};
        case AlgorithmId::Shabal256:
            return {0xC97C50DDu, 14u, 10u};
        case AlgorithmId::Whirlpool:
            return {0x3F84D5B5u, 3u, 16u};
        case AlgorithmId::Sha512:
            return {0xB5470917u, 20u, 7u};
        case AlgorithmId::CryptoNightR:
            return {0x6A09E667u, 11u, 22u};
        case AlgorithmId::CryptoNightFast:
            return {0xBB67AE85u, 9u, 18u};
        case AlgorithmId::CryptoNightLite:
            return {0x3C6EF372u, 7u, 24u};
    }

    throw std::runtime_error("Unknown GhostRider algorithm ID");
}

Digest256 launch_digest_kernel(const Digest256& input, AlgorithmId algorithm) {
    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    Digest256 output{};

    const StageParams params = stage_params(algorithm);

    CUDA_CHECK(cudaMalloc(&d_in, sizeof(uint32_t) * kDigestWords));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uint32_t) * kDigestWords));

    CUDA_CHECK(cudaMemcpy(d_in, input.data(), sizeof(uint32_t) * kDigestWords,
                          cudaMemcpyHostToDevice));

    stage_mix_kernel<<<1, 8>>>(d_in, d_out, params.seed, params.rot_a, params.rot_b);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(), d_out, sizeof(uint32_t) * kDigestWords,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return output;
}

}  // namespace

Digest256 initialize_state_from_header(const uint8_t* header80, uint32_t nonce) {
    Digest256 state{};
    for (std::size_t i = 0; i < kDigestWords; ++i) {
        const std::size_t b = i * 4;
        state[i] = (static_cast<uint32_t>(header80[b + 0]) << 24) |
                   (static_cast<uint32_t>(header80[b + 1]) << 16) |
                   (static_cast<uint32_t>(header80[b + 2]) << 8) |
                   (static_cast<uint32_t>(header80[b + 3]) << 0);
    }
    state[7] ^= nonce;
    return state;
}

Digest256 run_algorithm(AlgorithmId id, const Digest256& input) {
    return launch_digest_kernel(input, id);
}

}  // namespace ghostrider

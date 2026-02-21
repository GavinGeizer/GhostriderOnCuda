#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace ghostrider {

// GhostRider combines an X16R-derived hash family with CryptoNight-family stages.
enum class AlgorithmId : uint8_t {
    Blake256 = 0,
    Bmw256,
    Groestl256,
    Jh256,
    Keccak256,
    Skein256,
    Luffa256,
    CubeHash256,
    Shavite256,
    Simd256,
    Echo256,
    Hamsi256,
    Fugue256,
    Shabal256,
    Whirlpool,
    Sha512,
    CryptoNightR,
    CryptoNightFast,
    CryptoNightLite,
};

constexpr std::size_t kDigestWords = 8;
using Digest256 = std::array<uint32_t, kDigestWords>;

// Seed state from an 80-byte block header + nonce.
Digest256 initialize_state_from_header(const uint8_t* header80, uint32_t nonce);

// Generic dispatcher for algorithm skeleton stages.
Digest256 run_algorithm(AlgorithmId id, const Digest256& input);

}  // namespace ghostrider

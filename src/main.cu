#include "ghostrider/algorithms.cuh"
#include "ghostrider/pipeline.cuh"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    uint8_t header[80]{};
    for (uint8_t i = 0; i < 80; ++i) {
        header[i] = i;
    }

    constexpr uint32_t nonce = 0x12345678u;
    const auto seed = ghostrider::initialize_state_from_header(header, nonce);
    const auto digest = ghostrider::run_pipeline(seed, ghostrider::default_schedule());

    std::cout << "prototype digest: ";
    for (uint32_t w : digest) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << w;
    }
    std::cout << std::endl;

    return 0;
}

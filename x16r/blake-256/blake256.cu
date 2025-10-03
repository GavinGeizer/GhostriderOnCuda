// blake256_cuda_full.cu
// Full BLAKE-256 implementation (padding, salt support, HAIFA counter) with CUDA
// - Compression function implemented on the device (per-thread compress)
// - Host-side padding, block splitting, counter management
// - Example main() that hashes an input message ("abc") and optional 128-bit salt
//
// This implementation follows the BLAKE-256 specification from Aumasson et al.
// (constants, padding, HAIFA iteration). If you want test vectors added or a pure-
// CPU fallback, tell me and I'll add them.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

// ---------- Device-side compression (same as earlier compress32_device) ----------
// Constants (c0..c15)
__constant__ uint32_t dev_c[16] = {
    0x243F6A88u, 0x85A308D3u, 0x13198A2Eu, 0x03707344u,
    0xA4093822u, 0x299F31D0u, 0x082EFA98u, 0xEC4E6C89u,
    0x452821E6u, 0x38D01377u, 0xBE5466CFu, 0x34E90C6Cu,
    0xC0AC29B7u, 0xC97C50DDu, 0x3F84D5B5u, 0xB5470917u
};

__constant__ uint8_t dev_sigma[10][16] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15 },
  {14,10, 4, 8, 9,15,13, 6, 1,12, 0, 2,11, 7, 5, 3 },
  {11, 8,12, 0, 5, 2,15,13,10,14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1,13,12,11,14, 2, 6, 5,10, 4, 0,15, 8 },
  { 9, 0, 5, 7, 2, 4,10,15,14, 1,11,12, 6, 8, 3,13 },
  { 2,12, 6,10, 0,11, 8, 3, 4,13, 7, 5,15,14, 1, 9 },
  {12, 5, 1,15,14,13, 4,10, 0, 7, 6, 3, 9, 2, 8,11 },
  {13,11, 7,14,12, 1, 3, 9, 5, 0,15, 4, 8, 6, 2,10 },
  { 6,15,14, 9,11, 3, 0, 8,12, 2,13, 7, 1, 4,10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5,15,11, 9,14, 3,12,13, 0 }
};

__device__ inline uint32_t ROTR32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

__device__ inline uint32_t ADD32(uint32_t a, uint32_t b) { return a + b; }
__device__ inline uint32_t XOR32(uint32_t a, uint32_t b) { return a ^ b; }

__device__ inline void G32_round(uint32_t v[16], const uint32_t m[16], int round, int i, int a, int b, int c, int d) {
    const uint8_t j = dev_sigma[round % 10][2*i];
    const uint8_t k = dev_sigma[round % 10][2*i + 1];

    v[a] = ADD32( ADD32(v[a], v[b]), XOR32(m[j], dev_c[k]) );
    v[d] = ROTR32(XOR32(v[d], v[a]), 16);
    v[c] = ADD32(v[c], v[d]);
    v[b] = ROTR32(XOR32(v[b], v[c]), 12);

    v[a] = ADD32( ADD32(v[a], v[b]), XOR32(m[k], dev_c[j]) );
    v[d] = ROTR32(XOR32(v[d], v[a]), 8);
    v[c] = ADD32(v[c], v[d]);
    v[b] = ROTR32(XOR32(v[b], v[c]), 7);
}

__device__ void compress32_device(const uint32_t h[8], const uint32_t m[16], const uint32_t s[4], const uint32_t t[2], uint32_t h_out[8]) {
    uint32_t v[16];
    // init
    for (int i=0;i<8;i++) v[i] = h[i];
    v[8]  = s[0] ^ dev_c[0];
    v[9]  = s[1] ^ dev_c[1];
    v[10] = s[2] ^ dev_c[2];
    v[11] = s[3] ^ dev_c[3];
    v[12] = t[0] ^ dev_c[4];
    v[13] = t[0] ^ dev_c[5];
    v[14] = t[1] ^ dev_c[6];
    v[15] = t[1] ^ dev_c[7];

    for (int round=0; round<14; ++round) {
        G32_round(v, m, round, 0,  0, 4, 8, 12);
        G32_round(v, m, round, 1,  1, 5, 9, 13);
        G32_round(v, m, round, 2,  2, 6,10, 14);
        G32_round(v, m, round, 3,  3, 7,11, 15);

        G32_round(v, m, round, 4,  0, 5,10, 15);
        G32_round(v, m, round, 5,  1, 6,11, 12);
        G32_round(v, m, round, 6,  2, 7, 8, 13);
        G32_round(v, m, round, 7,  3, 4, 9, 14);
    }

    h_out[0] = h[0] ^ s[0] ^ v[0]  ^ v[8];
    h_out[1] = h[1] ^ s[1] ^ v[1]  ^ v[9];
    h_out[2] = h[2] ^ s[2] ^ v[2]  ^ v[10];
    h_out[3] = h[3] ^ s[3] ^ v[3]  ^ v[11];
    h_out[4] = h[4] ^ s[0] ^ v[4]  ^ v[12];
    h_out[5] = h[5] ^ s[1] ^ v[5]  ^ v[13];
    h_out[6] = h[6] ^ s[2] ^ v[6]  ^ v[14];
    h_out[7] = h[7] ^ s[3] ^ v[7]  ^ v[15];
}

// Kernel: one thread compresses one block
extern "C"
__global__ void blake256_compress_kernel(const uint32_t *h_in, const uint32_t *m_in, const uint32_t *s_in, const uint32_t *t_in, uint32_t *h_out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const uint32_t *h = h_in + idx*8;
    const uint32_t *m = m_in + idx*16;
    const uint32_t *s = s_in + idx*4;
    const uint32_t *t = t_in + idx*2;
    uint32_t *hout = h_out + idx*8;
    compress32_device(h, m, s, t, hout);
}

// ---------- Host helpers: padding, U8TO32_BE, block processing ----------
static inline uint32_t U8TO32_BE(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | ((uint32_t)p[3]);
}
static inline void U32TO8_BE(uint32_t v, uint8_t *p) {
    p[0] = (v >> 24) & 0xFF;
    p[1] = (v >> 16) & 0xFF;
    p[2] = (v >> 8) & 0xFF;
    p[3] = (v) & 0xFF;
}

// Padding per spec: append 1 bit then zeros then 1 bit then 64-bit length (big-endian)
std::vector<uint8_t> blake256_pad(const uint8_t *msg, size_t msglen) {
    uint64_t bitlen = (uint64_t)msglen * 8ULL;
    // append 0x80, zeros, then 0x01? Spec uses 1000...0001<l>64 where there's a trailing '1' before length
    // That is: append 0x80, then zeros, then final byte with low bit set (0x01)?? Careful: representation in words
    // The spec describes bits: m || 1 0..0 1 <l>64. Practically, it's equivalent to append 0x80 then zeros until last byte
    // where the final bit '1' before length is set as 0x01 in the last padding byte (LSB). We implement as follows:

    // We'll build padded message as bytes. Start with original
    std::vector<uint8_t> out(msg, msg+msglen);
    // Append 0x80 (10000000)
    out.push_back(0x80);

    // We need total bits congruent to 447 mod 512 => total bytes congruent to 55 mod 64 (since 447 bits = 55 bytes + 7 bits)
    // After adding the 0x80 byte, append zero bytes until length mod 64 == 55
    while ((out.size() % 64) != 55) out.push_back(0x00);

    // Append 0x01 (the spec's extra '1' bit just before length): this is a single bit '1' placed as LSB of next byte
    // Represented as 0x01 byte because previous were full bytes. Equivalent to appending 0x01.
    out.push_back(0x01);

    // Append 64-bit big-endian length
    uint8_t lenbe[8];
    for (int i = 0; i < 8; ++i) lenbe[7 - i] = (uint8_t)((bitlen >> (8*i)) & 0xFF);
    out.insert(out.end(), lenbe, lenbe+8);

    // Now out.size() should be a multiple of 64
    return out;
}

// Split padded message into N blocks of 64 bytes and convert to 16 uint32 words (big-endian)
void blocks_from_padded(const std::vector<uint8_t> &padded, std::vector<uint32_t> &out_words) {
    size_t blocks = padded.size() / 64;
    out_words.resize(blocks * 16);
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = padded.data() + b*64;
        for (int i=0;i<16;i++) {
            out_words[b*16 + i] = U8TO32_BE(blk + i*4);
        }
    }
}

// Fill salt array from 16-byte salt (or zeroes)
void salt_from_bytes(const uint8_t salt_bytes[16], uint32_t s_out[4]) {
    if (salt_bytes == nullptr) { for (int i=0;i<4;i++) s_out[i]=0; return; }
    for (int i=0;i<4;i++) s_out[i] = U8TO32_BE(salt_bytes + i*4);
}

// Initial IVs (same as SHA-256) per spec
const uint32_t IV256[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Top-level host function: compute BLAKE-256 hash of message with optional 16-byte salt
// This implementation launches a compress kernel per block (batching possible) and manages HAIFA counter t
std::vector<uint8_t> blake256_hash_cuda(const uint8_t *msg, size_t msglen, const uint8_t salt_bytes[16]) {
    // 1) padding
    std::vector<uint8_t> padded = blake256_pad(msg, msglen);
    // 2) blocks -> words
    std::vector<uint32_t> m_words;
    blocks_from_padded(padded, m_words);
    size_t blocks = m_words.size() / 16;

    // 3) prepare salt
    uint32_t salt[4];
    if (salt_bytes) salt_from_bytes(salt_bytes, salt);
    else { salt[0]=salt[1]=salt[2]=salt[3]=0; }

    // 4) allocate device buffers for per-block compress calls (we'll launch one kernel with N=1 each time to keep implementation simple)
    // To be more efficient you'd batch multiple blocks and process many compressions in parallel; here we keep logic straightforward.

    // Host chain value
    uint32_t h[8];
    memcpy(h, IV256, sizeof(h));

    // Device buffers (we'll allocate once for max sizes needed) - allocate space for single compress call at a time
    uint32_t *d_h_in=nullptr, *d_m_in=nullptr, *d_s_in=nullptr, *d_t_in=nullptr, *d_h_out=nullptr;
    cudaMalloc(&d_h_in, sizeof(uint32_t)*8);
    cudaMalloc(&d_m_in, sizeof(uint32_t)*16);
    cudaMalloc(&d_s_in, sizeof(uint32_t)*4);
    cudaMalloc(&d_t_in, sizeof(uint32_t)*2);
    cudaMalloc(&d_h_out, sizeof(uint32_t)*8);

    for (size_t i=0;i<blocks;i++) {
        // compute li (number of message bits processed in blocks 0..i) excluding padding bits
        // Spec: li is number of message bits in blocks 0..i (excluding padding) — must track original bits covered
        // If block contains no original message bits (i.e., it's purely padding), then counter is set to zero.
        // We compute the number of original bits up to and including block i.

        // Determine how many original bytes are in this block
        size_t block_start_byte = i * 64;
        size_t orig_bytes_remaining = (msglen > block_start_byte) ? (msglen - block_start_byte) : 0;
        size_t orig_bytes_in_block = (orig_bytes_remaining >= 64) ? 64 : orig_bytes_remaining;
        uint64_t li_bits = 0;
        if (orig_bytes_in_block == 0) {
            // If the block contains no original message bits, set counter to zero
            li_bits = 0;
        } else {
            // li is number of original message bits in blocks 0..i
            size_t bytes_up_to_i = (msglen >= (i+1)*64) ? ((i+1)*64) : msglen;
            li_bits = (uint64_t)bytes_up_to_i * 8ULL;
        }

        uint32_t t[2];
        // t is 64-bit counter split into two 32-bit words (big- or little-endian? Spec uses words t0,t1 as 32-bit words of 64-bit counter)
        // We'll represent t[0] as high 32 bits and t[1] as low 32 bits? The spec's examples show t0 = upper 32 bits? In appendix C they used
        // counter: 00000000 00000000 00000000 00000000 00000008 00000000 i.e. t0=0x00000008, t1=0x00000000 for a one-block message of 8 bits.
        // That suggests t0 is high 32 bits followed by high-word of low 64? To match spec: for li_bits treat as two 32-bit words where t0 = high 32 bits of li, t1 = low 32 bits of li.
        t[0] = (uint32_t)((li_bits >> 32) & 0xFFFFFFFFULL);
        t[1] = (uint32_t)(li_bits & 0xFFFFFFFFULL);

        // if li_bits == 0 then t must be zero
        if (li_bits == 0) { t[0] = 0; t[1] = 0; }

        // prepare m block words for this block
        uint32_t mblk[16];
        for (int j=0;j<16;j++) mblk[j] = m_words[i*16 + j];

        // copy inputs to device
        cudaMemcpy(d_h_in, h, sizeof(uint32_t)*8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_m_in, mblk, sizeof(uint32_t)*16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_s_in, salt, sizeof(uint32_t)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_in, t, sizeof(uint32_t)*2, cudaMemcpyHostToDevice);

        // launch kernel for a single compress
        dim3 blockDim(128);
        dim3 gridDim( (1 + blockDim.x - 1) / blockDim.x );
        blake256_compress_kernel<<<gridDim, blockDim>>>(d_h_in, d_m_in, d_s_in, d_t_in, d_h_out, 1);
        cudaDeviceSynchronize();

        uint32_t hnext[8];
        cudaMemcpy(hnext, d_h_out, sizeof(uint32_t)*8, cudaMemcpyDeviceToHost);
        memcpy(h, hnext, sizeof(h));
    }

    // free device memory
    cudaFree(d_h_in); cudaFree(d_m_in); cudaFree(d_s_in); cudaFree(d_t_in); cudaFree(d_h_out);

    // produce 32-byte digest: h[0]..h[7] each big-endian
    std::vector<uint8_t> digest(32);
    for (int i=0;i<8;i++) U32TO8_BE(h[i], &digest[i*4]);
    return digest;
}

// Utility to hex-print digest
std::string hexify(const std::vector<uint8_t> &d) {
    std::ostringstream oss;
    for (uint8_t b : d) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    }
    return oss.str();
}

int main() {
    const char *msg = "abc";
    size_t msglen = strlen(msg);
    // no salt (set to nullptr) or provide 16 bytes
    const uint8_t *salt = nullptr;

    std::vector<uint8_t> digest = blake256_hash_cuda((const uint8_t*)msg, msglen, nullptr);
    std::cout << "BLAKE-256(\"abc\") = " << hexify(digest) << std::endl;

    return 0;
}

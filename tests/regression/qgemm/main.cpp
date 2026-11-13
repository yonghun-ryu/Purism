#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include <tensor_cfg.h>

#define RT_CHECK(_expr) do { \
    int _ret = (_expr); \
    if (_ret != 0) { \
        printf("Error: '%s' returned %d!\n", #_expr, _ret); \
        cleanup(); exit(-1); \
    } \
} while (0)

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;

static const char* kernel_file = "kernel.vxbin";

static uint32_t xM  = QGEMM_M;
static uint32_t xN  = QGEMM_N;
static uint32_t xK  = QGEMM_K;
static uint32_t xGS = GROUP_SIZE;
static bool     xZPArr = false;   // -Z: per-group per-column zero points (Purism FP16 path)

static vx_device_h device       = nullptr;
static vx_buffer_h A_buf        = nullptr;
static vx_buffer_h B_int4_buf   = nullptr;
static vx_buffer_h C_buf        = nullptr;
static vx_buffer_h sw_buf       = nullptr;
static vx_buffer_h zp_buf       = nullptr;
static vx_buffer_h krnl_buf     = nullptr;
static vx_buffer_h args_buf     = nullptr;

static kernel_arg_t karg = {};

static void show_usage() {
    std::cout << "W4A16 GEMM (FP16 activations, INT4 weights, per-group scale) on the Purism TCU.\n";
    std::cout << "Usage: [-m M] [-n N] [-k K] [-g group_size] [-Z] [-h]\n";
    std::cout << "  -Z: per-group per-column zero points\n";
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "m:n:k:g:Zh")) != -1) {
        switch (c) {
        case 'm': xM  = atoi(optarg); break;
        case 'n': xN  = atoi(optarg); break;
        case 'k': xK  = atoi(optarg); break;
        case 'g': xGS = atoi(optarg); break;
        case 'Z': xZPArr = true; break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

static void cleanup() {
    if (!device) return;
    vx_mem_free(A_buf);
    vx_mem_free(B_int4_buf);
    vx_mem_free(C_buf);
    vx_mem_free(sw_buf);
    if (zp_buf) vx_mem_free(zp_buf);
    vx_mem_free(krnl_buf);
    vx_mem_free(args_buf);
    vx_dev_close(device);
}

// INT4S nibble sign-extension: 4-bit two's complement → int8_t
// Mirrors the RTL: bw_mask with src_bw=4, sign-extend via upper-fill.
static inline int8_t int4s_sext(uint8_t nib) {
    nib &= 0xF;
    return (nib & 0x8) ? (int8_t)(nib | 0xF0) : (int8_t)nib;
}

// IEEE-754 half<->float (round-to-nearest-even, subnormals via the standard
// path). Used only by the QGEMM_FP16 microbench to mirror the kernel's FP16
// activation and the OCU INT4->FP16 weight dequant.
static inline uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) // Inf/NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);            // overflow -> Inf
    if (exp <= 0) {                                               // subnormal/zero
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        uint32_t rem  = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;         // round-to-even
    return h;
}
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {                                                    // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &out, 4); return f;
}

// Weight-only FP16 reference: mirrors the QGEMM_FP16 kernel.
//   A_fp16[M][K] (uint16 bits) x dequant_fp16(B_int4[K][N]) -> FP32 TCU acc,
//   then x scale_w[g][n] per group, FP32 accumulate. Optional per-group per-column
//   zero points (-Z). INT4 values |v|<=8 are exact in FP16.
static void qgemm_fp16_ref(float* C, const uint16_t* A, const int8_t* B_i4_raw,
                           const float* scale_w, uint32_t M, uint32_t N,
                           uint32_t K, uint32_t gs, const int8_t* zp_arr = nullptr) {
    uint32_t G = K / gs;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (uint32_t g = 0; g < G; ++g) {
                float partial = 0.f;
                for (uint32_t k = g * gs; k < (g + 1) * gs; ++k) {
                    float a = f16_to_f32(A[m * K + k]);
                    float b = (float)int4s_sext((uint8_t)B_i4_raw[k * N + n])
                            - (zp_arr ? (float)zp_arr[g * N + n] : 0.f);
                    partial += a * b;
                }
                acc += partial * scale_w[g * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

int main(int argc, char** argv) {
    parse_args(argc, argv);

    uint32_t M = xM, N = xN, K = xK, gs = xGS;

    uint32_t G = K / gs;

    if (M % cfg::tileM || N % cfg::tileN || K % cfg::tileK) {
        printf("Error: M/N/K must be multiples of tileM=%u/tileN=%u/tileK=%u\n",
               cfg::tileM, cfg::tileN, cfg::tileK);
        return -1;
    }
    if (K % gs) { printf("Error: K must be divisible by group_size\n"); return -1; }
    // The kernel's group loop assumes each tileK step stays within one quant group
    // (one scale_w per group), i.e. group_size >= tileK. At NUM_THREADS>=8 the INT8
    // path has tileK=32, so a smaller group (e.g. 16) would span 2 groups per tile
    // and silently use the wrong scale -> require gs >= tileK. (Also gs % tileK == 0
    // so groups tile cleanly.)
    if (gs < cfg::tileK || gs % cfg::tileK) {
        printf("Error: group_size (%u) must be a multiple of tileK (%u) and >= tileK "
               "(kernel processes one group per tileK step)\n", gs, cfg::tileK);
        return -1;
    }

    printf("qgemm: M=%u N=%u K=%u group_size=%u G=%u zp=%s\n", M, N, K, gs, G, xZPArr ? "per-group" : "none");
    printf("TCU tile: M=%u N=%u K=%u  (threads=%u)\n",
           cfg::tileM, cfg::tileN, cfg::tileK, NUM_THREADS);

    RT_CHECK(vx_dev_open(&device));

    uint64_t isa;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa));
    if (!(isa & VX_ISA_EXT_TCU)) {
        printf("Error: TCU extension not present in this xclbin!\n");
        cleanup(); return -1;
    }

    // B_int4 layout: uint32_t[K/2][N/4] — 2-row × 4-col per 32-bit word (8 INT4 nibbles)
    size_t sA     = (size_t)M * K * sizeof(uint16_t);   // FP16 activation
    size_t sB_i4  = (K / 2) * (N / 4) * sizeof(uint32_t);
    size_t sC     = M * N * sizeof(float);
    size_t sSW    = G * N * sizeof(float);

    RT_CHECK(vx_mem_alloc(device, sA,    VX_MEM_READ,  &A_buf));
    RT_CHECK(vx_mem_alloc(device, sB_i4, VX_MEM_READ,  &B_int4_buf));
    RT_CHECK(vx_mem_alloc(device, sC,    VX_MEM_WRITE, &C_buf));
    RT_CHECK(vx_mem_alloc(device, sSW,   VX_MEM_READ,  &sw_buf));

    RT_CHECK(vx_mem_address(A_buf,      &karg.A_addr));
    RT_CHECK(vx_mem_address(B_int4_buf, &karg.B_int4_addr));
    RT_CHECK(vx_mem_address(C_buf,      &karg.C_addr));
    RT_CHECK(vx_mem_address(sw_buf,     &karg.scale_w_addr));

    // Generate data
    std::srand(42);
    // hB_i4_raw: raw INT4S values in [-7, 7] stored as int8_t (low nibble valid)
    std::vector<uint16_t> hA(M * K);                  // FP16 activation bits
    std::vector<int8_t>   hB_i4_raw(K * N);
    std::vector<uint32_t> hB_int4((K / 2) * (N / 4), 0);
    std::vector<float>    hSW(G * N), hC(M * N, 0.f), hRef(M * N);
    std::vector<int8_t>   hZP(xZPArr ? (size_t)G * N : 0);   // per-group per-column zero points
    for (auto& v : hZP) v = (int8_t)(rand() % 9 - 4);         // [-4, 4]: (q - zp) stays exact in FP16

    // small random FP16 activations in ~[-2, 2)
    for (auto& v : hA) v = f32_to_f16((float)(rand() % 400 - 200) / 100.0f);

    for (auto& v : hB_i4_raw) v = (int8_t)(rand() % 15 - 7);   // INT4S: [-7, 7]
    for (auto& v : hSW)       v = 1.f / 128.f + (float)(rand() % 100) / 1e5f;

    // Pack INT4 B matrix: uint32_t[K/2][N/4], 8 nibbles per word.
    // Word layout: nibbles 0..3 = row k, cols n..n+3; nibbles 4..7 = row k+1, same cols.
    for (uint32_t k = 0; k < K; k += 2) {
        for (uint32_t n = 0; n < N; n += 4) {
            uint32_t word = 0;
            for (uint32_t c = 0; c < 4; ++c) {
                uint32_t nib0 = (uint32_t)hB_i4_raw[ k      * N + n + c] & 0xF;
                uint32_t nib1 = (uint32_t)hB_i4_raw[(k + 1) * N + n + c] & 0xF;
                word |= nib0 << (c * 4);
                word |= nib1 << ((c + 4) * 4);
            }
            hB_int4[(k / 2) * (N / 4) + n / 4] = word;
        }
    }

    karg.M          = M;
    karg.N          = N;
    karg.K          = K;
    karg.group_size = gs;
    karg.grid_dim[0]  = N / cfg::tileN;
    karg.grid_dim[1]  = M / cfg::tileM;
    karg.block_dim[0] = NUM_THREADS;
    karg.block_dim[1] = 1;

    RT_CHECK(vx_copy_to_dev(A_buf,      hA.data(),      0, sA));
    // Repack B into fragB order for the in-pipeline OT path
    // (kernel.cpp mma_ot_i4). Replicates the kernel's direct-fragB per-lane
    // mapping exactly and asserts bijective K-pair coverage. Same byte size as
    // hB_int4, so the device buffer is reused; the golden still uses hB_int4.
    std::vector<uint32_t> hB_rep;
    {
        const uint32_t NT  = NUM_THREADS;
        const uint32_t NTN = N / cfg::tileN;
        const uint32_t NTK = K / cfg::tileK;
        hB_rep.assign((size_t)NTK * NTN * NT * 2, 0u);
        std::vector<uint8_t> used((size_t)(K / 2) * N, 0);
        constexpr uint32_t n_stride = cfg::b_sub_blocks * cfg::tcN;
        constexpr uint32_t k_stride = cfg::tcK * cfg::i_ratio;
        for (uint32_t kt = 0; kt < NTK; ++kt)
        for (uint32_t nt = 0; nt < NTN; ++nt)
        for (uint32_t lane = 0; lane < NT; ++lane) {
            uint32_t k_abs = kt * cfg::tileK, tile_col = nt * cfg::tileN, b_col_idx = tile_col / 4;
            uint32_t bidx = (cfg::b_block_size == NT) ? 0u : (lane / cfg::b_block_size);
            uint32_t lib  = (cfg::b_block_size == NT) ? lane : (lane % cfg::b_block_size);
            uint32_t b_block_col = (lib / cfg::tcK) + (bidx * cfg::tcN);
            uint32_t b_block_row = (lib % cfg::tcK) * cfg::i_ratio;
            for (uint32_t r = 0; r < cfg::NRB; ++r) {
                uint32_t k2 = b_block_row + (r / cfg::b_sub_steps) * k_stride;
                uint32_t n  = b_block_col + (r % cfg::b_sub_steps) * n_stride;
                uint32_t localN = n & 3u;
                uint32_t word = hB_int4[(k_abs + k2) / 2 * (N / 4) + b_col_idx + (n / 4)];
                uint32_t lo = (word >> (4 * localN))       & 0xFu;   // row k2
                uint32_t hi = (word >> (4 * (4 + localN))) & 0xFu;   // row k2+1
                uint32_t w = r / 4, sidx = r % 4;
                size_t idx = ((size_t)(kt * NTN + nt) * NT + lane) * 2 + w;
                hB_rep[idx] |= (lo << (8 * sidx)) | (hi << (8 * sidx + 4));
                size_t cov = (size_t)((k_abs + k2) / 2) * N + (tile_col + n);
                if (used[cov]++) { printf("repack: duplicate K-pair coverage\n"); cleanup(); exit(-1); }
            }
        }
        for (auto u : used) if (u != 1) { printf("repack: K-pair coverage hole\n"); cleanup(); exit(-1); }
        printf("repack: %zu words, coverage OK\n", hB_rep.size());
    }
    RT_CHECK(vx_copy_to_dev(B_int4_buf, hB_rep.data(), 0, sB_i4));
    RT_CHECK(vx_copy_to_dev(sw_buf,     hSW.data(),     0, sSW));
    karg.zp_addr = 0;
    if (xZPArr) {
        RT_CHECK(vx_mem_alloc(device, hZP.size(), VX_MEM_READ, &zp_buf));
        RT_CHECK(vx_mem_address(zp_buf, &karg.zp_addr));
        RT_CHECK(vx_copy_to_dev(zp_buf, hZP.data(), 0, hZP.size()));
    }

    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buf));
    RT_CHECK(vx_upload_bytes(device, &karg, sizeof(karg), &args_buf));

    // Run kernel and measure time
    auto t0 = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buf, args_buf));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops        = 2.0 * M * N * K;           // multiply + accumulate
    double gops       = ops / (elapsed_ms * 1e6);  // GOPS

    printf("Elapsed: %.2f ms\n", elapsed_ms);
    printf("Throughput: %.4f GOPS\n", gops);

    // Download result
    RT_CHECK(vx_copy_from_dev(hC.data(), C_buf, 0, sC));

    // CPU reference: follows hardware pipeline order (INT4 sign-ext → ZP subtract → GEMM → PSSU)
    qgemm_fp16_ref(hRef.data(), hA.data(), hB_i4_raw.data(),
                   hSW.data(), M, N, K, gs, xZPArr ? hZP.data() : nullptr);

    // Verify
    int errors = 0;
    const int MAX_ERRORS = 10;
    const float REL_TOL  = 1e-4f;
    double max_rel = 0.0, max_abs = 0.0; uint32_t exact = 0, nans = 0;
    for (uint32_t i = 0; i < M * N; ++i) {
        float ref = hRef[i];
        float got = hC[i];
        if (std::isnan(got) || std::isnan(ref)) ++nans;
        float diff = std::fabs(got - ref);
        double rel = diff / (std::fabs(ref) + 1e-7f);
        if (rel > max_rel) max_rel = rel;
        if (diff > max_abs) max_abs = diff;
        if (got == ref) ++exact;
        float tol  = REL_TOL * std::fabs(ref) + 1e-7f;
        if (diff > tol) {
            if (errors < MAX_ERRORS)
                printf("MISMATCH [%u]: got=%f ref=%f\n", i, got, ref);
            ++errors;
        }
    }
    printf("ACCURACY: max_rel=%.3e max_abs=%.3e bit_exact=%u/%u nan=%u\n", max_rel, max_abs, exact, M * N, nans);

    cleanup();

    if (errors) {
        printf("FAILED (%d errors / %u)\n", errors, M * N);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}

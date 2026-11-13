// Purism format-matrix GEMM test: B source formats I4/U4/I8/U8/FP8(E4M3,E5M2)/FP4 converted
// by the OCU (TCU operand transformer) into FP16 or INT8, per-group per-column zero
// points, group scales (two-scale for INT8) applied by the PSSU on the TCU output path.
#include "common.h"
#include <cmath>
#include <cstring>
#include <fstream>
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
#ifdef PG_INT8
using cfg = vt::wmma_config_t<NUM_THREADS, vt::int8, vt::int32>;
#else
using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;
#endif
static constexpr uint32_t NP     = cfg::NRB >> PG_LG_PACK;
static constexpr uint32_t SRC_BW = PG_SRC_4BIT ? 4 : 8;

static const char* kernel_file = "kernel.vxbin";
static uint32_t xM = 16, xN = 32, xK = 128, xGS = 128;
static bool     xZP = false;
static std::string xCaseFile, xOutFile;

static vx_device_h device = nullptr;
static vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, sw_buf = nullptr,
                   sa_buf = nullptr, zp_buf = nullptr, krnl_buf = nullptr, args_buf = nullptr;

static void cleanup() {
    if (!device) return;
    for (auto b : {A_buf, B_buf, C_buf, sw_buf, sa_buf, zp_buf, krnl_buf, args_buf})
        if (b) vx_mem_free(b);
    vx_dev_close(device);
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "m:n:k:g:Zf:o:h")) != -1) {
        switch (c) {
        case 'm': xM = atoi(optarg); break;
        case 'n': xN = atoi(optarg); break;
        case 'k': xK = atoi(optarg); break;
        case 'g': xGS = atoi(optarg); break;
        case 'Z': xZP = true; break;
        case 'f': xCaseFile = optarg; break;
        case 'o': xOutFile = optarg; break;
        default:
            printf("Usage: [-m M] [-n N] [-k K] [-g group_size] [-Z per-group zero points]\n"
                   "       [-f case file (I4 source)] [-o C output, f32 [M][N]]\n");
            exit(c == 'h' ? 0 : -1);
        }
    }
}

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return h;
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FFu, out;
    if (exp == 0) {
        if (!mant) out = sign;
        else { exp = 113; while (!(mant & 0x400u)) { mant <<= 1; exp--; } out = sign | (exp << 23) | ((mant & 0x3FFu) << 13); }
    } else out = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &out, 4); return f;
}

// FP8 decode (OCP): E4M3 bias 7 (0x7F/0xFF NaN, no Inf); E5M2 bias 15 (IEEE-like)
static float fp8_to_f32(uint8_t v, bool e4m3) {
    int es = e4m3 ? 4 : 5, ms = 7 - es, bias = e4m3 ? 7 : 15;
    int s = v >> 7, e = (v >> ms) & ((1 << es) - 1), m = v & ((1 << ms) - 1);
    float mag = (e == 0) ? std::ldexp((float)m, 1 - bias - ms)
                         : std::ldexp((float)((1 << ms) | m), e - bias - ms);
    return s ? -mag : mag;
}

// FP4 decode (OCP E2M1): no Inf/NaN
static float fp4_to_f32(uint32_t v) {
    static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (v & 8) ? -mag[v & 7] : mag[v & 7];
}

static bool src_signed() { return PG_SRC == 11 || PG_SRC == 9; }

// integer source code -> value minus zero point (both interpreted in the source format)
static int32_t int_deq(uint32_t code, int8_t zp_byte) {
    uint32_t mask = (1u << SRC_BW) - 1;
    int32_t q = (int32_t)(code & mask), z = (int32_t)((uint8_t)zp_byte & mask);
    if (src_signed()) {
        if (q & (1 << (SRC_BW - 1))) q -= (1 << SRC_BW);
        if (z & (1 << (SRC_BW - 1))) z -= (1 << SRC_BW);
    }
    return q - z;
}

static uint32_t rand_code() {
    for (;;) {
        uint32_t v = (uint32_t)rand() & ((1u << SRC_BW) - 1);
        if (PG_SRC == 5 && (v & 0x7F) == 0x7F) continue;              // E4M3 NaN
        if (PG_SRC == 6 && ((v >> 2) & 0x1F) == 0x1F) continue;       // E5M2 Inf/NaN
        return v;
    }
}

// -f: GEMM case file, I4 source only. Header magic + u32
// {M, N, K, gs, has_zp}; "GEMMCASE" (FP16 build): u16 A_fp16[M][K]; "GEMMCAS8" (INT8
// build): i8 A[M][K] + f32 sa[M]; then u32 B[K/2][N/4] (word (k/2, n/4) holds W[k][n]
// at bits (k%2)*16 + (n%4)*4), f32 sw[G][N], (i8 zp[G][N]).
struct gemm_case_t {
    uint32_t M, N, K, gs;
    std::vector<uint8_t>  A;       // raw A bytes (fp16 or int8)
    std::vector<float>    sa, sw;
    std::vector<uint32_t> codes;   // 4-bit codes [K][N]
    std::vector<int8_t>   zp;
};
static bool load_case(const std::string& path, gemm_case_t& c) {
#ifdef PG_INT8
    const char* want = "GEMMCAS8"; const uint32_t a_bytes = 1;
#else
    const char* want = "GEMMCASE"; const uint32_t a_bytes = 2;
#endif
    if (PG_SRC != 11) { printf("Error: -f needs PG_SRC=11 (I4)\n"); return false; }
    std::ifstream f(path, std::ios::binary);
    char magic[8]; uint32_t h[5];
    f.read(magic, 8); f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (!f || memcmp(magic, want, 8)) { printf("Error: '%s' is not a %s case\n", path.c_str(), want); return false; }
    c.M = h[0]; c.N = h[1]; c.K = h[2]; c.gs = h[3];
    uint32_t G = c.K / c.gs;
    c.A.resize((size_t)c.M * c.K * a_bytes);
    f.read(reinterpret_cast<char*>(c.A.data()), c.A.size());
#ifdef PG_INT8
    c.sa.resize(c.M);
    f.read(reinterpret_cast<char*>(c.sa.data()), c.sa.size() * 4);
#endif
    std::vector<uint32_t> w((size_t)(c.K / 2) * (c.N / 4));
    f.read(reinterpret_cast<char*>(w.data()), w.size() * 4);
    c.sw.resize((size_t)G * c.N);
    f.read(reinterpret_cast<char*>(c.sw.data()), c.sw.size() * 4);
    c.zp.assign((size_t)G * c.N, 0);
    if (h[4]) f.read(reinterpret_cast<char*>(c.zp.data()), c.zp.size());
    if (!f) { printf("Error: short read from '%s'\n", path.c_str()); return false; }
    c.codes.resize((size_t)c.K * c.N);
    for (uint32_t k = 0; k < c.K; ++k)
        for (uint32_t n = 0; n < c.N; ++n)
            c.codes[(size_t)k * c.N + n] = (w[(size_t)(k / 2) * (c.N / 4) + n / 4] >> ((k % 2) * 16 + (n % 4) * 4)) & 0xF;
    printf("Loaded case %s: M=%u N=%u K=%u gs=%u zp=%s\n", path.c_str(), c.M, c.N, c.K, c.gs, h[4] ? "per-(g,n)" : "none");
    xZP = h[4] != 0;
    return true;
}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    gemm_case_t gc;
    const bool case_mode = !xCaseFile.empty();
    if (case_mode) {
        if (!load_case(xCaseFile, gc)) return -1;
        xM = gc.M; xN = gc.N; xK = gc.K; xGS = gc.gs;
    }
    uint32_t M = xM, N = xN, K = xK, gs = xGS, G = K / gs;
    if (M % cfg::tileM || N % cfg::tileN || K % cfg::tileK || K % gs || gs % cfg::tileK) {
        printf("Error: M%%%u N%%%u K%%%u gs%%%u must be 0\n", cfg::tileM, cfg::tileN, cfg::tileK, cfg::tileK);
        return -1;
    }
#ifdef PG_INT8
    if (PG_SRC_FP8 || PG_SRC_FP4) { printf("Error: FP sources convert to FP16 only\n"); return -1; }
#endif
    printf("purism_gemm: src=%d dst=%s M=%u N=%u K=%u gs=%u zp=%d (LG_PACK=%d NP=%u)\n", PG_SRC,
#ifdef PG_INT8
           "INT8",
#else
           "FP16",
#endif
           M, N, K, gs, (int)xZP, PG_LG_PACK, NP);

    std::srand(7);
    std::vector<uint32_t> codes((size_t)K * N);
    for (auto& v : codes) v = rand_code();
    std::vector<int8_t> zp((size_t)G * N, 0);
    if (xZP && !PG_SRC_FP8 && !PG_SRC_FP4) {
        for (auto& z : zp) {
            if (PG_SRC == 11)      z = (int8_t)(rand() % 9 - 4);
            else if (PG_SRC == 12) z = (int8_t)(rand() % 16);
            else if (PG_SRC == 9)  z = (int8_t)(rand() % 61 - 30);
            else                   z = (int8_t)(uint8_t)(rand() % 256);
        }
    }
    std::vector<float> sw((size_t)G * N), sa(M, 1.0f), ref((size_t)M * N, 0.f), out((size_t)M * N, 0.f);
    for (auto& v : sw) v = 1.f / 64.f + (float)(rand() % 100) / 1e4f;
#ifdef PG_INT8
    std::vector<int8_t> A((size_t)M * K);
    for (auto& v : A) v = (int8_t)(rand() % 17 - 8);
    for (auto& v : sa) v = 1.f / 127.f + (float)(rand() % 100) / 1e5f;
#else
    std::vector<uint16_t> A((size_t)M * K);
    for (auto& v : A) v = f32_to_f16((float)(rand() % 400 - 200) / 100.0f);
#endif
    if (case_mode) {
        codes = std::move(gc.codes); zp = std::move(gc.zp); sw = std::move(gc.sw);
        memcpy(A.data(), gc.A.data(), gc.A.size());
#ifdef PG_INT8
        sa = std::move(gc.sa);
#endif
    }

    // dequantized B as the TCU sees it
    auto deq = [&](uint32_t k, uint32_t n) -> double {
        uint32_t code = codes[(size_t)k * N + n];
        int8_t z = zp[(k / gs) * N + n];
        if (PG_SRC_FP8) return fp8_to_f32((uint8_t)code, PG_SRC == 5);
        if (PG_SRC_FP4) return fp4_to_f32(code);
        int32_t v = int_deq(code, z);
#ifdef PG_INT8
        v = std::max(-128, std::min(127, v));     // OCU INT2INT saturation
#endif
        return (double)v;
    };

    // golden: group partial in the TCU accumulator type, then PSSU FP32 scaled accumulation
    double max_ref = 0;
    for (uint32_t m = 0; m < M && !case_mode; ++m) for (uint32_t n = 0; n < N; ++n) {
        float y = 0.f;
        for (uint32_t g = 0; g < G; ++g) {
#ifdef PG_INT8
            int64_t p = 0;
            for (uint32_t k = g * gs; k < (g + 1) * gs; ++k) p += (int64_t)A[(size_t)m * K + k] * (int64_t)deq(k, n);
            float s = sw[g * N + n] * sa[m];
            y = y + (float)(int32_t)p * s;
#else
            double p = 0;
            for (uint32_t k = g * gs; k < (g + 1) * gs; ++k) p += (double)f16_to_f32(A[(size_t)m * K + k]) * deq(k, n);
            y = y + (float)p * sw[g * N + n];
#endif
        }
        ref[(size_t)m * N + n] = y;
        max_ref = std::max(max_ref, (double)std::fabs(y));
    }

    // pack B in TCU fragB order: lane register r holds i_ratio consecutive K elements of
    // one column (direct-fragB map); packed word w, slice s holds register r = (w << LG_PACK) | s,
    // its elements at source-element positions s*i_ratio + j
    const uint32_t NTN = N / cfg::tileN, NTK = K / cfg::tileK, NT = NUM_THREADS;
    constexpr uint32_t n_stride = cfg::b_sub_blocks * cfg::tcN;
    constexpr uint32_t k_stride = cfg::tcK * cfg::i_ratio;
    std::vector<uint32_t> packed((size_t)NTK * NTN * NT * NP, 0u);
    std::vector<uint8_t> used((size_t)K * N, 0);
    for (uint32_t kt = 0; kt < NTK; ++kt)
    for (uint32_t nt = 0; nt < NTN; ++nt)
    for (uint32_t lane = 0; lane < NT; ++lane) {
        uint32_t bidx = (cfg::b_block_size == NT) ? 0u : (lane / cfg::b_block_size);
        uint32_t lib  = (cfg::b_block_size == NT) ? lane : (lane % cfg::b_block_size);
        uint32_t b_block_col = (lib / cfg::tcK) + (bidx * cfg::tcN);
        uint32_t b_block_row = (lib % cfg::tcK) * cfg::i_ratio;
        for (uint32_t r = 0; r < cfg::NRB; ++r) {
            uint32_t k0 = kt * cfg::tileK + b_block_row + (r / cfg::b_sub_steps) * k_stride;
            uint32_t n  = nt * cfg::tileN + b_block_col + (r % cfg::b_sub_steps) * n_stride;
            uint32_t w = r >> PG_LG_PACK, s = r & ((1u << PG_LG_PACK) - 1);
            for (uint32_t j = 0; j < cfg::i_ratio; ++j) {
                uint32_t pos = s * cfg::i_ratio + j;
                size_t idx = ((size_t)(kt * NTN + nt) * NT + lane) * NP + w;
                packed[idx] |= codes[(size_t)(k0 + j) * N + n] << (pos * SRC_BW);
                if (used[(size_t)(k0 + j) * N + n]++) { printf("pack: duplicate coverage\n"); return -1; }
            }
        }
    }
    for (auto u : used) if (u != 1) { printf("pack: coverage hole\n"); return -1; }

    RT_CHECK(vx_dev_open(&device));
    kernel_arg_t karg = {};
    RT_CHECK(vx_mem_alloc(device, A.size() * sizeof(A[0]), VX_MEM_READ, &A_buf));
    RT_CHECK(vx_mem_alloc(device, packed.size() * 4, VX_MEM_READ, &B_buf));
    RT_CHECK(vx_mem_alloc(device, out.size() * 4, VX_MEM_WRITE, &C_buf));
    RT_CHECK(vx_mem_alloc(device, sw.size() * 4, VX_MEM_READ, &sw_buf));
    RT_CHECK(vx_mem_alloc(device, sa.size() * 4, VX_MEM_READ, &sa_buf));
    RT_CHECK(vx_mem_address(A_buf, &karg.A_addr));
    RT_CHECK(vx_mem_address(B_buf, &karg.B_addr));
    RT_CHECK(vx_mem_address(C_buf, &karg.C_addr));
    RT_CHECK(vx_mem_address(sw_buf, &karg.sw_addr));
    RT_CHECK(vx_mem_address(sa_buf, &karg.sa_addr));
    RT_CHECK(vx_copy_to_dev(A_buf, A.data(), 0, A.size() * sizeof(A[0])));
    RT_CHECK(vx_copy_to_dev(B_buf, packed.data(), 0, packed.size() * 4));
    RT_CHECK(vx_copy_to_dev(sw_buf, sw.data(), 0, sw.size() * 4));
    RT_CHECK(vx_copy_to_dev(sa_buf, sa.data(), 0, sa.size() * 4));
    if (xZP) {
        RT_CHECK(vx_mem_alloc(device, zp.size(), VX_MEM_READ, &zp_buf));
        RT_CHECK(vx_mem_address(zp_buf, &karg.zp_addr));
        RT_CHECK(vx_copy_to_dev(zp_buf, zp.data(), 0, zp.size()));
    }
    karg.M = M; karg.N = N; karg.K = K; karg.group_size = gs;
    karg.grid_dim[0] = N / cfg::tileN; karg.grid_dim[1] = M / cfg::tileM;
    karg.block_dim[0] = NUM_THREADS; karg.block_dim[1] = 1;

    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buf));
    RT_CHECK(vx_upload_bytes(device, &karg, sizeof(karg), &args_buf));
    RT_CHECK(vx_start(device, krnl_buf, args_buf));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    RT_CHECK(vx_copy_from_dev(out.data(), C_buf, 0, out.size() * 4));

    if (!xOutFile.empty()) {
        std::ofstream of(xOutFile, std::ios::binary);
        of.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)(out.size() * 4));
        printf("Wrote C [%u x %u] to %s\n", M, N, xOutFile.c_str());
    }
    if (case_mode) {   // output (-o) is checked offline
        cleanup();
        printf("PASSED (case mode: host check skipped)\n");
        return 0;
    }

    // error normalized by the output range (FP16 path: FEDP FP32 vs double reference order)
    double max_err = 0; uint32_t exact = 0, bad = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        double e = std::fabs((double)out[i] - (double)ref[i]) / (max_ref + 1e-30);
        if (std::isnan(out[i])) e = 1;
        if (out[i] == ref[i]) ++exact;
        if (e > 1e-5) { if (bad < 5) printf("  C[%zu][%zu] got=%g ref=%g\n", i / N, i % N, out[i], ref[i]); ++bad; }
        max_err = std::max(max_err, e);
    }
    printf("ACCURACY: max_err/range=%.3e bit_exact=%u/%zu\n", max_err, exact, out.size());
    cleanup();
    if (bad) { printf("FAILED (%u)\n", bad); return 1; }
    printf("PASSED\n");
    return 0;
}

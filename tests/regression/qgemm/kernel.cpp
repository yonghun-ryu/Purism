#include "common.h"
#include <cstring>
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

using ctx = vt::wmma_context<NUM_THREADS, vt::fp16, vt::fp32>;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;

// Purism intrinsics (hw/rtl/tcu: OCU operand transformer + PSSU on the TCU output path).
//   WMMA        .insn r CUSTOM0, 0,      2, x<fmt_d>, x<fmt_s>, x<ot_src>
//   WMMA.DRAIN  .insn r CUSTOM0, 1..4,   2, ...  (1/2 = one-scale bank0/1, 3/4 = two-scale bank0/1)
//   SETSCALE    .insn r CUSTOM0, 0,      3, x<{bank,wa,slot}>, f<scale>, x0
//   READ        .insn r CUSTOM0, 1,      3, f<rd>, x<r>, x0
//   SETZP       .insn r CUSTOM0, 2,      3, x0, x<zp word>, x0
// fmt_s is the native TCU type (FP16); ot_src = TCU_I4_ID makes the OT expand the
// packed INT4 B register (2 packed words per lane = 8 fragB registers).
#define TCU_I4_ID_K 11
template <int F3>
static inline __attribute__((always_inline))
void mma_ot_i4(ctx::fragment_acc &fragC, const ctx::fragment_a &fragA, uint32_t w0, uint32_t w1) {
    float b0, b1; memcpy(&b0, &w0, 4); memcpy(&b1, &w1, 4);
    register float fa0 __asm__("f0")  = fragA.data[0];
    register float fa1 __asm__("f1")  = fragA.data[1];
    register float fa2 __asm__("f2")  = fragA.data[2];
    register float fa3 __asm__("f3")  = fragA.data[3];
    register float fa4 __asm__("f4")  = fragA.data[4];
    register float fa5 __asm__("f5")  = fragA.data[5];
    register float fa6 __asm__("f6")  = fragA.data[6];
    register float fa7 __asm__("f7")  = fragA.data[7];
    register float fb0 __asm__("f10") = b0;
    register float fb1 __asm__("f11") = b1;
    register float fc0 __asm__("f24") = fragC.data[0];
    register float fc1 __asm__("f25") = fragC.data[1];
    register float fc2 __asm__("f26") = fragC.data[2];
    register float fc3 __asm__("f27") = fragC.data[3];
    register float fc4 __asm__("f28") = fragC.data[4];
    register float fc5 __asm__("f29") = fragC.data[5];
    register float fc6 __asm__("f30") = fragC.data[6];
    register float fc7 __asm__("f31") = fragC.data[7];
    register float fd0 __asm__("f24");
    register float fd1 __asm__("f25");
    register float fd2 __asm__("f26");
    register float fd3 __asm__("f27");
    register float fd4 __asm__("f28");
    register float fd5 __asm__("f29");
    register float fd6 __asm__("f30");
    register float fd7 __asm__("f31");
    __asm__ volatile (".insn r %[insn], %[f3], 2, x%[fmd], x%[fms], x%[src]"
      : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [fmd]"i"(vt::fp32::id), [fms]"i"(vt::fp16::id), [src]"i"(TCU_I4_ID_K),
        "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
        "f"(fb0), "f"(fb1),
        "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
    );
    fragC.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
}

template <int CTRL>   // CTRL = {bank, wa, slot[2:0]}
static inline __attribute__((always_inline)) void pssu_setscale(float s) {
    register float fs __asm__("f8") = s;
    __asm__ volatile (".insn r %[insn], 0, 3, x%[ctrl], x8, x0"
        :: [insn]"i"(RISCV_CUSTOM0), [ctrl]"i"(CTRL), "f"(fs) : "memory");
}

template <int R>
static inline __attribute__((always_inline)) float pssu_read() {
    register float fr __asm__("f8");
    __asm__ volatile (".insn r %[insn], 1, 3, x8, x%[r], x0"
        : "=f"(fr) : [insn]"i"(RISCV_CUSTOM0), [r]"i"(R) : "memory");
    return fr;
}

static inline __attribute__((always_inline)) void ot_setzp(uint64_t zw) {
    __asm__ volatile (".insn r %[insn], 2, 3, x0, %[z], x0"
        :: [insn]"i"(RISCV_CUSTOM0), [z]"r"(zw) : "memory");
}

// one-scale weight scales of bank B for the n steps (slot = n)
template <int B>
static inline __attribute__((always_inline)) void pssu_setscales_w(const float* s) {
    static_assert(cfg::n_steps <= 8, "slot field is 3 bits");
    pssu_setscale<(B << 4) | 0>(s[0]);
    if constexpr (cfg::n_steps > 1) pssu_setscale<(B << 4) | 1>(s[1]);
    if constexpr (cfg::n_steps > 2) pssu_setscale<(B << 4) | 2>(s[2]);
    if constexpr (cfg::n_steps > 3) pssu_setscale<(B << 4) | 3>(s[3]);
    if constexpr (cfg::n_steps > 4) pssu_setscale<(B << 4) | 4>(s[4]);
    if constexpr (cfg::n_steps > 5) pssu_setscale<(B << 4) | 5>(s[5]);
    if constexpr (cfg::n_steps > 6) pssu_setscale<(B << 4) | 6>(s[6]);
    if constexpr (cfg::n_steps > 7) pssu_setscale<(B << 4) | 7>(s[7]);
}

template <int R>
static inline __attribute__((always_inline)) void pssu_read_all(float* out) {
    if constexpr (R < cfg::NRC) {
        out[R] = pssu_read<R>();
        pssu_read_all<R + 1>(out);
    }
}

// Per-thread accumulator layout: thread t, register r. Must match the store
// mapping below (block_col = lane % tcN, block_row = lane / tcN) — (t & 1) was a
// tcN=2 (NT4) shortcut and mis-fetches the weight scale for tcN>2 (e.g. NT16
// tcN=4: cols 2,3 got col 0,1's scale). Use t % tcN to track any tile geometry.
// row_in_tile = (t / cfg::tcN) + (r / cfg::n_steps) * cfg::tcM
// col_in_tile = (t % cfg::tcN) + (r % cfg::n_steps) * cfg::tcN
static inline uint32_t col_in_tile(uint32_t t, uint32_t r) {
    return (t % cfg::tcN) + (r % cfg::n_steps) * cfg::tcN;
}

// Purism W4A16: packed INT4 B expanded by the OCU in the TCU operand path; group
// scales applied by the PSSU on the TCU output path into its accumulator. Per
// group: SETZP (per-B-register zero points) + SETSCALE (bank g&1), (G_k-1) WMMA +
// 1 WMMA.DRAIN. Per tile: one fill_fragment, NRC x PSSU.READ, store.
void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto pA      = reinterpret_cast<const uint16_t*>(arg->A_addr);
    auto pB_i4   = reinterpret_cast<const uint32_t*>(arg->B_int4_addr);
    auto pC      = reinterpret_cast<float*>(arg->C_addr);
    auto pScaleW = reinterpret_cast<const float*>(arg->scale_w_addr);
    auto pZP     = reinterpret_cast<const int8_t*>(arg->zp_addr);
    uint32_t M = arg->M, N = arg->N, K = arg->K, gs = arg->group_size, G = K / gs;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;
    uint32_t lane     = vx_thread_id();
    const uint32_t NTN = N / cfg::tileN;
    const uint32_t nt  = tile_col / cfg::tileN;

    // direct-fragB column of each (unpacked) B register for this lane
    constexpr uint32_t Q_NT = NUM_THREADS;
    uint32_t bidx = (cfg::b_block_size == Q_NT) ? 0u : (lane / cfg::b_block_size);
    uint32_t lib  = (cfg::b_block_size == Q_NT) ? lane : (lane % cfg::b_block_size);
    uint32_t b_block_col = (lib / cfg::tcK) + (bidx * cfg::tcN);
    constexpr uint32_t n_stride = cfg::b_sub_blocks * cfg::tcN;

    ctx::fragment_a   fragA;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, 0.0f);

    for (uint32_t g = 0; g < G; ++g) {
        uint32_t k_base = g * gs;
        uint32_t bank = g & 1;

        if (pZP) {
            uint64_t zw = 0;
            for (uint32_t r = 0; r < cfg::NRB; ++r) {
                uint32_t n = tile_col + b_block_col + (r % cfg::b_sub_steps) * n_stride;
                zw |= (uint64_t)(uint8_t)pZP[g * N + n] << (8 * r);
            }
            ot_setzp(zw);
        }
        float sw[cfg::n_steps];
        for (uint32_t n = 0; n < cfg::n_steps; ++n) {
            uint32_t n_abs = tile_col + col_in_tile(lane, n);
            sw[n] = (n_abs < N) ? pScaleW[g * N + n_abs] : 0.0f;
        }
        if (bank) pssu_setscales_w<1>(sw); else pssu_setscales_w<0>(sw);

        for (uint32_t k = 0; k < gs; k += ctx::tileK) {
            uint32_t k_abs = k_base + k;
            uint32_t kt = k_abs / cfg::tileK;
            const uint32_t* rep = pB_i4 + (((uint64_t)(kt * NTN + nt) * NUM_THREADS + lane) * 2);
            uint32_t w0 = rep[0], w1 = rep[1];
            ctx::load_matrix_sync(fragA, pA + (uint64_t)tile_row * K + k_abs, K);
            if (k + ctx::tileK < gs) {
                mma_ot_i4<0>(fragC, fragA, w0, w1);
            } else if (bank) {
                mma_ot_i4<2>(fragC, fragA, w0, w1);   // drain, one-scale, bank 1
            } else {
                mma_ot_i4<1>(fragC, fragA, w0, w1);   // drain, one-scale, bank 0
            }
        }
    }

    float y[cfg::NRC];
    pssu_read_all<0>(y);

    uint32_t block_row = lane / cfg::tcN;
    uint32_t block_col = lane % cfg::tcN;
    for (uint32_t r = 0; r < cfg::NRC; ++r) {
        uint32_t bm = r / cfg::n_steps;
        uint32_t bn = r % cfg::n_steps;
        uint32_t row = tile_row + block_row + bm * cfg::tcM;
        uint32_t col = tile_col + block_col + bn * cfg::tcN;
        if (row < M && col < N)
            pC[row * N + col] = y[r];
    }
}

int main() {
    auto arg = reinterpret_cast<kernel_arg_t*>(csr_read(VX_CSR_MSCRATCH));
    return vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                            reinterpret_cast<vx_kernel_func_cb>(kernel_body), arg);
}

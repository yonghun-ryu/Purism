#include "common.h"
#include <cstring>
#include <vx_spawn.h>
#include <vx_tensor.h>

// Purism format-matrix GEMM kernel: the OCU (TCU operand transformer) expands
// the packed B source format, the PSSU on the TCU output path applies the group
// scales into its accumulator (see hw/rtl/tcu/VX_tcu_operand_transformer.sv,
// VX_tcu_pssu.sv). Encodings:
//   WMMA        .insn r CUSTOM0, 0,    2, x<fmt_d>, x<fmt_s>, x<ot_src>
//   WMMA.DRAIN  .insn r CUSTOM0, 1..4, 2, ...  (1/2 one-scale bank0/1, 3/4 two-scale bank0/1)
//   SETSCALE    .insn r CUSTOM0, 0,    3, x<{bank,wa,slot}>, f<scale>, x0
//   READ        .insn r CUSTOM0, 1,    3, f<rd>, x<r>, x0
//   SETZP       .insn r CUSTOM0, 2,    3, x0, x<zp word>, x0

namespace vt = vortex::tensor;
#ifdef PG_INT8
using It = vt::int8;  using Ot = vt::int32;
#else
using It = vt::fp16;  using Ot = vt::fp32;
#endif
using ctx = vt::wmma_context<NUM_THREADS, It, Ot>;
using cfg = vt::wmma_config_t<NUM_THREADS, It, Ot>;

static_assert(cfg::NRA == 8 && cfg::NRB == 8 && cfg::NRC == 8, "register map below assumes 8/8/8 (f0-7, f10-17, f24-31)");
static constexpr uint32_t NP = cfg::NRB >> PG_LG_PACK;   // packed B words per lane per k-tile

template <int F3>
static inline __attribute__((always_inline))
void mma_ot(ctx::fragment_acc &fragC, const ctx::fragment_a &fragA, const uint32_t* w) {
    float b[8] = {};
    for (uint32_t i = 0; i < NP; ++i) memcpy(&b[i], &w[i], 4);
    register float fa0 __asm__("f0")  = fragA.data[0];
    register float fa1 __asm__("f1")  = fragA.data[1];
    register float fa2 __asm__("f2")  = fragA.data[2];
    register float fa3 __asm__("f3")  = fragA.data[3];
    register float fa4 __asm__("f4")  = fragA.data[4];
    register float fa5 __asm__("f5")  = fragA.data[5];
    register float fa6 __asm__("f6")  = fragA.data[6];
    register float fa7 __asm__("f7")  = fragA.data[7];
    register float fb0 __asm__("f10") = b[0];
    register float fb1 __asm__("f11") = b[1];
    register float fb2 __asm__("f12") = b[2];
    register float fb3 __asm__("f13") = b[3];
    register float fb4 __asm__("f14") = b[4];
    register float fb5 __asm__("f15") = b[5];
    register float fb6 __asm__("f16") = b[6];
    register float fb7 __asm__("f17") = b[7];
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
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [fmd]"i"(Ot::id), [fms]"i"(It::id), [src]"i"(PG_SRC),
        "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
        "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7),
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

template <int B, int WA, int S, int NS>
static inline __attribute__((always_inline)) void setscales(const float* s) {
    if constexpr (S < NS) {
        pssu_setscale<(B << 4) | (WA << 3) | S>(s[S]);
        setscales<B, WA, S + 1, NS>(s);
    }
}

template <int R>
static inline __attribute__((always_inline)) void read_all(float* out) {
    if constexpr (R < cfg::NRC) {
        out[R] = pssu_read<R>();
        read_all<R + 1>(out);
    }
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto pA  = reinterpret_cast<const It::dtype*>(arg->A_addr);
    auto pB  = reinterpret_cast<const uint32_t*>(arg->B_addr);
    auto pC  = reinterpret_cast<float*>(arg->C_addr);
    auto pSW = reinterpret_cast<const float*>(arg->sw_addr);
    auto pZP = reinterpret_cast<const int8_t*>(arg->zp_addr);
    uint32_t M = arg->M, N = arg->N, K = arg->K, gs = arg->group_size, G = K / gs;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;
    uint32_t lane     = vx_thread_id();
    const uint32_t NTN = N / cfg::tileN;
    const uint32_t nt  = tile_col / cfg::tileN;

    uint32_t block_row = lane / cfg::tcN;
    uint32_t block_col = lane % cfg::tcN;

    // direct-fragB column of each unpacked B register for this lane (zero-point slot)
    uint32_t bidx = (cfg::b_block_size == NUM_THREADS) ? 0u : (lane / cfg::b_block_size);
    uint32_t lib  = (cfg::b_block_size == NUM_THREADS) ? lane : (lane % cfg::b_block_size);
    uint32_t b_block_col = (lib / cfg::tcK) + (bidx * cfg::tcN);
    constexpr uint32_t n_stride = cfg::b_sub_blocks * cfg::tcN;

#ifdef PG_INT8
    // activation scale per m step (two-scale), constant over groups: load both banks once
    auto pSA = reinterpret_cast<const float*>(arg->sa_addr);
    float sa[cfg::m_steps];
    for (uint32_t m = 0; m < cfg::m_steps; ++m) {
        uint32_t row = tile_row + block_row + m * cfg::tcM;
        sa[m] = (row < M) ? pSA[row] : 0.0f;
    }
    setscales<0, 1, 0, cfg::m_steps>(sa);
    setscales<1, 1, 0, cfg::m_steps>(sa);
#endif

    ctx::fragment_a   fragA;
    ctx::fragment_acc fragC;
    ctx::fill_fragment(fragC, 0);

    for (uint32_t g = 0; g < G; ++g) {
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
            uint32_t col = tile_col + block_col + n * cfg::tcN;
            sw[n] = (col < N) ? pSW[g * N + col] : 0.0f;
        }
        if (bank) setscales<1, 0, 0, cfg::n_steps>(sw); else setscales<0, 0, 0, cfg::n_steps>(sw);

        for (uint32_t k = 0; k < gs; k += ctx::tileK) {
            uint32_t k_abs = g * gs + k;
            uint32_t kt = k_abs / cfg::tileK;
            const uint32_t* w = pB + ((uint64_t)(kt * NTN + nt) * NUM_THREADS + lane) * NP;
            uint32_t wl[NP];
            for (uint32_t i = 0; i < NP; ++i) wl[i] = w[i];
            ctx::load_matrix_sync(fragA, pA + (uint64_t)tile_row * K + k_abs, K);
#ifdef PG_INT8
            constexpr int D0 = 3, D1 = 4;   // two-scale drains
#else
            constexpr int D0 = 1, D1 = 2;   // one-scale drains
#endif
            if (k + ctx::tileK < gs) {
                mma_ot<0>(fragC, fragA, wl);
            } else if (bank) {
                mma_ot<D1>(fragC, fragA, wl);
            } else {
                mma_ot<D0>(fragC, fragA, wl);
            }
        }
    }

    float y[cfg::NRC];
    read_all<0>(y);

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

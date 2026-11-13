#pragma once
#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

// Purism format-matrix GEMM: C[M][N] = sum_g ( A x deq(B) )_g * Sw[g][n] (* Sa[m] for INT8)
//   PG_SRC : B source format = TCU fmt id (11 I4, 12 U4, 9 I8, 10 U8, 5 FP8 E4M3, 6 FP8 E5M2, 7 FP4 E2M1)
//   PG_INT8: native TCU type INT8 (A int8, INT32 partial sums, two-scale) instead of FP16
// B is uploaded pre-packed in TCU fragB order: packed[((kt*NTN + nt)*NT + lane)*NP + w]
// (NP = NRB >> LG_PACK words per lane per k-tile), see main.cpp pack_b().
#ifndef PG_SRC
#define PG_SRC 11
#endif

#define PG_SRC_FP4  ((PG_SRC) == 7)
#define PG_SRC_4BIT ((PG_SRC) == 11 || (PG_SRC) == 12 || PG_SRC_FP4)
#define PG_SRC_FP8  ((PG_SRC) == 5 || (PG_SRC) == 6)

#ifdef PG_INT8
#define PG_LG_PACK (PG_SRC_4BIT ? 1 : 0)
#else
#define PG_LG_PACK (PG_SRC_4BIT ? 2 : 1)
#endif

typedef struct {
    uint32_t grid_dim[2];
    uint32_t block_dim[2];
    uint32_t M, N, K;
    uint32_t group_size;
    uint64_t A_addr;        // fp16 or int8 [M][K]
    uint64_t B_addr;        // packed source words (fragB order)
    uint64_t C_addr;        // float [M][N]
    uint64_t sw_addr;       // float [G][N]
    uint64_t sa_addr;       // float [M] (INT8 two-scale)
    uint64_t zp_addr;       // int8 [G][N] (0 = no zero points)
} kernel_arg_t;

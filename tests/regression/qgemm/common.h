#pragma once
#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

// Default GEMM dimensions — must be multiples of TCU tile (tileM=8, tileN=4, tileK=16)
#ifndef QGEMM_M
#define QGEMM_M 32
#endif
#ifndef QGEMM_N
#define QGEMM_N 32
#endif
#ifndef QGEMM_K
#define QGEMM_K 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 16   // quantization group size along K (K must be divisible)
#endif

// C[M×N] = Σ_g ( A[M×K_g] × (B_int4[K_g×N] - zp[g][n]) ) × scale_w[g][n]
// A:       FP16 activations [M][K] row-major
// B_int4:  packed INT4S weights — uint32_t[K/2][N/4], 8 nibbles per word:
//          nibbles 0..3 = row k, cols n..n+3; nibbles 4..7 = row k+1, cols n..n+3.
//          Requires: K multiple of 2, N multiple of 4. The host repacks it into fragB
//          order before upload (main.cpp).
// C:       FP32 output [M][N] row-major
// scale_w: FP32 per-group per-column weight scale [K/GROUP_SIZE][N]
// zp:      optional INT8 per-group per-column zero points [K/GROUP_SIZE][N]
typedef struct {
    uint32_t grid_dim[2];    // {N/tileN, M/tileM}
    uint32_t block_dim[2];   // {NUM_THREADS, 1}
    uint32_t M, N, K;
    uint32_t group_size;
    uint64_t A_addr;         // uint16_t  [M][K]  (FP16)
    uint64_t B_int4_addr;    // uint32_t  [K/2][N/4]  (packed INT4S, 2-row × 4-col per word)
    uint64_t C_addr;         // float     [M][N]
    uint64_t scale_w_addr;   // float     [G][N]  G = K/group_size
    uint64_t zp_addr;        // int8_t    [G][N]  per-group per-column zero points (0 = none)
} kernel_arg_t;

`timescale 1ns/1ps
// VX_ocu_pkg.sv
// Types, parameters and structs shared by the OCU modules
// ---------------------------------------------------------------------------
// Format Descriptor (16-bit) bit assignment:
//   [15]    src_type   : 0=INT, 1=FP
//   [14:12] src_bw_enc : 0→1b, 1→2b, 2→3b, 3→4b, 4→5b, 5→6b, 6→7b, 7→8b
//   [11]    src_signed : (INT only) 0=unsigned, 1=signed
//   [10:8]  src_es     : (FP only) exponent field width (1~5)
//   [7]     dst_type   : 0=INT, 1=FP
//   [6:3]   dst_bw_enc : 0→1b, 1→2b, ..., 15→16b  (4-bit, supports FP16)
//   [2:0]   dst_es     : (FP only) exponent field width (1~5)
// ---------------------------------------------------------------------------

`ifndef VX_OCU_PKG_SV
`define VX_OCU_PKG_SV

package VX_ocu_pkg;

    // -----------------------------------------------------------------------
    // Format type encoding
    // -----------------------------------------------------------------------
    typedef enum logic {
        FMT_INT = 1'b0,
        FMT_FP  = 1'b1
    } fmt_type_e;

    // -----------------------------------------------------------------------
    // Format Descriptor raw (16-bit, from instruction / CSR)
    // -----------------------------------------------------------------------
    typedef logic [15:0] fmt_desc_t;

    // -----------------------------------------------------------------------
    // Decoded format descriptor (ctrl module output, used internally)
    // -----------------------------------------------------------------------
    typedef struct packed {
        // source format
        fmt_type_e  src_type;       // INT / FP
        logic [3:0] src_bw;         // actual bit width (1~8)
        logic       src_signed;     // INT: signedness. FP: always 1
        logic [2:0] src_es;         // FP: exponent field width
        logic [3:0] src_ms;         // FP: mantissa field width (= src_bw - src_es - 1)
        logic [4:0] src_bias;       // FP: exponent bias

        // destination format
        fmt_type_e  dst_type;       // INT / FP
        logic [4:0] dst_bw;         // actual bit width (1~16)
        logic [2:0] dst_es;         // FP: exponent field width
        logic [3:0] dst_ms;         // FP: mantissa field width
        logic [4:0] dst_bias;       // FP: exponent bias

        // derived: exponent bias delta (dst_bias - src_bias, FP->FP path)
        // signed 6-bit: covers -31~+31
        logic signed [5:0] bias_delta;
    } fmt_ctrl_t;

    // -----------------------------------------------------------------------
    // Bias lookup: exponent width → standard IEEE bias (2^(es-1) - 1)
    // -----------------------------------------------------------------------
    function automatic logic [4:0] get_bias(input logic [2:0] es);
        case (es)
            3'd1: return 5'd0;   // E1: bias=0 (non-standard, special purpose)
            3'd2: return 5'd1;   // FP4 E2M1: bias=1
            3'd3: return 5'd3;   // FP6 E3M2: bias=3
            3'd4: return 5'd7;   // FP8 E4M3: bias=7
            3'd5: return 5'd15;  // FP16 E5M10: bias=15
            default: return 5'd15;
        endcase
    endfunction

    // -----------------------------------------------------------------------
    // Bitwidth encoding → actual bitwidth
    //   src: 3-bit enc → 1~8
    //   dst: 4-bit enc → 1~16  (supports FP16)
    // -----------------------------------------------------------------------
    function automatic logic [3:0] decode_bw(input logic [2:0] bw_enc);
        return {1'b0, bw_enc} + 4'd1;  // 0→1, 1→2, ..., 7→8
    endfunction

    function automatic logic [4:0] decode_dst_bw(input logic [3:0] bw_enc);
        return {1'b0, bw_enc} + 5'd1;  // 0→1, 1→2, ..., 15→16
    endfunction

    // -----------------------------------------------------------------------
    // Elastic interface tag (minimal definition taken from the common Vortex structure)
    // -----------------------------------------------------------------------
    parameter int TAG_WIDTH = 32;   // e.g. PC(30) + warp_id(2)

    typedef logic [TAG_WIDTH-1:0] tag_t;

    // -----------------------------------------------------------------------
    // maximum source data width
    // -----------------------------------------------------------------------
    parameter int SRC_DATA_WIDTH = 8;   // up to INT8 / FP8
    parameter int DST_DATA_WIDTH = 16;  // up to FP16 / INT16

    // -----------------------------------------------------------------------
    // maximum zero-point width (INT8, per-group value)
    // -----------------------------------------------------------------------
    parameter int ZP_WIDTH = 8;

endpackage

`endif // VX_OCU_PKG_SV

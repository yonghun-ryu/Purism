`timescale 1ns/1ps
// VX_ocu_word.sv
// 32-bit packed word wrapper for VX_ocu_core
//
// Position: Operand Collector → [VX_ocu_word] → FEDP (Tensor Core)
//
// Role: slices the 32-bit packed word supplied by the OC (rs1_data[t]) into the smallest
//       element unit (MIN_SRC_BW bits) and feeds N_ELEM VX_ocu_core instances in parallel.
//
//   N_ELEM = XLEN / MIN_SRC_BW = 32 / 4 = 8  (default)
//
//   Element extraction:
//     BIT_START = i × MIN_SRC_BW  (generate-time localparam, constant → wire reorder)
//     elem_data  = SRC_DATA_WIDTH'(packed_data >> BIT_START)
//     elem_valid = (BIT_START & (src_bw-1) == 0)  ← alignment check (runtime, AND-mask)
//
//   Examples:
//     INT4 (src_bw=4): i=0..7 all valid, BIT_START=0,4,8,...,28
//     INT8 (src_bw=8): i=0,2,4,6 valid (BIT_START=0,8,16,24; only even i are 8-bit aligned)
//
// Interface: Vortex elastic handshake (valid/ready + tag), fully combinational
//
// ZP assumption: all elements of a word share one zero_point (group_size ≥ N_ELEM).
//          Sub-word per-group zero-points are not supported.
// ---------------------------------------------------------------------------


module VX_ocu_word
    import VX_ocu_pkg::*;
#(
    parameter int XLEN        = 32,                 // packed word width (Vortex XLEN)
    parameter int MIN_SRC_BW  = 4,                  // minimum element granularity
    parameter int MAX_SRC_BW  = SRC_DATA_WIDTH,     // passed to VX_ocu_core
    parameter int MAX_DST_BW  = DST_DATA_WIDTH,     // passed to VX_ocu_core
    localparam int N_ELEM     = XLEN / MIN_SRC_BW   // 32/4 = 8
)(
    input  logic                                     clk,
    input  logic                                     reset,

    // -----------------------------------------------------------------------
    // Upstream elastic interface (← Operand Collector)
    // -----------------------------------------------------------------------
    input  logic                                     valid_in,
    output logic                                     ready_in,
    input  tag_t                                     tag_in,

    // Packed 32-bit operand word from OC (e.g., rs1_data[t])
    input  logic [XLEN-1:0]                          packed_data,

    // TCU format IDs from instruction op_args.tcu (4-bit each)
    input  logic [3:0]                               fmt_s,   // source format
    input  logic [3:0]                               fmt_d,   // destination format

    // Zero-point (single scalar shared for all elements in this word)
    input  logic [ZP_WIDTH-1:0]                      zero_point,

    // -----------------------------------------------------------------------
    // Downstream elastic interface (→ FEDP)
    // -----------------------------------------------------------------------
    output logic                                     valid_out,
    input  logic                                     ready_out,
    output tag_t                                     tag_out,

    // N_ELEM converted outputs
    output logic [N_ELEM-1:0][DST_DATA_WIDTH-1:0]   dst_data,
    output logic [N_ELEM-1:0]                        elem_valid
);

    // -----------------------------------------------------------------------
    // Elastic handshake: fully combinational pass-through
    // -----------------------------------------------------------------------
    assign valid_out = valid_in;
    assign ready_in  = ready_out;
    assign tag_out   = tag_in;

    // -----------------------------------------------------------------------
    // fmt_s / fmt_d → fmt_desc_t decode
    //
    // fmt_desc bit layout (from VX_ocu_pkg.sv):
    //   [15]    src_type   : 0=INT, 1=FP
    //   [14:12] src_bw_enc : 3-bit, 0→1b … 7→8b
    //   [11]    src_signed : 0=unsigned, 1=signed (FP: always 1)
    //   [10:8]  src_es     : exponent width (FP only)
    //   [7]     dst_type   : 0=INT, 1=FP
    //   [6:3]   dst_bw_enc : 4-bit, 0→1b … 15→16b
    //   [2:0]   dst_es     : exponent width (FP only)
    //
    // fmt_s numeric literals map to VX_tcu_pkg constants (for future migration):
    //   4'd11 = TCU_I4_ID,       4'd12 = TCU_U4_ID
    //   4'd9  = TCU_I8_ID,       4'd10 = TCU_U8_ID
    //   4'd5  = TCU_FP8E4M3_ID,  4'd6  = TCU_FP8E5M2_ID
    //   4'd7  = TCU_FP4E2M1_ID
    //   4'd1  = TCU_FP16_ID
    // -----------------------------------------------------------------------
    function automatic fmt_desc_t tcu_fmt_to_desc(
        input logic [3:0] fmt_s_in,
        input logic [3:0] fmt_d_in
    );
        logic [7:0] src_bits;
        logic [7:0] dst_bits;

        case (fmt_s_in)
            4'd11:   src_bits = 8'b0_011_1_000; // TCU_I4_ID:      INT4 signed
            4'd12:   src_bits = 8'b0_011_0_000; // TCU_U4_ID:      INT4 unsigned
            4'd9:    src_bits = 8'b0_111_1_000; // TCU_I8_ID:      INT8 signed
            4'd10:   src_bits = 8'b0_111_0_000; // TCU_U8_ID:      INT8 unsigned
            4'd5:    src_bits = 8'b1_111_0_100; // TCU_FP8E4M3_ID: FP8 E4M3 (es=4)
            4'd6:    src_bits = 8'b1_111_0_101; // TCU_FP8E5M2_ID: FP8 E5M2 (es=5)
            4'd7:    src_bits = 8'b1_011_0_010; // TCU_FP4E2M1_ID: FP4 E2M1 (es=2)
            default: src_bits = 8'b0_111_1_000; // fallback: INT8 signed
        endcase

        case (fmt_d_in)
            4'd9:    dst_bits = 8'b0_0111_000;  // TCU_I8_ID:  INT8
            4'd1:    dst_bits = 8'b1_1111_101;  // TCU_FP16_ID: FP16 E5M10 (es=5)
            default: dst_bits = 8'b0_0111_000;  // fallback: INT8
        endcase

        return {src_bits, dst_bits};
    endfunction

    // -----------------------------------------------------------------------
    // Single shared VX_ocu_ctrl (decode once for all N_ELEM cores)
    // -----------------------------------------------------------------------
    fmt_desc_t fmt_desc_w;
    assign fmt_desc_w = tcu_fmt_to_desc(fmt_s, fmt_d);

    fmt_ctrl_t ctrl_w;
    VX_ocu_ctrl u_ctrl (
        .fmt_desc  (fmt_desc_w),
        .ctrl_out  (ctrl_w)
    );

    // -----------------------------------------------------------------------
    // N_ELEM parallel VX_ocu_core instances
    //
    // BIT_START = i * MIN_SRC_BW is a generate-time constant (genvar × localparam).
    // packed_data >> BIT_START is a constant shift → pure wire reordering (0 area).
    //
    // elem_valid[i]: runtime alignment check using AND-mask (cheaper than modulo).
    //   Element i is valid iff BIT_START is aligned to the runtime src_bw boundary.
    //   For power-of-2 src_bw: aligned ↔ (BIT_START & (src_bw-1)) == 0
    // -----------------------------------------------------------------------
    for (genvar i = 0; i < N_ELEM; ++i) begin : g_ocu
        localparam int BIT_START = i * MIN_SRC_BW;

        // Constant right shift → wire reorder by BIT_START positions.
        // Upper bits beyond packed_data width implicitly 0; VX_ocu_core masks
        // to ctrl.src_bw so they are always discarded.
        logic [SRC_DATA_WIDTH-1:0] elem_data;
        assign elem_data = SRC_DATA_WIDTH'(packed_data >> BIT_START);

        // Alignment check: BIT_START must be a multiple of src_bw.
        // For INT4 (src_bw=4): BIT_START ∈ {0,4,...,28} → all aligned → elem_valid = 1
        // For INT8 (src_bw=8): BIT_START ∈ {0,8,16,24} only → i=0,2,4,6 valid
        assign elem_valid[i] = ((6'(BIT_START) & (6'(ctrl_w.src_bw) - 6'd1)) == 6'd0);

        VX_ocu_core #(
            .MAX_SRC_BW (MAX_SRC_BW),
            .MAX_DST_BW (MAX_DST_BW)
        ) u_core (
            .src_data   (elem_data),
            .zero_point (zero_point),
            .ctrl       (ctrl_w),
            .dst_data   (dst_data[i])
        );
    end

    // -----------------------------------------------------------------------
    // Assertions (simulation only)
    // -----------------------------------------------------------------------
    // synthesis translate_off
    always_ff @(posedge clk) begin
        if (!reset && valid_in) begin
            assert ($countones(elem_valid) > 0)
                else $warning("[VX_ocu_word] No active elements — fmt_s=%0d src_bw=%0d",
                              fmt_s, ctrl_w.src_bw);
        end
    end
    // synthesis translate_on

endmodule

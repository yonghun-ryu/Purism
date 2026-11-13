`timescale 1ns/1ps
// VX_ocu_ctrl.sv
// Format Descriptor (16-bit) → fmt_ctrl_t decode (fully combinational)
//
// New formats or a changed ISA encoding only require changes in this module.
// ---------------------------------------------------------------------------


module VX_ocu_ctrl
    import VX_ocu_pkg::*;
(
    input  fmt_desc_t  fmt_desc,
    output fmt_ctrl_t  ctrl_out
);

    // extract fmt_desc fields
    logic        src_type_raw;
    logic [2:0]  src_bw_enc;
    logic        src_signed_raw;
    logic [2:0]  src_es_raw;
    logic        dst_type_raw;
    logic [3:0]  dst_bw_enc;
    logic [2:0]  dst_es_raw;

    assign src_type_raw   = fmt_desc[15];
    assign src_bw_enc     = fmt_desc[14:12];
    assign src_signed_raw = fmt_desc[11];
    assign src_es_raw     = fmt_desc[10:8];
    assign dst_type_raw   = fmt_desc[7];
    assign dst_bw_enc     = fmt_desc[6:3];
    assign dst_es_raw     = fmt_desc[2:0];

    always_comb begin
        // src type
        ctrl_out.src_type   = fmt_type_e'(src_type_raw);
        ctrl_out.src_bw     = decode_bw(src_bw_enc);
        ctrl_out.src_signed = (ctrl_out.src_type == FMT_FP) ? 1'b1 : src_signed_raw;
        ctrl_out.src_es     = (ctrl_out.src_type == FMT_FP) ? src_es_raw : 3'd0;
        ctrl_out.src_bias   = (ctrl_out.src_type == FMT_FP) ? get_bias(src_es_raw) : 5'd0;

        // src mantissa width = bw - es - 1 (excluding the sign bit), meaningful for FP only
        ctrl_out.src_ms     = (ctrl_out.src_type == FMT_FP)
                              ? (ctrl_out.src_bw - 4'(ctrl_out.src_es) - 4'd1)
                              : 4'd0;

        // dst type
        ctrl_out.dst_type   = fmt_type_e'(dst_type_raw);
        ctrl_out.dst_bw     = decode_dst_bw(dst_bw_enc);
        ctrl_out.dst_es     = (ctrl_out.dst_type == FMT_FP) ? dst_es_raw : 3'd0;
        ctrl_out.dst_bias   = (ctrl_out.dst_type == FMT_FP) ? get_bias(dst_es_raw) : 5'd0;

        // dst mantissa width: dst_bw(5b) - dst_es(3b) - 1; result fits in 4 bits (max 10 for FP16)
        ctrl_out.dst_ms     = (ctrl_out.dst_type == FMT_FP)
                              ? 4'(ctrl_out.dst_bw - 5'(ctrl_out.dst_es) - 5'd1)
                              : 4'd0;

        // bias_delta (FP→FP path): dst_bias - src_bias, signed 6-bit
        ctrl_out.bias_delta = $signed({1'b0, ctrl_out.dst_bias})
                            - $signed({1'b0, ctrl_out.src_bias});
    end

endmodule

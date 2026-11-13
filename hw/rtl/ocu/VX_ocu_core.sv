`timescale 1ns/1ps
// VX_ocu_core.sv
// Pure combinational data-path: pre-decoded ctrl → dst_data
//
// No elastic interface, no fmt_desc decode. Intended for use in
// VX_ocu_word where one shared VX_ocu_ctrl feeds N_ELEM cores.
// VX_ocu (standalone, with elastic interface) wraps this module.
// ---------------------------------------------------------------------------


module VX_ocu_core
    import VX_ocu_pkg::*;
#(
    parameter int MAX_SRC_BW = 8,
    parameter int MAX_DST_BW = 16
)(
    input  logic [SRC_DATA_WIDTH-1:0] src_data,
    input  logic [ZP_WIDTH-1:0]       zero_point,
    input  fmt_ctrl_t                 ctrl,
    output logic [DST_DATA_WIDTH-1:0] dst_data
);

    logic [DST_DATA_WIDTH-1:0] int2int_result;
    logic [DST_DATA_WIDTH-1:0] int2fp_result;
    logic [DST_DATA_WIDTH-1:0] fp2fp_result;

    VX_ocu_int2int #(
        .MAX_SRC_BW (MAX_SRC_BW),
        .MAX_DST_BW (MAX_DST_BW)
    ) u_int2int (
        .src_data   (src_data),
        .zero_point (zero_point),
        .ctrl       (ctrl),
        .result     (int2int_result)
    );

    VX_ocu_int2fp #(
        .MAX_SRC_BW (MAX_SRC_BW),
        .MAX_DST_BW (MAX_DST_BW)
    ) u_int2fp (
        .src_data   (src_data),
        .zero_point (zero_point),
        .ctrl       (ctrl),
        .result     (int2fp_result)
    );

    VX_ocu_fp2fp #(
        .MAX_SRC_BW (MAX_SRC_BW),
        .MAX_DST_BW (MAX_DST_BW)
    ) u_fp2fp (
        .src_data   (src_data),
        .ctrl       (ctrl),
        .result     (fp2fp_result)
    );

    // 3:1 output MUX
    always_comb begin
        dst_data = '0;
        if (ctrl.src_type == FMT_INT) begin
            dst_data = (ctrl.dst_type == FMT_INT) ? int2int_result : int2fp_result;
        end else begin
            dst_data = fp2fp_result;
        end
    end

endmodule

// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_tcu_operand_transformer — Purism Operand Conversion Unit (OCU), placed
// between the TCU dispatch (operand collector output) and the PEs.
//
// A WMMA uop whose op_args.tcu.ot_src is non-zero carries a packed B operand in
// the source format ot_src (I4/U4/I8/U8/FP8 E4M3/E5M2/FP4 E2M1). fmt_s is the
// destination type (FP16, INT8, or FP8 E4M3 for 4-bit sources; the FEDP has no FP8
// mode, so an FP8 E4M3 fmt_s needs a PE that takes it). One packed 32-bit register holds the
// elements of 2^LG_PACK native B registers (LG_PACK = log2(dst_bw/src_bw)); the
// uop sequencer makes that many consecutive uops read the same packed register
// and tags each with step_w = which slice it consumes. Per lane, VX_ocu_word
// (combinational INT2INT / INT2FP / FP2FP datapath) converts the word with the
// zero point of the unpacked B register (b_idx), and the slice for this uop is
// placed in the native register layout:
//   FP16 dst: {e[2s+1], e[2s]} (K-paired), NaN-boxed to XLEN
//   INT8 dst: {e[4s+3], .., e[4s]} (K-quad)
//   FP8 E4M3 dst: same as INT8, each element a bit select of its FP16 form. 4-bit
//   sources (I4/U4 minus a zero point: |v| <= 15; FP4) have FP16 exponent 0 or 14..18
//   and at most 3 significand bits, so the select is exact.
// where e[] are the source elements in order. Elements of an 8-bit source sit at
// the even VX_ocu_word outputs (0,2,4,6).
//
// OT.SETZP (op_type INST_TCU_SETZP) loads the per-warp, per-lane zero-point
// register: 8 bits per unpacked B register. It is consumed here, so it is ordered
// with the following uops by construction, then passes on to commit.

`include "VX_define.vh"

module VX_tcu_operand_transformer import VX_gpu_pkg::*, VX_tcu_pkg::*; (
    input  wire clk,
    input  wire reset,

    VX_execute_if.slave  execute_if_in,
    VX_execute_if.master execute_if_out
);
    localparam NT      = `NUM_THREADS;
    localparam ZP_REGS = TCU_NRB;           // one 8-bit zero point per unpacked B register
    localparam ZP_W    = ZP_REGS * 8;
    `STATIC_ASSERT (ZP_W <= `XLEN, ("zero-point register must fit in one integer register"));

    wire [3:0] ot_src = execute_if_in.data.op_args.tcu.ot_src;
    wire [3:0] fmt_s  = execute_if_in.data.op_args.tcu.fmt_s;
    wire [1:0] step_w = execute_if_in.data.op_args.tcu.step_w;
    wire [3:0] b_idx  = execute_if_in.data.op_args.tcu.b_idx;
    wire       is_mma = (execute_if_in.data.op_type == INST_TCU_WMMA)
                     || (execute_if_in.data.op_type == INST_TCU_DRAIN);
    wire       is_setzp = (execute_if_in.data.op_type == INST_TCU_SETZP);
    wire       do_conv  = is_mma && (ot_src != 4'd0);
    wire       src_4b   = (ot_src == 4'(TCU_I4_ID)) || (ot_src == 4'(TCU_U4_ID)) || (ot_src == 4'(TCU_FP4E2M1_ID));
    wire       dst_fp16 = (fmt_s == 4'(TCU_FP16_ID));
    wire       dst_e4m3 = (fmt_s == 4'(TCU_FP8E4M3_ID));

    wire in_fire = execute_if_in.valid && execute_if_in.ready;
    wire zp_we   = in_fire && is_setzp;

    // zero points read as 0 until a warp's first OT.SETZP (the RAM itself has no reset)
    reg [`NUM_WARPS-1:0] zp_set;
    always @(posedge clk) begin
        if (reset) begin
            zp_set <= '0;
        end else if (zp_we) begin
            zp_set[execute_if_in.data.wid] <= 1'b1;
        end
    end

    wire [NT-1:0][31:0] conv_rd32;

    for (genvar t = 0; t < NT; ++t) begin : g_lanes
        wire [7:0][15:0] e;   // converted elements (FP16, or INT8 in [7:0])

        // per-warp zero-point word of this lane (distributed RAM, 1 write + 1 async read)
        (* ram_style = "distributed" *) reg [ZP_W-1:0] zp_mem [`NUM_WARPS];
        always @(posedge clk) begin
            if (zp_we) zp_mem[execute_if_in.data.wid] <= execute_if_in.data.rs1_data[t][ZP_W-1:0];
        end
        wire [ZP_W-1:0] zp_word = zp_set[execute_if_in.data.wid] ? zp_mem[execute_if_in.data.wid] : '0;
        wire [7:0] zp = zp_word[b_idx * 8 +: 8];

        // elastic ports bypassed: the OT stage provides the handshake.
        VX_ocu_word ocu_word (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (1'b1),
            .ready_out  (1'b1),
            .tag_in     ('0),
            `UNUSED_PIN (ready_in),
            `UNUSED_PIN (valid_out),
            `UNUSED_PIN (tag_out),
            .packed_data(execute_if_in.data.rs2_data[t][31:0]),
            .fmt_s      (ot_src),
            .fmt_d      (dst_e4m3 ? 4'(TCU_FP16_ID) : fmt_s),
            .zero_point (zp),
            .dst_data   (e),
            `UNUSED_PIN (elem_valid)
        );

        // FP16 {s, e[4:0], m[9:0]} -> FP8 E4M3 {s, e[4], e[2:0], m[9:7]}
        wire [3:0][7:0] e4m3;
        for (genvar i = 0; i < 4; ++i) begin : g_e4m3
            wire [15:0] f = e[(step_w[0] ? 4 : 0) + i];
            assign e4m3[i] = {f[15:14], f[12:7]};
            `UNUSED_VAR ({f[13], f[6:0]})
        end

        logic [31:0] rd32;
        always_comb begin
            case ({src_4b, dst_fp16})
            2'b11: // 4-bit -> FP16: 4 slices of 2 elements
                rd32 = {e[2*step_w+1], e[2*step_w]};
            2'b01: // 8-bit -> FP16: 2 slices of 2 elements (8-bit elements at even outputs)
                rd32 = step_w[0] ? {e[6], e[4]} : {e[2], e[0]};
            2'b10: // 4-bit -> INT8 / FP8 E4M3: 2 slices of 4 elements
                if (dst_e4m3) begin
                    rd32 = e4m3;
                end else begin
                    rd32 = step_w[0] ? {e[7][7:0], e[6][7:0], e[5][7:0], e[4][7:0]}
                                     : {e[3][7:0], e[2][7:0], e[1][7:0], e[0][7:0]};
                end
            default: // 8-bit -> INT8: 1 slice of 4 elements
                rd32 = {e[6][7:0], e[4][7:0], e[2][7:0], e[0][7:0]};
            endcase
        end
        assign conv_rd32[t] = rd32;
    end

    tcu_exe_t ot_data_next;

    always_comb begin
        ot_data_next = execute_if_in.data;
        if (do_conv) begin
            for (integer t = 0; t < NT; t++) begin
                // NaN-box like a 32-bit value in a wider F register (FEDP reads [31:0])
                ot_data_next.rs2_data[t]       = '1;
                ot_data_next.rs2_data[t][31:0] = conv_rd32[t];
            end
        end
    end

    // 2-entry elastic stage (full throughput, no combinational ready path)
    VX_elastic_buffer #(
        .DATAW ($bits(tcu_exe_t)),
        .SIZE  (2)
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (execute_if_in.valid),
        .ready_in  (execute_if_in.ready),
        .data_in   (ot_data_next),
        .data_out  (execute_if_out.data),
        .ready_out (execute_if_out.ready),
        .valid_out (execute_if_out.valid)
    );

endmodule

`timescale 1ns/1ps
// VX_pssu_fmul — pipelined FP32 multiply
//
// Default (no TCU_BHF):
//   VIVADO (synthesis): Xilinx Floating Point IP v7.1, FP32×FP32, C_LATENCY=8.
//   Simulation: pure-SV behavioral IEEE 754 FP32 multiply, LATENCY=8 shift-reg.
//
// TCU_BHF defined: Hardfloat VX_tcu_bhf_fmul, LATENCY=2.

`ifndef TCU_BHF

module VX_pssu_fmul #(
    parameter LATENCY = 8   // must match xil_fmul C_LATENCY=8
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] y
);
    // -----------------------------------------------------------------------
    // IEEE 754 FP32 multiply (combinational, round-to-nearest-even)
    // Handles: normal × normal, zero, ±Inf, NaN
    // Denormal inputs flushed to zero (FTZ).
    // -----------------------------------------------------------------------
    function automatic logic [31:0] fp32_mul_beh(
        input logic [31:0] fa,
        input logic [31:0] fb
    );
        logic        s_a, s_b, s_y;
        logic [7:0]  e_a, e_b;
        logic [22:0] m_a, m_b;
        logic [23:0] sig_a, sig_b;
        logic [47:0] sig_prod;
        logic signed [9:0] e_sum;
        logic [22:0] m_y;
        logic        guard, sticky;
        logic [23:0] m_y24;
        begin
            s_a = fa[31]; e_a = fa[30:23]; m_a = fa[22:0];
            s_b = fb[31]; e_b = fb[30:23]; m_b = fb[22:0];
            s_y = s_a ^ s_b;

            // NaN: any NaN input, or Inf×0
            if ((e_a == 8'hFF && m_a != 0) || (e_b == 8'hFF && m_b != 0) ||
                (e_a == 8'hFF && e_b == 8'h00) || (e_b == 8'hFF && e_a == 8'h00)) begin
                fp32_mul_beh = 32'hFFC00000;
            end else if (e_a == 8'hFF || e_b == 8'hFF) begin
                // Inf × nonzero → ±Inf
                fp32_mul_beh = {s_y, 8'hFF, 23'h0};
            end else if (e_a == 8'h00 || e_b == 8'h00) begin
                // Zero (denormals flushed) → ±0
                fp32_mul_beh = {s_y, 31'h0};
            end else begin
                // Normal × Normal
                sig_a    = {1'b1, m_a};
                sig_b    = {1'b1, m_b};
                sig_prod = 48'(sig_a) * 48'(sig_b);

                if (sig_prod[47]) begin
                    // Product ∈ [2,4): implicit 1 at bit 47
                    e_sum  = $signed({2'b0, e_a}) + $signed({2'b0, e_b}) - 10'sd126;
                    m_y    = sig_prod[46:24];
                    guard  = sig_prod[23];
                    sticky = |sig_prod[22:0];
                end else begin
                    // Product ∈ [1,2): implicit 1 at bit 46
                    e_sum  = $signed({2'b0, e_a}) + $signed({2'b0, e_b}) - 10'sd127;
                    m_y    = sig_prod[45:23];
                    guard  = sig_prod[22];
                    sticky = |sig_prod[21:0];
                end

                // Round-to-nearest-even
                m_y24 = {1'b0, m_y} + {23'h0, (guard && (sticky || m_y[0]))};
                if (m_y24[23]) begin
                    e_sum = e_sum + 10'sd1;
                    m_y   = 23'h0;
                end else begin
                    m_y = m_y24[22:0];
                end

                // Overflow / underflow
                if (e_sum >= 10'sd255) begin
                    fp32_mul_beh = {s_y, 8'hFF, 23'h0};    // ±Inf
                end else if (e_sum <= 10'sd0) begin
                    fp32_mul_beh = {s_y, 31'h0};            // ±0
                end else begin
                    fp32_mul_beh = {s_y, e_sum[7:0], m_y};
                end
            end
        end
    endfunction

`ifdef VIVADO
    // Xilinx Floating Point v7.1 IP — FP32×FP32 multiply, C_LATENCY=8
    // NonBlocking flow control; ACLKEN gates pipeline (maps to enable).
    xil_fmul u_fmul (
        .aclk                (clk),
        .aclken              (enable),
        .s_axis_a_tvalid     (1'b1),
        .s_axis_a_tdata      (a),
        .s_axis_b_tvalid     (1'b1),
        .s_axis_b_tdata      (b),
        .m_axis_result_tvalid (),
        .m_axis_result_tdata  (y)
    );
`else
    // Behavioral model: LATENCY-cycle shift register (simulation only).
    // LATENCY must match xil_fmul C_LATENCY=8 so sim and synthesis agree.
    logic [31:0] y_comb;
    assign y_comb = fp32_mul_beh(a, b);

    logic [31:0] pipe_r [LATENCY];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < LATENCY; i++) pipe_r[i] <= '0;
        end else if (enable) begin
            pipe_r[0] <= y_comb;
            for (int i = 1; i < LATENCY; i++) pipe_r[i] <= pipe_r[i-1];
        end
    end

    assign y = pipe_r[LATENCY-1];
`endif

endmodule

`else // TCU_BHF

`include "VX_define.vh"

module VX_pssu_fmul #(
    parameter LATENCY = 2   // BHF: MUL_LATENCY=1 + RND_LATENCY=1
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] y
);
    wire [4:0] fflags_nc;

    VX_tcu_bhf_fmul #(
        .IN_EXPW     (8),
        .IN_SIGW     (24),
        .OUT_EXPW    (8),
        .OUT_SIGW    (24),
        .MUL_LATENCY (1),
        .RND_LATENCY (1),
        .IN_REC      (0),
        .OUT_REC     (0)
    ) u_fmul (
        .clk   (clk),
        .reset (reset),
        .enable(enable),
        .frm   (3'h0),   // round-to-nearest-even
        .a     (a),
        .b     (b),
        .y     (y),
        .fflags(fflags_nc)
    );

endmodule

`endif // TCU_BHF

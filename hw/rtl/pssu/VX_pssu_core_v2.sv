`timescale 1ns/1ps
// VX_pssu_core_v2 — Vortex PSSU execution core
//
// Two operations selected by `op`:
//
//   PSSU_SCALE (op=0):  rd = f(rs1) × rs2 × rs3
//     is_fp=0: f(rs1) = INT32_to_FP32(rs1)  (TCU INT8 path, default)
//     is_fp=1: f(rs1) = rs1 as-is (FP32 passthrough, TCU FP16 path)
//     Pipeline (LATENCY = 2 + 2×FMUL_LATENCY = 18cy with Xilinx 8cy FMUL / 6cy BHF):
//       Cycle 0 (comb):    sign + abs_v negation (INT32 path) / rs1 raw latch (FP32 path)
//       Cycle 1 (reg):     s0_sign_r, s0_abs_r, s0_rs1_r, s0_is_fp_r latched  [Stage S-1]
//       Cycle 1 (comb):    LZ tree + rounding adder + FP32 pack (INT32) / bypass (FP32)
//       Cycle 2 (reg):     int_fp32_r, sa_fp32_r, sb_fp32_r latched            [Stage S0]
//       Cycles 2..2+L:     FMUL#1  sa × sb  → combined
//       Cycles 2+L..2+2L:  FMUL#2  int_fp32_delayed × combined  → rd
//
//   PSSU_ACC (op=1):    rd = rs1 + rs2
//     Pipeline (LATENCY = 2cy, pipelined fp32_add):
//       Cycle 1 (reg):     alignment + add/sub intermediate state
//       Cycle 2 (reg):     normalize + pack → rd
//
//   PSSU_FMA (op=2):    rd = rs3 + f(rs1) × rs2   (fused scale+acc, throughput ×2 vs SCALE+ACC)
//     Reuses the SCALE datapath with sb(=FMUL#1's rs3 operand) forced to 1.0, so
//     fmul2_out = f(rs1)×rs2×1.0 = f(rs1)×rs2 at cycle SCALE_LATENCY.  A trailing
//     2-stage fp32 adder then adds rs3 (delayed SCALE_LATENCY to align), giving the
//     result at cycle FMA_LATENCY = SCALE_LATENCY + 2.  FMUL#1/#2 input timing is
//     identical to SCALE → no multiplier resource conflict.  is_fp gates f(rs1) as in
//     SCALE.  All three ops are padded to FMA_LATENCY (single fixed-latency unit).

module VX_pssu_core_v2 (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [1:0]  op,       // 0=PSSU_SCALE, 1=PSSU_ACC, 2=PSSU_FMA, 3=PSSU_FMA2
    input  wire        is_fp,    // SCALE/FMA: 0=rs1 INT32 (default), 1=rs1 FP32 passthrough
    // operands
    input  wire [31:0] rs1,      // INT32 partial_sum (is_fp=0) / FP32 value (is_fp=1)
    input  wire [31:0] rs2,      // FP32 scale_A  (SCALE/FMA) / FP32 addend_A (ACC)
    input  wire [31:0] rs3,      // FP32 scale_B  (SCALE, FMA2) / FP32 accumulator (FMA) / unused (ACC)
    input  wire [31:0] acc,      // FP32 accumulator (FMA2 only)
    output wire [31:0] rd        // FP32 result
);

`ifdef TCU_BHF
    localparam FMUL_LATENCY = 2;
`else
    localparam FMUL_LATENCY = 8;
`endif
    localparam SCALE_LATENCY = 2 + 2 * FMUL_LATENCY;  // 18cy (Xilinx 8cy×2+2) / 6cy (BHF 2cy×2+2)
    localparam FMA_LATENCY   = SCALE_LATENCY + 2;     // + trailing fp32 add; unified unit latency
    localparam OP_FMA        = 2'd2;
    localparam OP_FMA2       = 2'd3;  // rd = acc + f(rs1) × rs2 × rs3 (two-scale fused scale+acc)
    localparam OP_ACC        = 2'd1;

    // op delayed 1..FMA_LATENCY cycles.  op_pipe[i] = op delayed (i+1) cycles.
    //   op_pipe[0]            : selects sb mux at the S0 latch (instruction latched 1cy ago)
    //   op_pipe[FMA_LATENCY-1]: selects output mux (aligned to FMA_LATENCY)
    logic [1:0] op_pipe [FMA_LATENCY];
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < FMA_LATENCY; i++) op_pipe[i] <= '0;
        end else if (enable) begin
            op_pipe[0] <= op;
            for (int i = 1; i < FMA_LATENCY; i++) op_pipe[i] <= op_pipe[i-1];
        end
    end
    wire is_fma_s0 = (op_pipe[0] == OP_FMA);  // FMA at the S0 latch stage (sb mux)

    // -----------------------------------------------------------------------
    // PSSU_SCALE path
    // -----------------------------------------------------------------------

    // Stage S-1: sign + abs_v (32-bit 2's complement, 4×CARRY8 ≈ 1.0ns)
    // Breaks the critical path: abs carry chain is isolated in this stage.
    // Also latches raw rs1 and is_fp for the FP32 passthrough path.
    logic        s0_sign_r;
    logic [31:0] s0_abs_r;
    logic [31:0] s0_sa_r, s0_sb_r;
    logic [31:0] s0_rs1_r;   // raw rs1 for FP32 passthrough (is_fp=1)
    logic        s0_is_fp_r;

    always_ff @(posedge clk) begin
        if (reset) begin
            s0_sign_r  <= '0; s0_abs_r  <= '0;
            s0_sa_r    <= '0; s0_sb_r   <= '0;
            s0_rs1_r   <= '0; s0_is_fp_r <= '0;
        end else if (enable) begin
            s0_sign_r  <= rs1[31];
            s0_abs_r   <= rs1[31] ? (~rs1 + 32'd1) : rs1;
            s0_sa_r    <= rs2;
            s0_sb_r    <= rs3;
            s0_rs1_r   <= rs1;
            s0_is_fp_r <= is_fp;
        end
    end

    // Stage S0: LZ detection + rounding adder + FP32 pack (from stable s0_abs_r)
    // Module-level wires allow (* max_fanout *) to reduce high-fanout routing overhead.
    (* max_fanout = 8 *) logic l1_w, l2_w, l3_w, l4_w, l5_w;
    logic [31:0] m1_w, m2_w, m3_w, m4_w;
    assign l1_w = |s0_abs_r[31:16];
    assign m1_w = l1_w ? s0_abs_r           : {s0_abs_r[15:0], 16'b0};
    assign l2_w = |m1_w[31:24];
    assign m2_w = l2_w ? m1_w               : {m1_w[23:0],     8'b0};
    assign l3_w = |m2_w[31:28];
    assign m3_w = l3_w ? m2_w               : {m2_w[27:0],     4'b0};
    assign l4_w = |m3_w[31:30];
    assign m4_w = l4_w ? m3_w               : {m3_w[29:0],     2'b0};
    assign l5_w = m4_w[31];

    logic [4:0]  lz_w;
    logic [30:0] abs_shifted_w;
    logic        round_bit_w, sticky_w;
    logic [23:0] mant24_w;

    assign lz_w          = {~l1_w, ~l2_w, ~l3_w, ~l4_w, ~l5_w};
    assign abs_shifted_w = l5_w ? m4_w[30:0] : {m4_w[29:0], 1'b0};
    assign round_bit_w   = abs_shifted_w[7];
    assign sticky_w      = |abs_shifted_w[6:0];
    assign mant24_w      = {1'b0, abs_shifted_w[30:8]}
                         + {23'h0, (round_bit_w && (sticky_w || abs_shifted_w[8]))};

    wire [31:0] int_fp32_c =
        s0_is_fp_r ? s0_rs1_r  // FP32 passthrough: rs1 is already IEEE 754
                   : (s0_abs_r == '0) ? 32'h0
                                      : {s0_sign_r,
                                         8'd127 + 8'd31 - {3'h0, lz_w} + {7'h0, mant24_w[23]},
                                         mant24_w[23] ? 23'h0 : mant24_w[22:0]};

    // Stage S0→S1 register
    logic [31:0] int_fp32_r, sa_fp32_r, sb_fp32_r;

    always_ff @(posedge clk) begin
        if (reset) begin
            int_fp32_r <= '0;
            sa_fp32_r  <= '0;
            sb_fp32_r  <= '0;
        end else if (enable) begin
            int_fp32_r <= int_fp32_c;
            sa_fp32_r  <= s0_sa_r;   // 1 extra delay via S-1 to align with 2-stage abs path
            // FMA forces sb=1.0 so FMUL#1=rs2×1.0=rs2, FMUL#2=f(rs1)×rs2 (rs3 is the acc addend, not a scale)
            sb_fp32_r  <= is_fma_s0 ? 32'h3F80_0000 : s0_sb_r;
        end
    end

    // FMUL#1: sa × sb → combined_scale
    wire [31:0] fmul1_out;

    VX_pssu_fmul #(.LATENCY(FMUL_LATENCY)) u_fmul1 (
        .clk   (clk),
        .reset (reset),
        .enable(enable),
        .a     (sa_fp32_r),
        .b     (sb_fp32_r),
        .y     (fmul1_out)
    );

    // int_fp32_r delayed FMUL_LATENCY cycles to align with fmul1_out
    logic [31:0] int_delay_r [FMUL_LATENCY];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < FMUL_LATENCY; i++) int_delay_r[i] <= '0;
        end else if (enable) begin
            int_delay_r[0] <= int_fp32_r;
            for (int i = 1; i < FMUL_LATENCY; i++) int_delay_r[i] <= int_delay_r[i-1];
        end
    end

    // FMUL#2: int_fp32_delayed × combined_scale → scaled result
    wire [31:0] fmul2_out;

    VX_pssu_fmul #(.LATENCY(FMUL_LATENCY)) u_fmul2 (
        .clk   (clk),
        .reset (reset),
        .enable(enable),
        .a     (int_delay_r[FMUL_LATENCY-1]),
        .b     (fmul1_out),
        .y     (fmul2_out)
    );

    // -----------------------------------------------------------------------
    // PSSU_ACC path: rd = rs1 + rs2  (FP32 add, 2-stage pipelined)
    //
    // Stage 1: special-case detect + mantissa alignment + add/subtract
    // Stage 2: normalize + pack
    //
    // Packed stage-1 bus: {special(1), special_val(32), sign(1), e_big(8), m_sum(25)} = 67b
    // -----------------------------------------------------------------------

    // Stage 1: alignment + add/subtract
    // Mantissas carry 3 extra low bits (guard, round, sticky) so stage 2 can round
    // to nearest even like an IEEE fadd. Layout (28b): [27]=carry, [26]=hidden,
    // [25:3]=fraction, [2]=G, [1]=R, [0]=S. Subnormal inputs are treated as zero.
    // Packed stage-1 bus: {special(1), special_val(32), sign(1), e_big(8), m_sum(28)} = 70b
    function automatic logic [69:0] fp32_add_s1(
        input logic [31:0] fa, fb
    );
        logic        s_a, s_b, s_y;
        logic [7:0]  e_a, e_b, e_big, e_diff;
        logic [27:0] m_a, m_b, m_big, m_sml, m_sum, m_shr;
        logic        sticky;
        logic [31:0] spec_val;
        begin
            s_a = fa[31]; e_a = fa[30:23]; m_a = {2'b01, fa[22:0], 3'b000};
            s_b = fb[31]; e_b = fb[30:23]; m_b = {2'b01, fb[22:0], 3'b000};
            s_y = 1'b0; e_big = 8'h0; m_sum = 28'h0; spec_val = 32'h0;
            if (e_a == 8'hFF || e_b == 8'hFF) begin
                if ((e_a == 8'hFF && fa[22:0] != 23'h0) ||
                    (e_b == 8'hFF && fb[22:0] != 23'h0))
                    spec_val = 32'hFFC00000;
                else if (e_a == 8'hFF && e_b == 8'hFF && s_a != s_b)
                    spec_val = 32'hFFC00000;
                else
                    spec_val = (e_a >= e_b) ? fa : fb;
                fp32_add_s1 = {1'b1, spec_val, s_y, e_big, m_sum};
            end else if (e_a == 8'h00) begin
                fp32_add_s1 = {1'b1, fb, s_y, e_big, m_sum};
            end else if (e_b == 8'h00) begin
                fp32_add_s1 = {1'b1, fa, s_y, e_big, m_sum};
            end else begin
                if (e_a >= e_b) begin
                    e_big = e_a; s_y = s_a; m_big = m_a; m_sml = m_b; e_diff = e_a - e_b;
                end else begin
                    e_big = e_b; s_y = s_b; m_big = m_b; m_sml = m_a; e_diff = e_b - e_a;
                end
                // right shift with sticky: every bit shifted out is OR-ed into bit 0
                if (e_diff >= 8'd27) begin
                    m_shr  = 28'h0;
                    sticky = (m_sml != 28'h0);
                end else begin
                    m_shr  = m_sml >> e_diff;
                    sticky = ((m_sml & ((28'h1 << e_diff) - 28'h1)) != 28'h0);
                end
                m_sml = m_shr | {27'h0, sticky};
                if (s_a == s_b)
                    m_sum = m_big + m_sml;
                else if (m_big >= m_sml)
                    m_sum = m_big - m_sml;
                else begin
                    m_sum = m_sml - m_big; s_y = ~s_y;
                end
                fp32_add_s1 = {1'b0, spec_val, s_y, e_big, m_sum};
            end
        end
    endfunction

    // Stage 2: normalize + round to nearest even + pack
    function automatic logic [31:0] fp32_add_s2(
        input logic [69:0] s1
    );
        logic        special;
        logic [31:0] spec_val;
        logic        s_y;
        logic [7:0]  e_big;
        logic [27:0] m_sum;
        logic [26:0] m;
        logic [4:0]  lz;
        logic [9:0]  e;
        logic [24:0] mant;
        logic        round_up;
        begin
            {special, spec_val, s_y, e_big, m_sum} = s1;
            if (special) begin
                fp32_add_s2 = spec_val;
            end else if (m_sum == 28'h0) begin
                fp32_add_s2 = 32'h0;
            end else begin
                if (m_sum[27]) begin
                    m = {m_sum[27:2], m_sum[1] | m_sum[0]};
                    e = {2'b00, e_big} + 10'd1;
                end else begin
                    lz = 5'd0;
                    for (integer b = 0; b < 27; b++) begin
                        if (m_sum[b]) lz = 5'(26 - b);
                    end
                    m = m_sum[26:0] << lz;
                    e = {2'b00, e_big} - {5'b0, lz};
                end
                round_up = m[2] & (m[1] | m[0] | m[3]);
                mant = {1'b0, m[26:3]} + {24'h0, round_up};
                if (mant[24]) begin
                    mant = mant >> 1;
                    e = e + 10'd1;
                end
                if ($signed(e) <= 0) begin
                    fp32_add_s2 = {s_y, 31'h0};                   // underflow: flush to zero
                end else if (e >= 10'd255) begin
                    fp32_add_s2 = {s_y, 8'hFF, 23'h0};            // overflow: infinity
                end else begin
                    fp32_add_s2 = {s_y, e[7:0], mant[22:0]};
                end
            end
        end
    endfunction

    // Stage 1 pipeline register (70-bit packed intermediate state)
    logic [69:0] acc_s1_r;

    always_ff @(posedge clk) begin
        if (reset)
            acc_s1_r <= '0;
        else if (enable)
            acc_s1_r <= fp32_add_s1(rs1, rs2);
    end

    // Stage 2 output register
    (* shreg_extract = "no" *) logic [31:0] acc_out_r;

    always_ff @(posedge clk) begin
        if (reset)
            acc_out_r <= '0;
        else if (enable)
            acc_out_r <= fp32_add_s2(acc_s1_r);
    end

    // -----------------------------------------------------------------------
    // PSSU_FMA trailing adder: rd = rs3(acc) + fmul2_out(=f(rs1)×rs2)
    //
    // fmul2_out is valid at cycle SCALE_LATENCY.  rs3 is delayed SCALE_LATENCY to
    // align, then a 2-stage fp32 adder (reusing fp32_add_s1/s2) produces the FMA
    // result at cycle FMA_LATENCY = SCALE_LATENCY + 2.
    // -----------------------------------------------------------------------

    // rs3 (acc addend) delayed SCALE_LATENCY cycles: acc_add_delay_r[i] = rs3 delayed (i+1)
    logic [31:0] acc_add_delay_r [SCALE_LATENCY];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < SCALE_LATENCY; i++) acc_add_delay_r[i] <= '0;
        end else if (enable) begin
            acc_add_delay_r[0] <= (op == OP_FMA2) ? acc : rs3;
            for (int i = 1; i < SCALE_LATENCY; i++) acc_add_delay_r[i] <= acc_add_delay_r[i-1];
        end
    end

    logic [69:0] fma_s1_r;   // FMA adder stage-1 (at SCALE_LATENCY+1)
    logic [31:0] fma_out_r;  // FMA result       (at SCALE_LATENCY+2 = FMA_LATENCY)

    always_ff @(posedge clk) begin
        if (reset) begin
            fma_s1_r  <= '0;
            fma_out_r <= '0;
        end else if (enable) begin
            fma_s1_r  <= fp32_add_s1(fmul2_out, acc_add_delay_r[SCALE_LATENCY-1]);
            fma_out_r <= fp32_add_s2(fma_s1_r);
        end
    end

    // -----------------------------------------------------------------------
    // Output selection — all three ops padded to FMA_LATENCY (single fixed-latency unit)
    // PSSU_SCALE: fmul2_out at SCALE_LATENCY, padded +2.
    // PSSU_ACC:   acc_out_r at cycle 2, padded to FMA_LATENCY.
    // PSSU_FMA:   fma_out_r at FMA_LATENCY.
    // -----------------------------------------------------------------------

    // SCALE output padded +2 to reach FMA_LATENCY
    logic [31:0] scale_pad_r [2];

    always_ff @(posedge clk) begin
        if (reset) begin
            scale_pad_r[0] <= '0; scale_pad_r[1] <= '0;
        end else if (enable) begin
            scale_pad_r[0] <= fmul2_out;
            scale_pad_r[1] <= scale_pad_r[0];
        end
    end

    // PSSU_ACC output is 2-cycle; pad it to FMA_LATENCY with a shift register.
    logic [31:0] acc_delay_r [FMA_LATENCY-2];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < FMA_LATENCY-2; i++) acc_delay_r[i] <= '0;
        end else if (enable) begin
            acc_delay_r[0] <= acc_out_r;
            for (int i = 1; i < FMA_LATENCY-2; i++) acc_delay_r[i] <= acc_delay_r[i-1];
        end
    end

    wire [1:0] op_out = op_pipe[FMA_LATENCY-1];  // op aligned to output stage
    assign rd = (op_out == OP_FMA || op_out == OP_FMA2) ? fma_out_r
              : (op_out == OP_ACC) ? acc_delay_r[FMA_LATENCY-3]
                                   : scale_pad_r[1];

endmodule

// Testbench for VX_pssu_core_v2
//
// Tests:
//   [A] PSSU_SCALE: rd = INT32_to_FP32(rs1) × rs2 × rs3
//   [B] PSSU_ACC:   rd = rs1 + rs2
//   [C] Multi-group: SCALE → ACC → SCALE → ACC (realistic GEMM sequence)
//   [D] Negative partial_sum accumulation

`timescale 1ns/1ps

module tb_pssu_core_v2;

`ifdef TCU_BHF
    localparam FMUL_LATENCY  = 2;
`else
    localparam FMUL_LATENCY  = 8;
`endif
    localparam SCALE_LATENCY = 2 + 2 * FMUL_LATENCY;
    localparam FMA_LATENCY   = SCALE_LATENCY + 2;  // unified unit latency (all ops padded to this)

    logic        clk, reset, enable;
    logic [1:0]  op;
    logic        is_fp;
    logic [31:0] rs1, rs2, rs3, rd;
    logic [31:0] acc = '0;   // FMA2 accumulator input (op 3, not tested here)

    VX_pssu_core_v2 dut (.*);

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function automatic logic [31:0] to_fp32(input real v);
        return $shortrealtobits(shortreal'(v));
    endfunction

    function automatic real from_fp32(input logic [31:0] b);
        return real'($bitstoshortreal(b));
    endfunction

    function automatic logic fp32_close(input logic [31:0] got, input real exp_r);
        real g, e, diff;
        g    = from_fp32(got);
        e    = exp_r;
        diff = g - e; if (diff < 0.0) diff = -diff;
        if (e < 0.0) e = -e;
        if (e < 1e-30) return (diff < 1e-15);
        else           return (diff / e) < 1e-4;
    endfunction

    int pass_cnt, fail_cnt;

    task automatic chk(input string name, input logic [31:0] got, input real exp_r);
        if (fp32_close(got, exp_r)) begin
            $display("  PASS  %-45s  got=%e  exp=%e", name, from_fp32(got), exp_r);
            pass_cnt++;
        end else begin
            $display("  FAIL  %-45s  got=%e  exp=%e", name, from_fp32(got), exp_r);
            fail_cnt++;
        end
    endtask

    task automatic apply_scale(
        input  logic [31:0] p_rs1,
        input  logic [31:0] p_rs2,
        input  logic [31:0] p_rs3,
        output logic [31:0] result
    );
        @(posedge clk); #1;
        op = 2'd0; is_fp = 0; rs1 = p_rs1; rs2 = p_rs2; rs3 = p_rs3; enable = 1;
        repeat (FMA_LATENCY) @(posedge clk);
        #1; result = rd;
        enable = 0; op = 2'd0; is_fp = 0; rs1 = 0; rs2 = 0; rs3 = 0;
        @(posedge clk); #1;
    endtask

    // is_fp=1: rs1 is already FP32 (bit-identical passthrough), not INT32.
    task automatic apply_scale_fp(
        input  logic [31:0] p_rs1,
        input  logic [31:0] p_rs2,
        input  logic [31:0] p_rs3,
        output logic [31:0] result
    );
        @(posedge clk); #1;
        op = 2'd0; is_fp = 1; rs1 = p_rs1; rs2 = p_rs2; rs3 = p_rs3; enable = 1;
        repeat (FMA_LATENCY) @(posedge clk);
        #1; result = rd;
        enable = 0; op = 2'd0; is_fp = 0; rs1 = 0; rs2 = 0; rs3 = 0;
        @(posedge clk); #1;
    endtask

    task automatic apply_acc(
        input  logic [31:0] p_rs1,
        input  logic [31:0] p_rs2,
        output logic [31:0] result
    );
        @(posedge clk); #1;
        op = 2'd1; is_fp = 0; rs1 = p_rs1; rs2 = p_rs2; rs3 = 0; enable = 1;
        repeat (FMA_LATENCY) @(posedge clk);
        #1; result = rd;
        enable = 0; op = 2'd0; is_fp = 0; rs1 = 0; rs2 = 0; rs3 = 0;
        @(posedge clk); #1;
    endtask

    // PSSU_FMA: rd = rs3 + f(rs1) × rs2  (is_fp gates f as in SCALE)
    task automatic apply_fma(
        input  logic        p_is_fp,
        input  logic [31:0] p_rs1,   // fragC
        input  logic [31:0] p_rs2,   // sw
        input  logic [31:0] p_rs3,   // acc
        output logic [31:0] result
    );
        @(posedge clk); #1;
        op = 2'd2; is_fp = p_is_fp; rs1 = p_rs1; rs2 = p_rs2; rs3 = p_rs3; enable = 1;
        repeat (FMA_LATENCY) @(posedge clk);
        #1; result = rd;
        enable = 0; op = 2'd0; is_fp = 0; rs1 = 0; rs2 = 0; rs3 = 0;
        @(posedge clk); #1;
    endtask

    // Module-level temporaries used in initial block
    logic [31:0] result;
    logic [31:0] acc_reg;
    real         expected_acc;

    // -----------------------------------------------------------------------
    // Test sequence
    // -----------------------------------------------------------------------
    initial begin
        pass_cnt = 0; fail_cnt = 0;
        enable = 0; op = 0; is_fp = 0; rs1 = 0; rs2 = 0; rs3 = 0;
        reset  = 1;
        repeat (4) @(posedge clk);
        reset = 0;
        @(posedge clk); #1;

        // ===================================================================
        $display("\n=== Group A: PSSU_SCALE ===");

        // A1: 4 × 2.0 × 0.5 = 4.0
        apply_scale(32'd4, to_fp32(2.0), to_fp32(0.5), result);
        chk("A1: INT32(4) × 2.0 × 0.5 = 4.0", result, 4.0);

        // A2: 1 × 1.0 × 1.0 = 1.0
        apply_scale(32'd1, to_fp32(1.0), to_fp32(1.0), result);
        chk("A2: INT32(1) × 1.0 × 1.0 = 1.0", result, 1.0);

        // A3: -8 × 0.5 × 0.25 = -1.0
        apply_scale(32'hFFFFFFF8, to_fp32(0.5), to_fp32(0.25), result);
        chk("A3: INT32(-8) × 0.5 × 0.25 = -1.0", result, -1.0);

        // A4: 100 × 2.0 × 4.0 = 800.0
        apply_scale(32'd100, to_fp32(2.0), to_fp32(4.0), result);
        chk("A4: INT32(100) × 2.0 × 4.0 = 800.0", result, 800.0);

        // A5: 0 × 2.0 × 2.0 = 0.0
        apply_scale(32'd0, to_fp32(2.0), to_fp32(2.0), result);
        chk("A5: INT32(0) × 2.0 × 2.0 = 0.0", result, 0.0);

        // A6: -1 × 2.0 × 2.0 = -4.0
        apply_scale(32'hFFFFFFFF, to_fp32(2.0), to_fp32(2.0), result);
        chk("A6: INT32(-1) × 2.0 × 2.0 = -4.0", result, -4.0);

        // A7: 1000000 × 1e-3 × 1e-3 = 1.0
        apply_scale(32'd1000000, to_fp32(0.001), to_fp32(0.001), result);
        chk("A7: INT32(1000000) × 1e-3 × 1e-3 = 1.0", result, 1.0);

        // A8: INT32_MAX × 1.0 × 1.0 ≈ 2.147484e9
        apply_scale(32'h7FFFFFFF, to_fp32(1.0), to_fp32(1.0), result);
        chk("A8: INT32_MAX × 1.0 × 1.0 = 2147483647.0", result, 2147483647.0);

        // A9: INT32_MIN × 1.0 × 1.0 = -2147483648.0
        apply_scale(32'h80000000, to_fp32(1.0), to_fp32(1.0), result);
        chk("A9: INT32_MIN × 1.0 × 1.0 = -2147483648.0", result, -2147483648.0);

        // A10: 7 × (-1.5) × 2.0 = -21.0
        apply_scale(32'd7, to_fp32(-1.5), to_fp32(2.0), result);
        chk("A10: INT32(7) × (-1.5) × 2.0 = -21.0", result, -21.0);

        // ===================================================================
        $display("\n=== Group B: PSSU_ACC ===");

        // B1: 1.0 + 2.0 = 3.0
        apply_acc(to_fp32(1.0), to_fp32(2.0), result);
        chk("B1: 1.0 + 2.0 = 3.0", result, 3.0);

        // B2: 0.0 + 4.0 = 4.0 (first-group init with vzero)
        apply_acc(32'h0, to_fp32(4.0), result);
        chk("B2: 0.0 + 4.0 = 4.0 (vzero init)", result, 4.0);

        // B3: -1.5 + 1.0 = -0.5
        apply_acc(to_fp32(-1.5), to_fp32(1.0), result);
        chk("B3: -1.5 + 1.0 = -0.5", result, -0.5);

        // B4: 100.0 + (-100.0) = 0.0
        apply_acc(to_fp32(100.0), to_fp32(-100.0), result);
        chk("B4: 100.0 + (-100.0) = 0.0", result, 0.0);

        // B5: 1.5e10 + 2.5e10 = 4.0e10
        apply_acc(to_fp32(1.5e10), to_fp32(2.5e10), result);
        chk("B5: 1.5e10 + 2.5e10 = 4.0e10", result, 4.0e10);

        // ===================================================================
        $display("\n=== Group C: Multi-group GEMM sequence ===");
        // group 0: ps=10,  sA=0.1, sB=0.2 → 0.2,  acc=0.2
        // group 1: ps=20,  sA=0.2, sB=0.3 → 1.2,  acc=1.4
        // group 2: ps=5,   sA=0.5, sB=0.4 → 1.0,  acc=2.4

        acc_reg = 32'h0; expected_acc = 0.0;

        apply_scale(32'd10, to_fp32(0.1), to_fp32(0.2), result);
        chk("C1a: SCALE g0 (10×0.1×0.2=0.2)", result, 0.2);
        apply_acc(acc_reg, result, acc_reg);
        expected_acc = expected_acc + 10.0 * 0.1 * 0.2;
        chk("C1b: ACC g0", acc_reg, expected_acc);

        apply_scale(32'd20, to_fp32(0.2), to_fp32(0.3), result);
        chk("C2a: SCALE g1 (20×0.2×0.3=1.2)", result, 1.2);
        apply_acc(acc_reg, result, acc_reg);
        expected_acc = expected_acc + 20.0 * 0.2 * 0.3;
        chk("C2b: ACC g1", acc_reg, expected_acc);

        apply_scale(32'd5, to_fp32(0.5), to_fp32(0.4), result);
        chk("C3a: SCALE g2 (5×0.5×0.4=1.0)", result, 1.0);
        apply_acc(acc_reg, result, acc_reg);
        expected_acc = expected_acc + 5.0 * 0.5 * 0.4;
        chk("C3b: ACC g2", acc_reg, expected_acc);

        // ===================================================================
        $display("\n=== Group D: Negative partial_sum accumulation ===");
        // group 0: ps=-50, sA=0.01, sB=2.0 → -1.0, acc=-1.0
        // group 1: ps=100, sA=0.01, sB=2.0 →  2.0, acc= 1.0

        acc_reg = 32'h0; expected_acc = 0.0;

        apply_scale(32'hFFFFFFCE, to_fp32(0.01), to_fp32(2.0), result); // -50 = 0xFFFFFFCE
        chk("D1a: SCALE -50×0.01×2.0=-1.0", result, -1.0);
        apply_acc(acc_reg, result, acc_reg);
        expected_acc = expected_acc + (-50.0) * 0.01 * 2.0;
        chk("D1b: ACC g0", acc_reg, expected_acc);

        apply_scale(32'd100, to_fp32(0.01), to_fp32(2.0), result);
        chk("D2a: SCALE 100×0.01×2.0=2.0", result, 2.0);
        apply_acc(acc_reg, result, acc_reg);
        expected_acc = expected_acc + 100.0 * 0.01 * 2.0;
        chk("D2b: ACC g1", acc_reg, expected_acc);

        // ===================================================================
        $display("\n=== Group E: PSSU_SCALE is_fp=1 (FP32 passthrough, TCU FP16 path) ===");

        // E1: same 32'h41200000 bit pattern, two interpretations -- proves the
        // is_fp mux actually changes behavior (not just a dead wire).
        //   is_fp=0: INT32(1092616192) x 2.0 x 0.5 = 1092616192.0
        //   is_fp=1: FP32(10.0)        x 2.0 x 0.5 = 10.0
        apply_scale(32'h41200000, to_fp32(2.0), to_fp32(0.5), result);
        chk("E1a: is_fp=0, INT32(0x41200000) x 1.0 = 1092616192.0", result, 1092616192.0);
        apply_scale_fp(32'h41200000, to_fp32(2.0), to_fp32(0.5), result);
        chk("E1b: is_fp=1, FP32(0x41200000=10.0) x 1.0 = 10.0", result, 10.0);

        // E2: FP32(1.0) × 1.0 × 1.0 = 1.0
        apply_scale_fp(to_fp32(1.0), to_fp32(1.0), to_fp32(1.0), result);
        chk("E2: FP32(1.0) x 1.0 x 1.0 = 1.0", result, 1.0);

        // E3: FP32(-8.0) × 0.5 × 0.25 = -1.0
        apply_scale_fp(to_fp32(-8.0), to_fp32(0.5), to_fp32(0.25), result);
        chk("E3: FP32(-8.0) x 0.5 x 0.25 = -1.0", result, -1.0);

        // E4: FP32(0.0) × 2.0 × 2.0 = 0.0
        apply_scale_fp(to_fp32(0.0), to_fp32(2.0), to_fp32(2.0), result);
        chk("E4: FP32(0.0) x 2.0 x 2.0 = 0.0", result, 0.0);

        // E5: FP32(-1.5) × 2.0 × 2.0 = -6.0
        apply_scale_fp(to_fp32(-1.5), to_fp32(2.0), to_fp32(2.0), result);
        chk("E5: FP32(-1.5) x 2.0 x 2.0 = -6.0", result, -6.0);

        // E6: mode isolation -- op sequence is_fp=1 then is_fp=0 back-to-back must
        // not leak state (checks s0_is_fp_r is re-latched every cycle, not sticky).
        apply_scale_fp(to_fp32(3.0), to_fp32(2.0), to_fp32(1.0), result);
        chk("E6a: FP32(3.0) x 2.0 x 1.0 = 6.0", result, 6.0);
        apply_scale(32'd3, to_fp32(2.0), to_fp32(1.0), result);
        chk("E6b: INT32(3) x 2.0 x 1.0 = 6.0 (is_fp reverts to 0)", result, 6.0);

        // ===================================================================
        $display("\n=== Group F: PSSU_FMA (fused scale+acc): rd = rs3 + f(rs1) x rs2 ===");

        // F1: is_fp=1, 1.0 + FP32(2.0)*3.0 = 7.0
        apply_fma(1'b1, to_fp32(2.0), to_fp32(3.0), to_fp32(1.0), result);
        chk("F1: 1.0 + FP32(2.0)x3.0 = 7.0", result, 7.0);

        // F2: is_fp=1, 0.0 + FP32(4.0)*0.5 = 2.0  (first-group init, acc=0)
        apply_fma(1'b1, to_fp32(4.0), to_fp32(0.5), 32'h0, result);
        chk("F2: 0.0 + FP32(4.0)x0.5 = 2.0 (init)", result, 2.0);

        // F3: is_fp=0, 10.0 + INT32(5)*2.0 = 20.0  (INT8 path fragC)
        apply_fma(1'b0, 32'd5, to_fp32(2.0), to_fp32(10.0), result);
        chk("F3: 10.0 + INT32(5)x2.0 = 20.0", result, 20.0);

        // F4: is_fp=1, negative product: 5.0 + FP32(-3.0)*2.0 = -1.0
        apply_fma(1'b1, to_fp32(-3.0), to_fp32(2.0), to_fp32(5.0), result);
        chk("F4: 5.0 + FP32(-3.0)x2.0 = -1.0", result, -1.0);

        // F5: FMA accumulation chain (the rescale use case: acc <- acc + fragC*sw)
        //   g0: fragC=10.0, sw=0.1 -> acc=1.0
        //   g1: fragC=20.0, sw=0.2 -> acc=5.0
        //   g2: fragC=5.0,  sw=0.4 -> acc=7.0
        acc_reg = 32'h0; expected_acc = 0.0;
        apply_fma(1'b1, to_fp32(10.0), to_fp32(0.1), acc_reg, acc_reg);
        expected_acc = expected_acc + 10.0 * 0.1;
        chk("F5a: FMA chain g0", acc_reg, expected_acc);
        apply_fma(1'b1, to_fp32(20.0), to_fp32(0.2), acc_reg, acc_reg);
        expected_acc = expected_acc + 20.0 * 0.2;
        chk("F5b: FMA chain g1", acc_reg, expected_acc);
        apply_fma(1'b1, to_fp32(5.0), to_fp32(0.4), acc_reg, acc_reg);
        expected_acc = expected_acc + 5.0 * 0.4;
        chk("F5c: FMA chain g2", acc_reg, expected_acc);

        // F6: op isolation -- FMA then SCALE back-to-back must not leak op/sb state
        apply_fma(1'b1, to_fp32(2.0), to_fp32(2.0), to_fp32(1.0), result);
        chk("F6a: FMA 1.0 + 2.0x2.0 = 5.0", result, 5.0);
        apply_scale_fp(to_fp32(3.0), to_fp32(2.0), to_fp32(1.0), result);
        chk("F6b: SCALE FP32(3.0)x2.0x1.0 = 6.0 (op reverts to SCALE)", result, 6.0);

        // ===================================================================
        $display("\n--- RESULT: %0d PASS / %0d FAIL ---\n", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS");
        else               $display("SOME TESTS FAILED");
        $finish;
    end

    initial begin
        #200000; $display("TIMEOUT"); $finish;
    end

endmodule

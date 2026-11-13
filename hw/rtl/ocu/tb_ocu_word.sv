`timescale 1ns/1ps
// tb_ocu_word.sv — functional test of VX_ocu_word
//
// Test cases:
//   O1-O4: INT4S→INT8 nibble slicing and step_w packing
//   O5-O5c: FP8E4M3→FP16 conversion and step_w selection
//   O6-O7: zero_point behavior
//   O8: FP4E2M1→FP16 all codes, zero_point ignored (±4, ±6 are not Inf/NaN)
//
// checks elem_valid and dst_data[i] directly, plus a model of the output packing MUX

`include "VX_ocu_pkg.sv"

module tb_ocu_word;
    import VX_ocu_pkg::*;

    // TCU format IDs (match VX_tcu_pkg.sv)
    localparam [3:0] TCU_FP16_ID    = 4'd1;
    localparam [3:0] TCU_FP8E4M3_ID = 4'd5;
    localparam [3:0] TCU_FP8E5M2_ID = 4'd6;
    localparam [3:0] TCU_FP4E2M1_ID = 4'd7;
    localparam [3:0] TCU_I8_ID      = 4'd9;
    localparam [3:0] TCU_I4_ID      = 4'd11;

    localparam N_ELEM = 8;

    // DUT inputs
    logic        clk    = 0;
    logic        reset  = 1;
    logic        valid_in  = 1;
    logic        ready_out = 1;
    tag_t        tag_in    = '0;
    logic [31:0] packed_data;
    logic  [3:0] fmt_s;
    logic  [3:0] fmt_d;
    logic  [7:0] zero_point;

    // DUT outputs
    logic                           ready_in;
    logic                           valid_out;
    tag_t                           tag_out;
    logic [N_ELEM-1:0][DST_DATA_WIDTH-1:0] dst_data;
    logic [N_ELEM-1:0]              elem_valid;

    // DUT
    VX_ocu_word dut (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .tag_in     (tag_in),
        .packed_data(packed_data),
        .fmt_s      (fmt_s),
        .fmt_d      (fmt_d),
        .zero_point (zero_point),
        .valid_out  (valid_out),
        .ready_out  (ready_out),
        .tag_out    (tag_out),
        .dst_data   (dst_data),
        .elem_valid (elem_valid)
    );

    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // PASS/FAIL counters
    // -------------------------------------------------------------------------
    int pass_cnt = 0, fail_cnt = 0;

    task automatic chk16(
        input string name,
        input logic [15:0] got,
        input logic [15:0] exp
    );
        if (got === exp) begin
            $display("  PASS  %-55s  got=0x%04X", name, got);
            pass_cnt++;
        end else begin
            $display("  FAIL  %-55s  got=0x%04X  exp=0x%04X", name, got, exp);
            fail_cnt++;
        end
    endtask

    task automatic chk8(
        input string name,
        input logic [7:0] got,
        input logic [7:0] exp
    );
        if (got === exp) begin
            $display("  PASS  %-55s  got=0x%02X", name, got);
            pass_cnt++;
        end else begin
            $display("  FAIL  %-55s  got=0x%02X  exp=0x%02X", name, got, exp);
            fail_cnt++;
        end
    endtask

    task automatic chk32(
        input string name,
        input logic [31:0] got,
        input logic [31:0] exp
    );
        if (got === exp) begin
            $display("  PASS  %-55s  got=0x%08X", name, got);
            pass_cnt++;
        end else begin
            $display("  FAIL  %-55s  got=0x%08X  exp=0x%08X", name, got, exp);
            fail_cnt++;
        end
    endtask

    // -------------------------------------------------------------------------
    // output packing MUX model: dst_data → rd32
    // fmt_d == TCU_FP16_ID: 2 × FP16
    // else:                  4 × INT8 (lower byte)
    // -------------------------------------------------------------------------
    function automatic logic [31:0] pack_rd32(
        input logic [N_ELEM-1:0][15:0] d,
        input logic [3:0] fd,
        input logic sw
    );
        if (fd == TCU_FP16_ID) begin
            return sw ? {d[6], d[4]} : {d[2], d[0]};
        end else begin
            return sw ? {d[7][7:0], d[6][7:0], d[5][7:0], d[4][7:0]}
                      : {d[3][7:0], d[2][7:0], d[1][7:0], d[0][7:0]};
        end
    endfunction

    task automatic apply(
        input logic [31:0] pd,
        input logic [3:0]  fs,
        input logic [3:0]  fd,
        input logic [7:0]  zp
    );
        packed_data  = pd;
        fmt_s        = fs;
        fmt_d        = fd;
        zero_point   = zp;
        #2;  // wait for the combinational logic to settle
    endtask

    // -------------------------------------------------------------------------
    // test sequence
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_ocu_word.vcd");
        $dumpvars(0, tb_ocu_word);

        packed_data = '0; fmt_s = '0; fmt_d = '0; zero_point = '0;
        @(posedge clk); #1; reset = 0;
        @(posedge clk); #1;

        // ===================================================================
        $display("\n=== O1: INT4S→INT8, 0x87654321, step_w=0 ===");
        apply(32'h87654321, TCU_I4_ID, TCU_I8_ID, 8'd0);

        // per-element check (nibble order: LSB→elem[0])
        chk16("dst_data[0] nibble0=1 → INT8=0x01",  dst_data[0], 16'h0001);
        chk16("dst_data[1] nibble1=2 → INT8=0x02",  dst_data[1], 16'h0002);
        chk16("dst_data[2] nibble2=3 → INT8=0x03",  dst_data[2], 16'h0003);
        chk16("dst_data[3] nibble3=4 → INT8=0x04",  dst_data[3], 16'h0004);
        chk16("dst_data[4] nibble4=5 → INT8=0x05",  dst_data[4], 16'h0005);
        chk16("dst_data[5] nibble5=6 → INT8=0x06",  dst_data[5], 16'h0006);
        chk16("dst_data[6] nibble6=7 → INT8=0x07",  dst_data[6], 16'h0007);
        chk16("dst_data[7] nibble7=8→-8 → INT8=0xF8", dst_data[7], 16'hFFF8);
        chk8("elem_valid==8'hFF (INT4, all valid)", elem_valid, 8'hFF);
        // packing MUX: step_w=0
        chk32("O1 rd32 step_w=0 = 0x04030201",
              pack_rd32(dst_data, TCU_I8_ID, 1'b0), 32'h04030201);

        // ===================================================================
        $display("\n=== O2: INT4S→INT8, 0x87654321, step_w=1 ===");
        // same input, step_w=1
        chk32("O2 rd32 step_w=1 = 0xF8070605",
              pack_rd32(dst_data, TCU_I8_ID, 1'b1), 32'hF8070605);

        // ===================================================================
        $display("\n=== O3: INT4S→INT8, 0x98765432, step_w=0 ===");
        apply(32'h98765432, TCU_I4_ID, TCU_I8_ID, 8'd0);

        // nibble[0..7] = 2,3,4,5,6,7,-8,-7
        chk16("dst_data[6] nibble6=8→-8 → 0xFFF8", dst_data[6], 16'hFFF8);
        chk16("dst_data[7] nibble7=9→-7 → 0xFFF9", dst_data[7], 16'hFFF9);
        chk32("O3 rd32 step_w=0 = 0x05040302",
              pack_rd32(dst_data, TCU_I8_ID, 1'b0), 32'h05040302);

        // ===================================================================
        $display("\n=== O4: INT4S→INT8, 0x98765432, step_w=1 ===");
        chk32("O4 rd32 step_w=1 = 0xF9F80706",
              pack_rd32(dst_data, TCU_I8_ID, 1'b1), 32'hF9F80706);

        // ===================================================================
        $display("\n=== O5: FP8E4M3→FP16, 0x3C003C00 (symmetric input) ===");
        apply(32'h3C003C00, TCU_FP8E4M3_ID, TCU_FP16_ID, 8'd0);

        // byte0=0x00→0.0→0x0000, byte1=0x3C→1.5→0x3E00
        // byte2=0x00→0x0000, byte3=0x3C→0x3E00
        chk16("dst_data[0] byte0=0x00 → FP16=0x0000", dst_data[0], 16'h0000);
        chk16("dst_data[2] byte1=0x3C → FP16=0x3E00", dst_data[2], 16'h3E00);
        chk16("dst_data[4] byte2=0x00 → FP16=0x0000", dst_data[4], 16'h0000);
        chk16("dst_data[6] byte3=0x3C → FP16=0x3E00", dst_data[6], 16'h3E00);
        chk8("elem_valid=8'b01010101 (FP8, i=0,2,4,6 valid)",
             elem_valid, 8'b01010101);
        chk32("O5 rd32 step_w=0 = 0x3E000000",
              pack_rd32(dst_data, TCU_FP16_ID, 1'b0), 32'h3E000000);
        chk32("O5 rd32 step_w=1 = 0x3E000000 (symmetric, same)",
              pack_rd32(dst_data, TCU_FP16_ID, 1'b1), 32'h3E000000);

        // ===================================================================
        $display("\n=== O5b/O5c: FP8E4M3→FP16, 0x423C003C (step_w selection) ===");
        // byte0=0x3C→1.5→0x3E00, byte1=0x00→0.0→0x0000
        // byte2=0x3C→1.5→0x3E00, byte3=0x42→2.5→0x4100
        apply(32'h423C003C, TCU_FP8E4M3_ID, TCU_FP16_ID, 8'd0);

        chk16("dst_data[0] byte0=0x3C → FP16=0x3E00", dst_data[0], 16'h3E00);
        chk16("dst_data[2] byte1=0x00 → FP16=0x0000", dst_data[2], 16'h0000);
        chk16("dst_data[4] byte2=0x3C → FP16=0x3E00", dst_data[4], 16'h3E00);
        chk16("dst_data[6] byte3=0x42 → FP16=0x4100 (2.5)", dst_data[6], 16'h4100);
        chk32("O5b rd32 step_w=0 = 0x00003E00",
              pack_rd32(dst_data, TCU_FP16_ID, 1'b0), 32'h00003E00);
        chk32("O5c rd32 step_w=1 = 0x41003E00",
              pack_rd32(dst_data, TCU_FP16_ID, 1'b1), 32'h41003E00);

        // ===================================================================
        $display("\n=== O6: ZP=0 (symmetric QAT) — must equal O1 ===");
        apply(32'h87654321, TCU_I4_ID, TCU_I8_ID, 8'd0);
        chk32("O6 rd32 step_w=0 = 0x04030201 (ZP=0)",
              pack_rd32(dst_data, TCU_I8_ID, 1'b0), 32'h04030201);

        // ===================================================================
        $display("\n=== O7: ZP=1 (asymmetric QAT), 0x87654321, step_w=0 ===");
        // elem[0]=1-1=0, elem[1]=2-1=1, elem[2]=3-1=2, elem[3]=4-1=3
        apply(32'h87654321, TCU_I4_ID, TCU_I8_ID, 8'd1);
        chk8("O7 dst_data[0][7:0] elem[0]-1=0 → 0x00",  dst_data[0][7:0], 8'h00);
        chk8("O7 dst_data[1][7:0] elem[1]-1=1 → 0x01",  dst_data[1][7:0], 8'h01);
        chk8("O7 dst_data[2][7:0] elem[2]-1=2 → 0x02",  dst_data[2][7:0], 8'h02);
        chk8("O7 dst_data[3][7:0] elem[3]-1=3 → 0x03",  dst_data[3][7:0], 8'h03);
        chk32("O7 rd32 step_w=0 = 0x03020100",
              pack_rd32(dst_data, TCU_I8_ID, 1'b0), 32'h03020100);

        // ===================================================================
        $display("\n=== O8: FP4E2M1→FP16, codes 0-7 / 8-F, ZP=0xA5 (ignored for FP) ===");
        apply(32'h76543210, TCU_FP4E2M1_ID, TCU_FP16_ID, 8'hA5);
        chk16("dst_data[0] 0x0 → +0.0", dst_data[0], 16'h0000);
        chk16("dst_data[1] 0x1 → 0.5 (subnormal)", dst_data[1], 16'h3800);
        chk16("dst_data[2] 0x2 → 1.0", dst_data[2], 16'h3C00);
        chk16("dst_data[3] 0x3 → 1.5", dst_data[3], 16'h3E00);
        chk16("dst_data[4] 0x4 → 2.0", dst_data[4], 16'h4000);
        chk16("dst_data[5] 0x5 → 3.0", dst_data[5], 16'h4200);
        chk16("dst_data[6] 0x6 → 4.0", dst_data[6], 16'h4400);
        chk16("dst_data[7] 0x7 → 6.0", dst_data[7], 16'h4600);
        chk8("elem_valid==8'hFF (FP4, all valid)", elem_valid, 8'hFF);
        apply(32'hFEDCBA98, TCU_FP4E2M1_ID, TCU_FP16_ID, 8'hA5);
        chk16("dst_data[0] 0x8 → -0.0", dst_data[0], 16'h8000);
        chk16("dst_data[1] 0x9 → -0.5", dst_data[1], 16'hB800);
        chk16("dst_data[2] 0xA → -1.0", dst_data[2], 16'hBC00);
        chk16("dst_data[3] 0xB → -1.5", dst_data[3], 16'hBE00);
        chk16("dst_data[4] 0xC → -2.0", dst_data[4], 16'hC000);
        chk16("dst_data[5] 0xD → -3.0", dst_data[5], 16'hC200);
        chk16("dst_data[6] 0xE → -4.0", dst_data[6], 16'hC400);
        chk16("dst_data[7] 0xF → -6.0", dst_data[7], 16'hC600);

        // ===================================================================
        $display("\n--- RESULT: %0d PASS / %0d FAIL ---\n", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS");
        else               $display("SOME TESTS FAILED");
        $finish;
    end

    initial begin
        #50000; $display("TIMEOUT"); $finish;
    end

endmodule

`timescale 1ns/1ps
// tb_ot_formats.sv — the real VX_tcu_operand_transformer + VX_tcu_uops over every supported
// (source, destination) pair: I4/U4 -> INT8, FP8 E4M3, FP16; I8/U8 -> FP16; FP4 -> FP8 E4M3, FP16;
// FP8 E4M3/E5M2 -> FP16. The golden decodes the element to a real, applies the zero point
// (integer sources) and re-encodes it, independent of the converter logic.

`include "VX_define.vh"

module tb_ot_formats import VX_gpu_pkg::*, VX_tcu_pkg::*;;
    localparam NT = `NUM_THREADS;

    logic clk = 0, reset = 1;
    always #5 clk = ~clk;

    VX_execute_if #(.data_t (tcu_exe_t)) in_if();
    VX_execute_if #(.data_t (tcu_exe_t)) out_if();

    VX_tcu_operand_transformer dut (
        .clk (clk), .reset (reset), .execute_if_in (in_if), .execute_if_out (out_if)
    );
    assign out_if.ready = 1'b1;

    ibuffer_t ibuf_in, ibuf_out;
    logic uop_start = 0, uop_next = 0;
    wire  uop_done;
    VX_tcu_uops uops (
        .clk (clk), .reset (reset), .ibuf_in (ibuf_in), .ibuf_out (ibuf_out),
        .start (uop_start), .next (uop_next), .done (uop_done)
    );

    function automatic bit is_int(input int f); return f == TCU_I4_ID || f == TCU_U4_ID || f == TCU_I8_ID || f == TCU_U8_ID; endfunction
    function automatic int src_bw(input int f); return (f == TCU_I4_ID || f == TCU_U4_ID || f == TCU_FP4E2M1_ID) ? 4 : 8; endfunction

    // source code -> sign and magnitude. skip = Inf/NaN (out of scope here)
    task automatic decode(input int f, input int c, input int z, output bit s, output real m, output bit skip);
        int v, zv, es, ms, bias, e, f_;
        skip = 0;
        if (is_int(f)) begin
            int bw = src_bw(f);
            v  = c;
            zv = z & ((1 << bw) - 1);
            if (f == TCU_I4_ID || f == TCU_I8_ID) begin
                if (v  >= (1 << (bw-1))) v  -= (1 << bw);
                if (zv >= (1 << (bw-1))) zv -= (1 << bw);
            end
            v -= zv;
            s = (v < 0);
            m = real'(s ? -v : v);
        end else begin
            int bw = src_bw(f);
            es   = (f == TCU_FP4E2M1_ID) ? 2 : ((f == TCU_FP8E4M3_ID) ? 4 : 5);
            ms   = bw - 1 - es;
            bias = (1 << (es-1)) - 1;
            s    = (c >> (bw-1)) & 1;
            e    = (c >> ms) & ((1 << es) - 1);
            f_   = c & ((1 << ms) - 1);
            if (f == TCU_FP8E4M3_ID && e == 15 && f_ == 7) skip = 1;
            if (f == TCU_FP8E5M2_ID && e == 31) skip = 1;
            m = (e == 0) ? real'(f_) * 2.0**(1 - bias - ms)
                         : real'((1 << ms) + f_) * 2.0**(e - bias - ms);
        end
    endtask

    // exact re-encode; inexact = the value does not fit the destination
    function automatic int encode(input bit s, input real m, input int es, input int ms, output bit inexact);
        int bias = (1 << (es-1)) - 1, e = 0, fr = 0;
        real n = m;
        inexact = 0;
        if (m != 0.0) begin
            e = 0;
            while (n >= 2.0) begin n /= 2.0; e++; end
            while (n <  1.0) begin n *= 2.0; e--; end
            e += bias;
            if (e < 1) begin n = m * 2.0**(bias - 1); e = 0; end else n -= 1.0;
            n *= 2.0**ms;
            fr = int'($floor(n));
            if (real'(fr) != n) inexact = 1;
        end
        return (int'(s) << (es + ms)) | (e << ms) | fr;
    endfunction

    int errs = 0, checks = 0;
    int pair_checks [string];

    task automatic check_pair(input string name, input int src, input int dst, input int lg_pack, input int n_iter);
        int nb  = src_bw(src);
        int dbw = (dst == TCU_FP16_ID) ? 16 : 8;
        int per = 32 / dbw;                   // elements per output word
        for (int it = 0; it < n_iter; ++it) begin
            logic [NT-1:0][31:0] pw;
            logic [NT-1:0][TCU_NRB*8-1:0] zpw;
            int step = $urandom_range((1 << lg_pack) - 1);
            int bi   = $urandom_range(TCU_NRB - 1);
            for (int t = 0; t < NT; ++t) begin
                pw[t] = $urandom;
                for (int b = 0; b < TCU_NRB; ++b) zpw[t][b*8 +: 8] = $urandom;
            end
            // OT.SETZP
            @(negedge clk);
            in_if.valid = 1;
            in_if.data  = '0;
            in_if.data.op_type = INST_ALU_BITS'(INST_TCU_SETZP);
            for (int t = 0; t < NT; ++t) in_if.data.rs1_data[t] = `XLEN'(zpw[t]);
            // WMMA
            @(negedge clk);
            in_if.data = '0;
            in_if.data.op_type = INST_ALU_BITS'(INST_TCU_WMMA);
            in_if.data.op_args.tcu.ot_src = 4'(src);
            in_if.data.op_args.tcu.fmt_s  = 4'(dst);
            in_if.data.op_args.tcu.step_w = 2'(step);
            in_if.data.op_args.tcu.b_idx  = 4'(bi);
            for (int t = 0; t < NT; ++t) in_if.data.rs2_data[t] = `XLEN'(pw[t]);
            @(negedge clk);
            in_if.valid = 0;
            while (!(out_if.valid && INST_TCU_BITS'(out_if.data.op_type) == INST_TCU_WMMA)) @(negedge clk);
            for (int t = 0; t < NT; ++t) begin
                logic [31:0] got = out_if.data.rs2_data[t][31:0];
                for (int k = 0; k < per; ++k) begin
                    int idx = step * per + k;
                    int c   = (pw[t] >> (idx * nb)) & ((1 << nb) - 1);
                    int z   = (zpw[t] >> (bi * 8)) & 255;
                    bit s, skip, inexact;
                    real m;
                    int ex, g;
                    decode(src, c, z, s, m, skip);
                    if (skip) continue;
                    if (dst == TCU_I8_ID) begin
                        int v = s ? -int'(m) : int'(m);
                        ex = v & 255;
                        inexact = 0;
                    end else if (dst == TCU_FP16_ID) ex = encode(s, m, 5, 10, inexact);
                    else                             ex = encode(s, m, 4, 3, inexact);
                    if (m == 0.0 && is_int(src)) ex &= ((1 << (dbw - 1)) - 1); // integer zero is +0
                    g = (got >> (k * dbw)) & ((1 << dbw) - 1);
                    checks++;
                    pair_checks[name]++;
                    if (inexact || g != ex) begin
                        errs++;
                        if (errs < 20) $display("FAIL %s lane=%0d step=%0d k=%0d code=%0x zp=%0x got=%0x exp=%0x inexact=%0d",
                                                name, t, step, k, c, z, g, ex, inexact);
                    end
                end
            end
            @(negedge clk);
        end
        // uop sequencer: same pack factor as the OT slices above
        @(negedge clk);
        ibuf_in = '0;
        ibuf_in.ex_type = EX_TCU;
        ibuf_in.op_type = INST_ALU_BITS'(INST_TCU_WMMA);
        ibuf_in.op_args.tcu.ot_src = 4'(src);
        ibuf_in.op_args.tcu.fmt_s  = 4'(dst);
        reset = 1; @(negedge clk); reset = 0;
        uop_start = 1; @(negedge clk); uop_start = 0;
        for (int u = 0; u < TCU_UOPS; ++u) begin
            int bidx = int'(ibuf_out.op_args.tcu.b_idx);
            int rs2  = int'(ibuf_out.rs2) & 31;
            int sw   = int'(ibuf_out.op_args.tcu.step_w);
            checks++;
            pair_checks[name]++;
            if (rs2 != TCU_RB + (bidx >> lg_pack) || sw != (bidx & ((1 << lg_pack) - 1))) begin
                errs++;
                if (errs < 20) $display("FAIL %s uop=%0d b_idx=%0d rs2=%0d step_w=%0d lg_pack=%0d", name, u, bidx, rs2, sw, lg_pack);
            end
            uop_next = 1; @(negedge clk); uop_next = 0;
        end
    endtask

    initial begin
        in_if.valid = 0;
        in_if.data  = '0;
        ibuf_in     = '0;
        repeat (4) @(negedge clk);
        reset = 0;
        @(negedge clk);
        check_pair("I4->INT8",  TCU_I4_ID,      TCU_I8_ID,      1, 400);
        check_pair("U4->INT8",  TCU_U4_ID,      TCU_I8_ID,      1, 400);
        check_pair("I4->E4M3",  TCU_I4_ID,      TCU_FP8E4M3_ID, 1, 400);
        check_pair("U4->E4M3",  TCU_U4_ID,      TCU_FP8E4M3_ID, 1, 400);
        check_pair("I4->FP16",  TCU_I4_ID,      TCU_FP16_ID,    2, 400);
        check_pair("U4->FP16",  TCU_U4_ID,      TCU_FP16_ID,    2, 400);
        check_pair("I8->FP16",  TCU_I8_ID,      TCU_FP16_ID,    1, 400);
        check_pair("U8->FP16",  TCU_U8_ID,      TCU_FP16_ID,    1, 400);
        check_pair("FP4->E4M3", TCU_FP4E2M1_ID, TCU_FP8E4M3_ID, 1, 400);
        check_pair("FP4->FP16", TCU_FP4E2M1_ID, TCU_FP16_ID,    2, 400);
        check_pair("E4M3->FP16",TCU_FP8E4M3_ID, TCU_FP16_ID,    1, 400);
        check_pair("E5M2->FP16",TCU_FP8E5M2_ID, TCU_FP16_ID,    1, 400);
        foreach (pair_checks[n]) $display("  %-12s %0d checks", n, pair_checks[n]);
        $display("--- RESULT: %0d checks, %0d FAIL ---", checks, errs);
        $finish;
    end

    initial begin #200000000; $display("TIMEOUT"); $finish; end
endmodule

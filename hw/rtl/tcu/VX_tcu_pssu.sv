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

// VX_tcu_pssu — Purism Partial Sum Scaling Unit on the TCU output path.
//
// The TCU accumulates a group's unscaled contribution P_i in the C registers.
// The last k-step uops of a WMMA.DRAIN produce the final P_i of each C register;
// instead of only writing it back, the PSSU applies the group scales and adds it
// to an internal scaled-output accumulator:
//     Y[w][r] <- Y[w][r] + f(P_i) * Sw[w][b][n] * (two_scale ? Sa[w][b][m] : 1)
// (f = INT32->FP32 for the INT PE, identity for the FP PE; r = m*N_STEPS + n),
// and the C register is written back with 0 so the next group starts from P = 0.
// Y is kept per warp and per C register, separate from the TCU accumulator.
//
//   PSSU.SETSCALE  loads one per-lane scale slot (Sw[n] or Sa[m]) of bank b
//   PSSU.READ      writes Y[w][r] to rd and clears it (next tile starts from 0)
//   OT.SETZP       is consumed by the OT and only commits here
//
// Hazards: a SETSCALE to bank b waits until no older drain of that warp/bank is
// in flight; a READ waits until no older drain of that warp is in flight (drains
// are counted from the OT output, in program order, to the Y write-back). Younger
// drains of a warp are held at the OT while it has a SETSCALE/READ not yet retired
// here (VX_tcu_unit), so the in-flight counts only ever contain older drains. A
// drain whose Y entry is still being updated waits for it. Eligible bypass ops are
// selected ahead of PE results; ineligible ones never block them.

`include "VX_define.vh"

module VX_tcu_pssu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter FLAGS_W = 4
) (
    input  wire clk,
    input  wire reset,

    // drain issue (OT output, program order): one per last-k drain uop
    input  wire                 drain_issue,
    input  wire [NW_WIDTH-1:0]  drain_issue_wid,
    input  wire                 drain_issue_bank,

    // PE results + {drain_last, bank, two_scale, is_int}
    input  wire                 pe_valid,
    output wire                 pe_ready,
    input  tcu_res_t            pe_data,
    input  wire [FLAGS_W-1:0]   pe_flags,

    // non-MMA ops from the OT (SETSCALE / READ / SETZP)
    input  wire                 byp_valid,
    output wire                 byp_ready,
    input  tcu_exe_t            byp_data,

    VX_result_if.master         result_if
);
    localparam NT     = `NUM_THREADS;
    localparam NW     = `NUM_WARPS;
    localparam NRC    = TCU_M_STEPS * TCU_N_STEPS;
    localparam LG_N   = `UP($clog2(TCU_N_STEPS));
    localparam LG_RC  = `UP($clog2(NRC));
    localparam LG_M   = `UP($clog2(TCU_M_STEPS));
    localparam NS     = TCU_N_STEPS;
    localparam CNT_W  = 8;
    localparam FP32_ONE = 32'h3F80_0000;

`ifdef TCU_BHF
    localparam CORE_LATENCY = 4 + 2 * 2;
`else
    localparam CORE_LATENCY = 4 + 2 * 8;
`endif

    // ------------------------------------------------------------------
    // state
    // ------------------------------------------------------------------
    // Y and the scale banks are per-lane distributed RAMs (1 write + 1 async read
    // port each, see g_lanes); valid/pending bits and counters are flops.
    localparam Y_AW  = NW_WIDTH + LG_RC;
    localparam SW_AW = NW_WIDTH + 1 + LG_N;
    localparam SA_AW = NW_WIDTH + 1 + LG_M;
    reg [NW-1:0][NRC-1:0] y_valid;
    reg [NW-1:0][NRC-1:0] y_pending;
    reg [NW-1:0][1:0][CNT_W-1:0] inflight;
    wire [NT-1:0][31:0] y_rd;       // Y read port (drain accumulator / PSSU.READ)

    // ------------------------------------------------------------------
    // PE stream decode
    // ------------------------------------------------------------------
    wire pe_drain     = pe_flags[0];
    wire pe_bank      = pe_flags[1];
    wire pe_two_scale = pe_flags[2];
    wire pe_is_int    = pe_flags[3];

    wire [NW_WIDTH-1:0] pe_wid = pe_data.wid;
    wire [LG_RC-1:0] pe_r = LG_RC'(5'(pe_data.rd) - 5'(TCU_RC));
    wire [LG_N-1:0]  pe_n = LG_N'(pe_r % LG_RC'(NS));
    wire [LG_M-1:0]  pe_m = LG_M'(pe_r / LG_RC'(NS));

    // ------------------------------------------------------------------
    // bypass stream decode
    // ------------------------------------------------------------------
    wire [INST_TCU_BITS-1:0] byp_op = INST_TCU_BITS'(byp_data.op_type);
    wire [NW_WIDTH-1:0] byp_wid  = byp_data.wid;
    wire                byp_bank = byp_data.op_args.tcu.bank;
    wire [3:0]          byp_slot = byp_data.op_args.tcu.slot;
    wire is_setscale = (byp_op == INST_TCU_SETSCALE);
    wire is_read     = (byp_op == INST_TCU_READ);
    `UNUSED_VAR (byp_data)   // only header fields, op_args.tcu and rs1_data[31:0] are used
    `UNUSED_VAR (byp_slot)

    wire byp_eligible = is_setscale ? (inflight[byp_wid][byp_bank] == '0)
                      : is_read     ? (inflight[byp_wid][0] == '0) && (inflight[byp_wid][1] == '0)
                                    : 1'b1;

    // ------------------------------------------------------------------
    // output stage
    // ------------------------------------------------------------------
    wire      out_ready;
    tcu_res_t out_data;

    // An eligible bypass op goes first: drains older than it (same warp/bank) are
    // already counted in `inflight` and must retire before it becomes eligible, and
    // younger drains are held at the OT until it retires (VX_tcu_unit), so taking it
    // ahead of PE results is always safe and keeps it from being starved.
    wire pe_block   = pe_drain && y_pending[pe_wid][pe_r];
    wire byp_sel    = byp_valid && byp_eligible;
    wire pe_go      = pe_valid && !pe_block && !byp_sel;
    wire out_valid  = pe_go || byp_sel;

    assign pe_ready  = out_ready && !pe_block && !byp_sel;
    assign byp_ready = out_ready && byp_eligible;

    wire pe_fire    = pe_valid && pe_ready;
    wire byp_fire   = byp_valid && byp_ready;
    wire drain_fire = pe_fire && pe_drain;

    always_comb begin
        out_data = pe_data;
        if (!byp_sel) begin
            if (pe_drain) begin
                out_data.data = '0;   // C register restarts from 0 for the next group
            end
        end else begin
            out_data.uuid  = byp_data.uuid;
            out_data.wid   = byp_data.wid;
            out_data.tmask = byp_data.tmask;
            out_data.PC    = byp_data.PC;
            out_data.wb    = byp_data.wb;
            out_data.rd    = byp_data.rd;
            out_data.pid   = byp_data.pid;
            out_data.sop   = byp_data.sop;
            out_data.eop   = byp_data.eop;
            for (integer t = 0; t < NT; t++) begin
                out_data.data[t] = '1;   // NaN-boxed FP32
                out_data.data[t][31:0] = y_valid[byp_wid][LG_RC'(byp_slot)] ? y_rd[t] : 32'h0;
            end
        end
    end

    VX_elastic_buffer #(
        .DATAW ($bits(tcu_res_t)),
        .SIZE  (2)
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (out_valid),
        .ready_in  (out_ready),
        .data_in   (out_data),
        .data_out  (result_if.data),
        .ready_out (result_if.ready),
        .valid_out (result_if.valid)
    );

    // ------------------------------------------------------------------
    // scaling datapath (one VX_pssu_core_v2 per lane, fixed latency, never stalls)
    // ------------------------------------------------------------------
    wire [NT-1:0][31:0] core_rd;

    // one read address per cycle: the selected bypass op (READ), else the PE result (drain)
    wire [Y_AW-1:0]  y_raddr  = byp_sel ? {byp_wid, LG_RC'(byp_slot)} : {pe_wid, pe_r};
    wire [Y_AW-1:0]  y_waddr  = {done_wid, done_r};
    wire [SW_AW-1:0] sw_raddr = {pe_wid, pe_bank, pe_n};
    wire [SA_AW-1:0] sa_raddr = {pe_wid, pe_bank, pe_m};
    wire [SW_AW-1:0] sw_waddr = {byp_wid, byp_bank, byp_slot[LG_N-1:0]};
    wire [SA_AW-1:0] sa_waddr = {byp_wid, byp_bank, byp_slot[LG_M-1:0]};
    wire sw_we = byp_fire && is_setscale && !byp_data.op_args.tcu.wa;
    wire sa_we = byp_fire && is_setscale &&  byp_data.op_args.tcu.wa;

    for (genvar t = 0; t < NT; ++t) begin : g_lanes
        (* ram_style = "distributed" *) reg [31:0] y_mem  [(1 << Y_AW)];
        (* ram_style = "distributed" *) reg [31:0] sw_mem [(1 << SW_AW)];
        (* ram_style = "distributed" *) reg [31:0] sa_mem [(1 << SA_AW)];

        always @(posedge clk) begin
            if (done_v) y_mem[y_waddr]  <= core_rd[t];
            if (sw_we)  sw_mem[sw_waddr] <= byp_data.rs1_data[t][31:0];
            if (sa_we)  sa_mem[sa_waddr] <= byp_data.rs1_data[t][31:0];
        end
        assign y_rd[t] = y_mem[y_raddr];

        wire [31:0] p   = pe_data.data[t][31:0];
        wire [31:0] s1  = sw_mem[sw_raddr];
        wire [31:0] s2  = pe_two_scale ? sa_mem[sa_raddr] : FP32_ONE;
        wire [31:0] acc = y_valid[pe_wid][pe_r] ? y_rd[t] : 32'h0;

        VX_pssu_core_v2 core (
            .clk    (clk),
            .reset  (reset),
            .enable (1'b1),
            .op     (2'd3),          // FMA2: acc + f(P) * s1 * s2
            .is_fp  (~pe_is_int),
            .rs1    (p),
            .rs2    (s1),
            .rs3    (s2),
            .acc    (acc),
            .rd     (core_rd[t])
        );
    end

    // metadata aligned with the core output
    reg [CORE_LATENCY-1:0] md_valid;
    reg [NW_WIDTH-1:0]     md_wid  [CORE_LATENCY];
    reg [LG_RC-1:0]        md_r    [CORE_LATENCY];
    reg                    md_bank [CORE_LATENCY];

    always @(posedge clk) begin
        if (reset) begin
            md_valid <= '0;
        end else begin
            md_valid <= {md_valid[CORE_LATENCY-2:0], drain_fire};
        end
        md_wid[0]  <= pe_wid;
        md_r[0]    <= pe_r;
        md_bank[0] <= pe_bank;
        for (integer i = 1; i < CORE_LATENCY; i++) begin
            md_wid[i]  <= md_wid[i-1];
            md_r[i]    <= md_r[i-1];
            md_bank[i] <= md_bank[i-1];
        end
    end

    wire                done_v    = md_valid[CORE_LATENCY-1];
    wire [NW_WIDTH-1:0] done_wid  = md_wid[CORE_LATENCY-1];
    wire [LG_RC-1:0]    done_r    = md_r[CORE_LATENCY-1];
    wire                done_bank = md_bank[CORE_LATENCY-1];

    // ------------------------------------------------------------------
    // state updates
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            y_valid   <= '0;
            y_pending <= '0;
            inflight  <= '0;
        end else begin
            if (drain_fire) begin
                y_pending[pe_wid][pe_r] <= 1'b1;
            end
            if (done_v) begin
                y_valid[done_wid][done_r]   <= 1'b1;
                y_pending[done_wid][done_r] <= 1'b0;
            end
            if (byp_fire && is_read) begin
                y_valid[byp_wid][LG_RC'(byp_slot)] <= 1'b0;
            end
            for (integer w = 0; w < NW; w++) begin
                for (integer b = 0; b < 2; b++) begin
                    inflight[w][b] <= inflight[w][b]
                        + CNT_W'(drain_issue && (drain_issue_wid == NW_WIDTH'(w)) && (drain_issue_bank == b[0]))
                        - CNT_W'(done_v && (done_wid == NW_WIDTH'(w)) && (done_bank == b[0]));
                end
            end
        end
    end


endmodule

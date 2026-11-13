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

`include "VX_define.vh"

module VX_tcu_unit import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `NUM_TCU_LANES;
    localparam PE_COUNT   = 2;

    `STATIC_ASSERT (BLOCK_SIZE == `ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_exe_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (tcu_res_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks

        VX_execute_if #(
            .data_t (tcu_exe_t)
        ) pe_execute_if[PE_COUNT]();

        VX_result_if #(
            .data_t (tcu_res_t)
        ) pe_result_if[PE_COUNT]();

`ifdef EXT_PURISM_ENABLE
        // Purism: per_block_execute_if -> OCU (operand transformer) -> {PEs | bypass}
        //         PE results (+ per-PE drain flags) and bypass ops -> PSSU -> result
        localparam FLAGS_W = 4;   // {is_int, two_scale, bank, drain_last}
        localparam FLAGQ_DEPTH = 128;

        VX_execute_if #(
            .data_t (tcu_exe_t)
        ) ot_execute_if();

        VX_tcu_operand_transformer operand_xformer (
            .clk            (clk),
            .reset          (reset),
            .execute_if_in  (per_block_execute_if[block_idx]),
            .execute_if_out (ot_execute_if)
        );

        wire [INST_TCU_BITS-1:0] ot_op = INST_TCU_BITS'(ot_execute_if.data.op_type);
        wire ot_is_mma = (ot_op == INST_TCU_WMMA) || (ot_op == INST_TCU_DRAIN);
        wire ot_drain_last = (ot_op == INST_TCU_DRAIN) && ot_execute_if.data.op_args.tcu.last_k;
        wire [FLAGS_W-2:0] ot_flags = {ot_execute_if.data.op_args.tcu.two_scale,
                                       ot_execute_if.data.op_args.tcu.bank,
                                       ot_drain_last};
        wire ot_pe_sel = ot_execute_if.data.op_args.tcu.fmt_s[3];

        // ---- MMA requests -> PEs, recording drain flags per PE (each PE is in-order)
        // A warp's SETSCALE/READ must take effect in the PSSU before any younger drain
        // of that warp reaches it: count those ops from the OT output (program order)
        // until the PSSU accepts them, and hold the warp's last-k drain uops at the OT
        // meanwhile. (The bypass op itself is already past the OT, so no deadlock.)
        wire byp_valid, byp_ready;
        tcu_exe_t byp_data;
        reg [`NUM_WARPS-1:0][3:0] byp_pend;
        wire [INST_TCU_BITS-1:0] byp_op = INST_TCU_BITS'(byp_data.op_type);
        wire ot_is_sr  = (ot_op == INST_TCU_SETSCALE) || (ot_op == INST_TCU_READ);
        wire byp_is_sr = (byp_op == INST_TCU_SETSCALE) || (byp_op == INST_TCU_READ);
        wire sr_issue  = ot_execute_if.valid && ot_execute_if.ready && ot_is_sr;
        wire sr_retire = byp_valid && byp_ready && byp_is_sr;
        wire drain_hold = ot_drain_last && (byp_pend[ot_execute_if.data.wid] != '0);

        always @(posedge clk) begin
            if (reset) begin
                byp_pend <= '0;
            end else begin
                for (integer w = 0; w < `NUM_WARPS; w++) begin
                    byp_pend[w] <= byp_pend[w]
                        + 4'(sr_issue  && (ot_execute_if.data.wid == NW_WIDTH'(w)))
                        - 4'(sr_retire && (byp_data.wid == NW_WIDTH'(w)));
                end
            end
        end

        wire [PE_COUNT-1:0] flagq_full;
        wire [PE_COUNT-1:0] pe_req_ready;
        wire req_ready = pe_req_ready[ot_pe_sel] && ~flagq_full[ot_pe_sel] && ~drain_hold;

        for (genvar i = 0; i < PE_COUNT; ++i) begin : g_pe_req
            assign pe_req_ready[i] = pe_execute_if[i].ready;
            assign pe_execute_if[i].valid = ot_execute_if.valid && ot_is_mma && (ot_pe_sel == i) && ~flagq_full[i] && ~drain_hold;
            assign pe_execute_if[i].data  = ot_execute_if.data;
        end

        // ---- bypass ops (SETSCALE / READ / SETZP)
        wire byp_in_ready;
        // bypass ops only use the header, op_args and rs1_data[31:0]: zero the rest so
        // synthesis trims those buffer bits (~5k flops of rs2/rs3 and upper rs1 data)
        tcu_exe_t byp_in_data;
        always_comb begin
            byp_in_data = ot_execute_if.data;
            byp_in_data.rs2_data = '0;
            byp_in_data.rs3_data = '0;
            for (integer t = 0; t < NUM_LANES; ++t) begin
                byp_in_data.rs1_data[t][`XLEN-1:32] = '0;
            end
        end

        VX_elastic_buffer #(
            .DATAW ($bits(tcu_exe_t)),
            .SIZE  (2)
        ) byp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (ot_execute_if.valid && ~ot_is_mma),
            .ready_in  (byp_in_ready),
            .data_in   (byp_in_data),
            .data_out  (byp_data),
            .ready_out (byp_ready),
            .valid_out (byp_valid)
        );

        assign ot_execute_if.ready = ot_is_mma ? req_ready : byp_in_ready;
        wire drain_issue = ot_execute_if.valid && ot_execute_if.ready && ot_drain_last;

        // ---- PE responses + flags
        wire [PE_COUNT-1:0] rsp_valid, rsp_ready;
        wire [PE_COUNT-1:0][$bits(tcu_res_t)+FLAGS_W-1:0] rsp_data;

        for (genvar i = 0; i < PE_COUNT; ++i) begin : g_pe_rsp
            wire [FLAGS_W-2:0] flagq_out;
            VX_fifo_queue #(
                .DATAW (FLAGS_W-1),
                .DEPTH (FLAGQ_DEPTH)
            ) flagq (
                .clk      (clk),
                .reset    (reset),
                .push     (pe_execute_if[i].valid && pe_execute_if[i].ready),
                .pop      (pe_result_if[i].valid && pe_result_if[i].ready),
                .data_in  (ot_flags),
                .data_out (flagq_out),
                `UNUSED_PIN (empty),
                `UNUSED_PIN (alm_empty),
                .full     (flagq_full[i]),
                `UNUSED_PIN (alm_full),
                `UNUSED_PIN (size)
            );
            assign rsp_valid[i] = pe_result_if[i].valid;
            assign rsp_data[i]  = {pe_result_if[i].data, 1'(i), flagq_out};
            assign pe_result_if[i].ready = rsp_ready[i];
        end

        wire rsp_out_valid, rsp_out_ready;
        wire [$bits(tcu_res_t)+FLAGS_W-1:0] rsp_out_data;

        VX_stream_arb #(
            .NUM_INPUTS (PE_COUNT),
            .DATAW      ($bits(tcu_res_t)+FLAGS_W),
            .ARBITER    ("R"),
            .OUT_BUF    (3)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid),
            .ready_in  (rsp_ready),
            .data_in   (rsp_data),
            .data_out  (rsp_out_data),
            .valid_out (rsp_out_valid),
            .ready_out (rsp_out_ready),
            `UNUSED_PIN (sel_out)
        );

        VX_tcu_pssu #(
            .FLAGS_W (FLAGS_W)
        ) pssu (
            .clk              (clk),
            .reset            (reset),
            .drain_issue      (drain_issue),
            .drain_issue_wid  (ot_execute_if.data.wid),
            .drain_issue_bank (ot_execute_if.data.op_args.tcu.bank),
            .pe_valid         (rsp_out_valid),
            .pe_ready         (rsp_out_ready),
            .pe_data          (rsp_out_data[FLAGS_W +: $bits(tcu_res_t)]),
            .pe_flags         (rsp_out_data[FLAGS_W-1:0]),
            .byp_valid        (byp_valid),
            .byp_ready        (byp_ready),
            .byp_data         (byp_data),
            .result_if        (per_block_result_if[block_idx])
        );
`else
        VX_pe_switch #(
            .PE_COUNT    (PE_COUNT),
            .NUM_LANES   (NUM_LANES),
            .ARBITER     ("R"),
            .REQ_OUT_BUF (0),
            .RSP_OUT_BUF (3)
        ) pe_switch (
            .clk            (clk),
            .reset          (reset),
            .pe_sel         (per_block_execute_if[block_idx].data.op_args.tcu.fmt_s[3]),
            .execute_in_if  (per_block_execute_if[block_idx]),
            .result_out_if  (per_block_result_if[block_idx]),
            .execute_out_if (pe_execute_if),
            .result_in_if   (pe_result_if)
        );
`endif

        VX_tcu_fp #(
            .INSTANCE_ID (`SFORMATF(("%s-fp%0d", INSTANCE_ID, block_idx)))
        ) tcu_fp (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[0]),
            .result_if  (pe_result_if[0])
        );

        VX_tcu_int #(
            .INSTANCE_ID (`SFORMATF(("%s-int%0d", INSTANCE_ID, block_idx)))
        ) tcu_int (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (pe_execute_if[1]),
            .result_if  (pe_result_if[1])
        );
    end

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) gather_unit (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule

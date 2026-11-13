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

// VX_dispatch_relay — elastic register slice(s) on a dispatch_if lane array.
// fmax lever B (NT32 2-SLR): the operand bundle (dispatch_t ~6-7k bits for NT32)
// travels from the base dispatch buffer (SLR0) to an accel FU (TCU/OCU/PSSU in
// SLR2/3). `DISPATCH_ACCEL_PIPE stages split that registered SLR crossing into
// shorter SLL hops. 0 = passthrough (bit-identical to no relay). Latency-tolerant:
// the accel FUs are deeply pipelined (TCU ~42cy) so +1cy/stage is absorbed, and
// dispatch_if is elastic (valid/ready) so both directions are registered.
module VX_dispatch_relay import VX_gpu_pkg::*; #(
    parameter NUM  = 1,
    parameter PIPE = `DISPATCH_ACCEL_PIPE
) (
    /* verilator lint_off UNUSEDSIGNAL */ // clk/reset unused when PIPE = 0
    input wire            clk,
    input wire            reset,
    /* verilator lint_on UNUSEDSIGNAL */
    VX_dispatch_if.slave  in_if  [NUM],
    VX_dispatch_if.master out_if [NUM]
);
    localparam DATAW = $bits(dispatch_t);

    for (genvar n = 0; n < NUM; ++n) begin : g_lane
        if (PIPE == 0) begin : g_bypass
            assign out_if[n].valid = in_if[n].valid;
            assign out_if[n].data  = in_if[n].data;
            assign in_if[n].ready  = out_if[n].ready;
        end else begin : g_pipe
            wire [PIPE:0]        s_valid;
            wire [PIPE:0]        s_ready;
            wire [DATAW-1:0]     s_data [PIPE:0];

            assign s_valid[0]   = in_if[n].valid;
            assign s_data[0]    = in_if[n].data;
            assign in_if[n].ready = s_ready[0];

            for (genvar i = 0; i < PIPE; ++i) begin : g_stage
                VX_elastic_buffer #(
                    .DATAW   (DATAW),
                    .SIZE    (2),
                    .OUT_REG (1)
                ) slice (
                    .clk       (clk),
                    .reset     (reset),
                    .valid_in  (s_valid[i]),
                    .ready_in  (s_ready[i]),
                    .data_in   (s_data[i]),
                    .valid_out (s_valid[i+1]),
                    .ready_out (s_ready[i+1]),
                    .data_out  (s_data[i+1])
                );
            end

            assign out_if[n].valid = s_valid[PIPE];
            assign out_if[n].data  = dispatch_t'(s_data[PIPE]);
            assign s_ready[PIPE]   = out_if[n].ready;
        end
    end

endmodule

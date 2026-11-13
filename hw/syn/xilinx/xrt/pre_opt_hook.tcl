set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl

# ---------------------------------------------------------------------------
# NT32_SPREAD=1 : floorplan for the single-core NT32 build on the U250 (Purism and its
# baseline). Without it no pblocks are created (upstream behavior).
#  - BUFG suppression: the placer/opt otherwise promote high-fanout valid/fire/enable nets
#    to global clock buffers; in the DFX ULP those nets fail as None-Partition (VPL 18-5241).
#    Use together with PLACE_NO_BUFG=1 (Makefile: place_design -no_bufg_opt).
#  - TCU (with the Purism OCU/PSSU stages) and its dispatch relay -> SLR2. A DSP cascade
#    cannot cross an SLR boundary, so the 512-DSP TCU must stay inside one SLR; SLR2 is
#    DSP-rich and clear of the DDR bank0 PHY (SLR0).
# Optional (env):
#  - NT32_BASE_SLR3=1 : base core (issue/commit/alu/fpu/sfu/lsu) -> SLR3, so it is not packed
#    into SLR0 together with the shell (routing congestion).
#  - NT32_MEM_REGION="X0Y12:X3Y14" : mem_unit -> that clock-region range (inside SLR3).
#  - NT32_DCACHE_REGION="X0Y9:X3Y11" : dcache -> that range (inside SLR2, below mem_unit), so the
#    mem_unit<->dcache nets cross in the left SLL columns. The range must hold the dcache BRAMs.
#  - NT32_RELAY_BAND=<path to relay_band.tcl> : spread the SLR3->SLR2 dispatch crossing over
#    the boundary width (RELAY_BAND_X selects the clock-region columns).
if {[info exists ::env(NT32_SPREAD)] && $::env(NT32_SPREAD) == 1} {
    set vnets [get_nets -hierarchical -quiet -filter {NAME =~ *stream_buffer*valid_in_r* || NAME =~ *g_buffer.valid_in_r* || NAME =~ *g_buffer.fire_in* || NAME =~ *pending_size*empty_r* || NAME =~ *pending_size*full_r* || NAME =~ *pending_size*alm_empty_r* || NAME =~ *pending_size*alm_full_r* || NAME =~ *g_shift_register*req_global_ready* || NAME =~ *g_shift_register*fire_in* || NAME =~ *g_shift_register*valid_in_r* || NAME =~ *operand_xformer*ot_valid* || NAME =~ *operand_xformer*ot_stall* || NAME =~ *div_elastic_adapter/E*}]
    if {[llength $vnets] > 0} {
        set_property CLOCK_BUFFER_TYPE NONE $vnets
        puts "pre_opt_hook: NT32_SPREAD -- BUFG suppressed on [llength $vnets] valid/fire/enable nets"
    } else {
        puts "pre_opt_hook: NT32_SPREAD -- WARNING no nets matched (BUFG suppress skipped)"
    }
    # The divider enable is a high-fanout net; with BUFG suppressed, let phys_opt replicate its driver.
    set hfnets [get_nets -hierarchical -quiet -filter {NAME =~ *div_elastic_adapter/E*}]
    if {[llength $hfnets] > 0} {
        set_property MAX_FANOUT 128 $hfnets
        puts "pre_opt_hook: NT32_SPREAD -- MAX_FANOUT 128 on [llength $hfnets] div enable nets"
    }
    # glob '?' matches the literal '[' and ']' of g_cores[N] ('\[' escaping is lost in -filter).
    set tcu_cells [get_cells -hierarchical -quiet -filter {NAME =~ *g_cores?0?.core*tcu_unit* || NAME =~ *g_cores?0?.core*tcu_dispatch_relay*}]
    if {[llength $tcu_cells] > 0} {
        create_pblock pblock_ntcu
        resize_pblock [get_pblocks pblock_ntcu] -add {CLOCKREGION_X0Y8:CLOCKREGION_X7Y11}
        add_cells_to_pblock [get_pblocks pblock_ntcu] $tcu_cells
        puts "pre_opt_hook: NT32_SPREAD -- TCU -> SLR2 ([llength $tcu_cells] cells)"
    }
    if {[info exists ::env(NT32_BASE_SLR3)] && $::env(NT32_BASE_SLR3) == 1} {
        set base_cells [get_cells -hierarchical -quiet -filter {(NAME =~ *g_cores?0?.core/issue/* || NAME =~ *g_cores?0?.core/commit/* || NAME =~ *g_cores?0?.core/execute/alu_unit/* || NAME =~ *g_cores?0?.core/execute/fpu_unit/* || NAME =~ *g_cores?0?.core/execute/sfu_unit/* || NAME =~ *g_cores?0?.core/execute/lsu_unit/*) && NAME !~ *dispatch_relay*}]
        if {[llength $base_cells] > 0} {
            create_pblock pblock_nbase
            resize_pblock [get_pblocks pblock_nbase] -add {CLOCKREGION_X0Y12:CLOCKREGION_X7Y15}
            add_cells_to_pblock [get_pblocks pblock_nbase] $base_cells
            puts "pre_opt_hook: NT32_BASE_SLR3 -- base (issue/commit/alu/fpu/sfu/lsu) -> SLR3 ([llength $base_cells] cells)"
        } else {
            puts "pre_opt_hook: NT32_BASE_SLR3 -- WARNING no base cells matched"
        }
    }
    if {[info exists ::env(NT32_MEM_REGION)] && $::env(NT32_MEM_REGION) ne ""} {
        set mem_cells [get_cells -hierarchical -quiet -filter {NAME =~ *g_cores?0?.core/mem_unit/*}]
        if {[llength $mem_cells] > 0} {
            lassign [split $::env(NT32_MEM_REGION) :] r0 r1
            create_pblock pblock_nmem
            resize_pblock [get_pblocks pblock_nmem] -add "CLOCKREGION_${r0}:CLOCKREGION_${r1}"
            set nbase [get_pblocks -quiet pblock_nbase]
            if {$nbase ne ""} { set_property PARENT $nbase [get_pblocks pblock_nmem] }
            add_cells_to_pblock [get_pblocks pblock_nmem] $mem_cells
            puts "pre_opt_hook: NT32_MEM_REGION -- mem_unit -> $::env(NT32_MEM_REGION) ([llength $mem_cells] cells)"
        } else {
            puts "pre_opt_hook: NT32_MEM_REGION -- WARNING no mem_unit cells matched"
        }
    }
    if {[info exists ::env(NT32_DCACHE_REGION)] && $::env(NT32_DCACHE_REGION) ne ""} {
        set dc_cells [get_cells -hierarchical -quiet -filter {NAME =~ *g_sockets?0?.socket/dcache/*}]
        if {[llength $dc_cells] > 0} {
            lassign [split $::env(NT32_DCACHE_REGION) :] r0 r1
            create_pblock pblock_ndcache
            resize_pblock [get_pblocks pblock_ndcache] -add "CLOCKREGION_${r0}:CLOCKREGION_${r1}"
            set ntcu [get_pblocks -quiet pblock_ntcu]
            if {$ntcu ne ""} { set_property PARENT $ntcu [get_pblocks pblock_ndcache] }
            add_cells_to_pblock [get_pblocks pblock_ndcache] $dc_cells
            puts "pre_opt_hook: NT32_DCACHE_REGION -- dcache -> $::env(NT32_DCACHE_REGION) ([llength $dc_cells] cells)"
        } else {
            puts "pre_opt_hook: NT32_DCACHE_REGION -- WARNING no dcache cells matched"
        }
    }
    if {[info exists ::env(NT32_RELAY_BAND)] && $::env(NT32_RELAY_BAND) ne ""} {
        source $::env(NT32_RELAY_BAND)
        relay_band
    }
}

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages

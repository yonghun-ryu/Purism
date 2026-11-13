# relay_band (sourced by pre_opt_hook.tcl when NT32_RELAY_BAND=<path to this file>).
# NT32 floorplan with the base core in SLR3 and the TCU in SLR2: the dispatch operand bundle
# (~6.5k bits) crosses from the issue stage (SLR3) into tcu_dispatch_relay (SLR2). With both ends
# floating, the placer funnels the crossing through a few middle SLL columns. This pins the relay
# flip-flops by bit index into column bands along the top two rows of SLR2
# (CLOCKREGION_XnY10:XnY11), so the crossing uses the whole boundary width. The bands are nested
# in pblock_ntcu. RELAY_BAND_X="4 5 6 7" (env) picks the clock-region columns (default 0..7).
proc relay_band {} {
    set xs {0 1 2 3 4 5 6 7}
    if {[info exists ::env(RELAY_BAND_X)] && $::env(RELAY_BAND_X) ne ""} { set xs $::env(RELAY_BAND_X) }
    set nb [llength $xs]
    set ffs [get_cells -hierarchical -quiet -filter {NAME =~ *g_cores?0?.core*tcu_dispatch_relay* && IS_SEQUENTIAL}]
    if {[llength $ffs] == 0} {
        puts "relay_band: WARNING no tcu_dispatch_relay flops matched"
        return
    }
    set maxi 0
    foreach c $ffs {
        if {[regexp {\[(\d+)\]$} $c -> i] && $i > $maxi} { set maxi $i }
    }
    set w [expr {$maxi + 1}]
    for {set b 0} {$b < $nb} {incr b} { set band($b) {} }
    set nobit {}
    foreach c $ffs {
        if {[regexp {\[(\d+)\]$} $c -> i]} {
            lappend band([expr {($i * $nb) / $w}]) $c
        } else {
            lappend nobit $c
        }
    }
    set parent [get_pblocks -quiet pblock_ntcu]
    for {set b 0} {$b < $nb} {incr b} {
        set x [lindex $xs $b]
        set pb [create_pblock pblock_relay$b]
        resize_pblock $pb -add "CLOCKREGION_X${x}Y10:CLOCKREGION_X${x}Y11"
        if {$parent ne ""} { set_property PARENT $parent $pb }
        add_cells_to_pblock $pb $band($b)
        puts "relay_band: band $b (X${x}Y10:X${x}Y11) [llength $band($b)] flops"
    }
    puts "relay_band: [llength $ffs] relay flops, bit width $w, [llength $nobit] unbanded (control)"
}

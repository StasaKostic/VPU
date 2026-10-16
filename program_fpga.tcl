# Program the ZCU104 with a bitstream.
# Usage: vivado -mode batch -source program_fpga.tcl [-tclargs <path/to/bitstream.bit>]

# Default: the newest bitstream saved by generate_bitstream.sh for this branch.
set bitstream [lindex [lsort [glob -nocomplain /home/stasa/CMOS2/repos/VPU/fpga_results/vlen-288/*/xilinx_gr_heep_wrapper.bit]] end]
if {$argc > 0} {
    set bitstream [lindex $argv 0]
}
if {![file exists $bitstream]} {
    error "Bitstream not found: $bitstream"
}

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xczu7*] 0]
current_hw_device $dev
set_property PROGRAM.FILE $bitstream $dev
program_hw_devices $dev
close_hw_target

#!/bin/bash
set -e
# Activate virtual environoment
source /home/stasa/CMOS2/repos/x-heep/.venv/bin/activate
# Set variables
export RISCV_XHEEP="/home/stasa/CMOS2/tools/rv32imc_zve32x_zvl128b"
# Go to right directory and branch
cd /home/stasa/CMOS2/repos/VPU
# Generate HDL files out of templates
make mcu-gen TARGET=zcu104
# Generate bitstream for FPGA
build_start=$(mktemp)
make vivado-fpga FPGA_BOARD=zcu104
# Save bitstream and reports to fpga_results/<branch>/<date>_<commit>, so later builds do not overwrite them
impl_dir=build/x-heep_systems_gr-heep_0/zcu104-vivado/x-heep_systems_gr-heep_0.runs/impl_1
if [ "$impl_dir/xilinx_gr_heep_wrapper.bit" -nt "$build_start" ]; then
    results_dir=fpga_results/$(git rev-parse --abbrev-ref HEAD)/$(date +%Y%m%d-%H%M%S)_$(git rev-parse --short HEAD)
    mkdir -p "$results_dir"
    cp "$impl_dir/xilinx_gr_heep_wrapper.bit" \
       "$impl_dir/xilinx_gr_heep_wrapper_timing_summary_routed.rpt" \
       "$impl_dir/xilinx_gr_heep_wrapper_utilization_placed.rpt" "$results_dir/"
    echo "Saved bitstream and reports to $results_dir"
else
    echo "### ERROR: no new bitstream was generated, nothing saved"
fi
rm -f "$build_start"
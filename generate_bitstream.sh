# Activate virtual environoment
source /home/stasa/CMOS2/repos/x-heep/.venv/bin/activate
# Set variables
export RISCV_XHEEP="/home/stasa/CMOS2/tools/rv32imc_zve32x_zvl128b"
# Go to right directory and branch
cd /home/stasa/CMOS2/repos/VPU
# Generate HDL files out of templates
make mcu-gen TARGET=zcu104
# Generate bitstream for FPGA
make vivado-fpga FPGA_BOARD=zcu104
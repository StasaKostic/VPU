# Prerequisites: generate the bitstream using generate_bitstream.sh, and program the board with the bitstream and connect the board and the flash memory.

# Compile the application
make app PROJECT=gemm16-b6 TARGET=zcu104 ARCH=rv32imc_zve32x_zvl256b LINKER=flash_load
# Load the program to the flash
make flash-prog
#!/usr/bin/env bash
# Environment script for VPU unit-test flow.

# ---- RISC-V toolchain (required for build-sw) ----
export RISCV_XHEEP="/scratch/tools/rv32imc_zve32x_zvl128b"
export PATH="$RISCV_XHEEP/bin:$PATH"
export VERILATOR_VERSION=5.040
export PATH="/softs/verilator/$VERILATOR_VERSION/bin:$PATH"
export VERIBLE_VERSION=v0.0-4023-gc1271a00
export PATH="/softs/verible/verible-${VERIBLE_VERSION}/bin:$PATH"

# ---- Questa/ModelSim (required for build-sim/run) ----
#export QUESTA_HOME="/softs/mentor/qsta/2025.1"
#export PATH="$QUESTA_HOME/bin:$PATH"
alias vsim="/softs/mentor/qsta/2025.1/linux_x86_64/vsim"
export MODEL_TECH="/softs/mentor/qsta/2025.1/linux_x86_64"

# ---- License ----
export MGLS_LICENSE_FILE=1717@edalicsrv.epfl.ch:16000@edalicsrv2.epfl.ch:16001@edalicsrv2.epfl.ch:16002@edalicsrv2.epfl.ch
export LM_LICENSE_FILE=16000@edalicsrv2.epfl.ch:16001@edalicsrv2.epfl.ch:16002@edalicsrv2.epfl.ch
export SALT_LICENSE_SERVER=1717@edalicsrv.epfl.ch:16000@edalicsrv2.epfl.ch:16001@edalicsrv2.epfl.ch:16002@edalicsrv2.epfl.ch

echo "Environment loaded."
echo "riscv32-unknown-elf-gcc -> $(command -v riscv32-unknown-elf-gcc || echo MISSING)"
echo "vsim -> $(command -v vsim || echo MISSING)"
echo "LM_LICENSE_FILE=${LM_LICENSE_FILE:-<unset>}"
echo "MODEL_TECH=${MODEL_TECH:-<unset>}"

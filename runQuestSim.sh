# Activate virtual environoment
source /home/stasa/CMOS2/repos/x-heep/.venv/bin/activate
# Set variables
export RISCV_XHEEP="/home/stasa/CMOS2/tools/rv32imc_zve32x_zvl128b"
export MODEL_TECH=/opt/altera_lite/25.1std/questa_fse/bin
# Go to right directory and branch
cd /home/stasa/CMOS2/repos/VPU
git checkout updated-gr-heep
# Generate HDL files out of templates
make mcu-gen
# Compile the application
make app PROJECT=gr_heep_vermu ARCH=rv32imc_zve32x_zvl128b
# Compile the HDL codes with QuestSim
make questasim-build
# Run simulation
cd /home/stasa/CMOS2/repos/VPU/build/x-heep_systems_gr-heep_0/sim-modelsim/
make run PLUSARGS="c firmware=../../../sw/build/main.hex"
## By default saving signals waveforms is disabled. If you want to enable different options (list of all options are in tb_top.sv) run:
#make run PLUSARGS="c firmware=../../../sw/build/main.hex vcd maxcycles=100 verbose"
## vcd file with waveforms is created in /home/stasa/CMOS2/repos/VPU/build/x-heep_systems_gr-heep_0/sim-modelsim/. To view the file: gtkwave waveform.vcd. To open in Questasim it has to be converted from .vcd to .wlf.
## If vcd file is too large and gtkwave can't open it, convert it to fst:
#vcd2fst waveform.vcd waveform.fst
#gtkwave waveform.fst --start 1375500ns


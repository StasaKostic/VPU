# Go to right directory and branch
cd /home/skostic/VPU
git checkout updated-gr-heep
# Load the environment
source server_env.sh
# Generate HDL files
make mcu-gen
# Compile the application
make app PROJECT=gr_heep_vermu ARCH=rv32imc_zve32x_zvl128b
# Compile the design with QuestaSim
make questasim-build
# Run the simulation
cd /home/skostic/VPU/build/x-heep_systems_gr-heep_0/sim-modelsim/
make run PLUSARGS="c firmware=../../../sw/build/main.hex"

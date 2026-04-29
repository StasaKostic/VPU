create_clock -add -name jtag_clk_pin -period 100.00 -waveform {0 50} [get_ports {jtag_tck_i}];
create_clock -add -name spi_slave_clk_pin -period 16.00 -waveform {0 8} [get_ports {spi_slave_sck_io}];

# System clock and JTAG clock are unrelated
set_clock_groups -asynchronous -group [get_clocks clk_out1_xilinx_clk_wizard_clk_wiz_0_0] -group [get_clocks jtag_clk_pin]

### Reset Constraints
set_false_path -from gr_heep_i/core_v_mini_mcu_i/debug_subsystem_i/dm_obi_top_i/i_dm_top/i_dm_csrs/dmcontrol_q_reg\[ndmreset\]/C
set_false_path -from gr_heep_i/rstgen_i/i_rstgen_bypass/synch_regs_q_reg[3]/C

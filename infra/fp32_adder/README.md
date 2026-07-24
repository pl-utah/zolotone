# Synopsys FP32 Adder Export

This directory is generated from `examples/fp32_add.py`.

Files:
- `fp32_adder_dut.hpp`: lowered C++ version of the current AST design
- `fp32_adder_ref.hpp`: bit-level FP32 golden reference
- `fp32_adder_dut_dpi.cpp`: DPI wrapper around the lowered C++ DUT model
- `fp32_adder_ref_dpi.cpp`: DPI wrapper around the golden reference
- `fp32_adder_dut_pkg.sv`: SystemVerilog DPI import package for the lowered C++ DUT
- `fp32_adder_ref_pkg.sv`: SystemVerilog DPI import package
- `fp32_adder_cpp_vs_cpp_tb.sv`: self-contained VCS testbench comparing C++ DUT vs C++ reference
- `run_vcs_cpp_vs_cpp.sh`: self-contained VCS invocation for the C++ DUT vs C++ reference debug flow

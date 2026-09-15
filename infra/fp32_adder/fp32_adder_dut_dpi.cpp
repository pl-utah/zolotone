#include <cstdint>

#include "fp32_adder_dut.cpp"

extern "C" std::uint32_t fp32_adder_dut_dpi(std::uint32_t lhs_bits, std::uint32_t rhs_bits) {
    return Zolotone::fp32_add(lhs_bits, rhs_bits);
}

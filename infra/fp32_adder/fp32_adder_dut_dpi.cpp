#include <cstdint>

#include "fp32_adder_dut.hpp"

extern "C" std::uint32_t fp32_adder_dut_dpi(std::uint32_t lhs_bits, std::uint32_t rhs_bits) {
    return fp32_add(lhs_bits, rhs_bits);
}

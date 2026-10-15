#include "csr.h"
#include <stdint.h>
#include <stdio.h>
#include "common.h"
#include <riscv_vector.h>

// Matrix multiplication of [N,O] x [O,M] for sub-byte / non-power-of-two
// bitwidths. Each element is padded (sign-extended) to a container type:
//   2, 3, 4, 7 bits -> int8_t
//   11, 15 bits     -> int16_t
//   23 bits         -> int32_t
// Results are kept in the container type, so they wrap modulo 2^container
// bits (same as the vector vmacc), and golden and vector results are compared.
//
// The matrices are streamed instead of stored: elements come from a
// counter-based generator (see common.h), so only one row of A and one row of
// B (B is transposed) live in memory at a time, and every C[i][j] is checked
// as soon as it is computed. Memory use is 2*O elements, independent of N and M.
// Only the dot products are timed; row generation is excluded.

// Keep the golden reference scalar so the timing baseline is not autovectorized
#define SCALAR_ONLY __attribute__((optimize("no-tree-vectorize")))
// Keep the kernels as real calls so the timed region is exactly one dot product
#define KERNEL __attribute__((noinline))

// Short run for RTL simulation: 4x4x4 matrices and only the first test.
// Set to 0 for the full benchmark.
#ifndef SIM_QUICK
#define SIM_QUICK 0
#endif

#if SIM_QUICK
#define N 4
#define M 4
#define O 4
#else
#define N 16
#define M 16
#define O 16
#endif

// Seed offset so B differs from A
#define B_SEED_XOR 0x5bd1e995U

// One row of A and one row of B, sized for the widest container (int32_t).
// Placed in ram2 (64 KiB), since ram1 only has room for the 2 KiB heap/stack.
static int32_t a_row_buf[O] __attribute__((section(".xheep_data_interleaved")));
static int32_t b_row_buf[O] __attribute__((section(".xheep_data_interleaved")));

// dot product of a[o] and b[o]
KERNEL SCALAR_ONLY int32_t dot_golden32(const int32_t *a, const int32_t *b, int o) {
  int32_t c = 0;
  for (int k = 0; k < o; ++k)
    c += a[k] * b[k];
  return c;
}

KERNEL SCALAR_ONLY int16_t dot_golden16(const int16_t *a, const int16_t *b, int o) {
  int16_t c = 0;
  for (int k = 0; k < o; ++k)
    c += a[k] * b[k];
  return c;
}

KERNEL SCALAR_ONLY int8_t dot_golden8(const int8_t *a, const int8_t *b, int o) {
  int8_t c = 0;
  for (int k = 0; k < o; ++k)
    c += a[k] * b[k];
  return c;
}

KERNEL int32_t dot32(const int32_t *ptr_a, const int32_t *ptr_b, int o) {
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  int k = o;
  vint32m1_t vec_s    = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);

  for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
    vl = __riscv_vsetvl_e32m1(k);

    vint32m1_t vec_a = __riscv_vle32_v_i32m1(ptr_a, vl);
    vint32m1_t vec_b = __riscv_vle32_v_i32m1(ptr_b, vl);

    vec_s = __riscv_vmacc_vv_i32m1_tu(vec_s, vec_a, vec_b, vl);
  }

  vint32m1_t vec_sum = __riscv_vredsum_vs_i32m1_i32m1(vec_s, vec_zero, vlmax);
  return __riscv_vmv_x_s_i32m1_i32(vec_sum);
}

KERNEL int16_t dot16(const int16_t *ptr_a, const int16_t *ptr_b, int o) {
  size_t vlmax = __riscv_vsetvlmax_e16m1();
  int k = o;
  vint16m1_t vec_s    = __riscv_vmv_v_x_i16m1(0, vlmax);
  vint16m1_t vec_zero = __riscv_vmv_v_x_i16m1(0, vlmax);

  for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
    vl = __riscv_vsetvl_e16m1(k);

    vint16m1_t vec_a = __riscv_vle16_v_i16m1(ptr_a, vl);
    vint16m1_t vec_b = __riscv_vle16_v_i16m1(ptr_b, vl);

    vec_s = __riscv_vmacc_vv_i16m1_tu(vec_s, vec_a, vec_b, vl);
  }

  vint16m1_t vec_sum = __riscv_vredsum_vs_i16m1_i16m1(vec_s, vec_zero, vlmax);
  return __riscv_vmv_x_s_i16m1_i16(vec_sum);
}

KERNEL int8_t dot8(const int8_t *ptr_a, const int8_t *ptr_b, int o) {
  size_t vlmax = __riscv_vsetvlmax_e8m1();
  int k = o;
  vint8m1_t vec_s    = __riscv_vmv_v_x_i8m1(0, vlmax);
  vint8m1_t vec_zero = __riscv_vmv_v_x_i8m1(0, vlmax);

  for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
    vl = __riscv_vsetvl_e8m1(k);

    vint8m1_t vec_a = __riscv_vle8_v_i8m1(ptr_a, vl);
    vint8m1_t vec_b = __riscv_vle8_v_i8m1(ptr_b, vl);

    vec_s = __riscv_vmacc_vv_i8m1_tu(vec_s, vec_a, vec_b, vl);
  }

  vint8m1_t vec_sum = __riscv_vredsum_vs_i8m1_i8m1(vec_s, vec_zero, vlmax);
  return __riscv_vmv_x_s_i8m1_i8(vec_sum);
}

int main() {
  unsigned int cycles_golden, cycles_vec, t, overhead;
  int speedup;
  int pass;

  CSR_CLEAR_BITS(CSR_REG_MCOUNTINHIBIT, 0x1);

  // Cost of an empty start/stop pair, subtracted from every timed dot product
  CSR_WRITE(CSR_REG_MCYCLE, 0);
  CSR_READ(CSR_REG_MCYCLE, &overhead);

  // For every C[i][j]: regenerates row i of A and row j of B (transposed) with
  // random `bits`-wide values padded to type T, times the golden and the vector
  // dot product separately, accumulates the cycles and compares the results.
  #define RUN_TEST_MATMUL(test_num, bits, seed, T, type_name, golden_fn, vec_fn) \
  do { \
    T *a_row = (T *)a_row_buf; \
    T *b_row = (T *)b_row_buf; \
    cycles_golden = 0; \
    cycles_vec = 0; \
    pass = 1; \
    \
    for (int i = 0; i < N; ++i) { \
      gen_row_##type_name(a_row, O, (seed), i, (bits)); \
      for (int j = 0; j < M; ++j) { \
        gen_row_##type_name(b_row, O, (seed) ^ B_SEED_XOR, j, (bits)); \
        \
        CSR_WRITE(CSR_REG_MCYCLE, 0); \
        T gold = golden_fn(a_row, b_row, O); \
        CSR_READ(CSR_REG_MCYCLE, &t); \
        cycles_golden += t - overhead; \
        \
        CSR_WRITE(CSR_REG_MCYCLE, 0); \
        T actual = vec_fn(a_row, b_row, O); \
        CSR_READ(CSR_REG_MCYCLE, &t); \
        cycles_vec += t - overhead; \
        \
        if (gold != actual) \
          pass = 0; \
      } \
    } \
    \
    speedup = (cycles_vec > 0) ? (int)((100.0f * cycles_golden) / cycles_vec) : 0; \
    \
    printf("Test %-2d (%2d-bit -> %-5s, N=%d,M=%d,O=%d): %s | Gold: %6u, Vec: %6u | Speedup: %2d.%02dx\r\n", \
            test_num, bits, #type_name, N, M, O, pass ? "PASS" : "FAIL", \
            cycles_golden, cycles_vec, speedup / 100, speedup % 100); \
  } while (0)

  printf("Starting rvv_matmul_bitwidth Performance Tests:\r\n");

  // Bitwidths padded to a byte
  RUN_TEST_MATMUL(1,  2, 0xde,       int8_t,  int8,  dot_golden8,  dot8);
#if !SIM_QUICK
  RUN_TEST_MATMUL(2,  3, 0xad,       int8_t,  int8,  dot_golden8,  dot8);
  RUN_TEST_MATMUL(3,  4, 0xbe,       int8_t,  int8,  dot_golden8,  dot8);
  RUN_TEST_MATMUL(4,  7, 0xef,       int8_t,  int8,  dot_golden8,  dot8);

  // Bitwidths padded to 16 bits
  RUN_TEST_MATMUL(5, 11, 0xbeef,     int16_t, int16, dot_golden16, dot16);
  RUN_TEST_MATMUL(6, 15, 0xdead,     int16_t, int16, dot_golden16, dot16);

  // Bitwidths padded to 32 bits
  RUN_TEST_MATMUL(7, 23, 0xdeadbeef, int32_t, int32, dot_golden32, dot32);
#endif

  return 0;
}

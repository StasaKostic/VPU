// VERMU counterpart of the HEEPtor gemm16-b*-dp10-vwr4 applications.
// It computes the same GEMM on the same data as the HEEPtor version:
//   C[m][n] = sum_k (A[m][k] * B[k][n]) >> (GEMM_INPUT_BITS - 1)
// VERMU is a hard SIMD unit, so the inputs cannot stay packed at their exact
// width: widths up to 8 bits are sign-extended to a byte (int8_t), widths up
// to 16 bits to a half-word (int16_t).
#include <stdint.h>
#include <stdio.h>
#include <riscv_vector.h>
#include "data.h"

// VERMU implements no widening instructions (vwmul, vsext, vnsra...), so the
// kernel widens by hand. Each element of twice the input width holds an even
// column n in its low half and the next odd column in its high half:
//   even: vsll by the input width moves it into the high half (low half = 0)
//   odd:  vand clears the low half (the even column)
// vmulh then keeps the high half of the 2x-wide product. Scaling A by
// 2^(INPUT_WIDTH + 1 - GEMM_INPUT_BITS) makes that high half equal
// (a * b) >> (GEMM_INPUT_BITS - 1), rounded toward minus infinity like the
// scalar reference. The accumulator is 2x wide, so the sums never wrap.
#if GEMM_INPUT_BITS <= 8
#define INPUT_WIDTH 8
typedef int16_t acc_t;
typedef vint16m1_t vacc_t;
#define VSETVL(n)           __riscv_vsetvl_e16m1(n)
#define VLE(p, vl)          __riscv_vle16_v_i16m1((const int16_t *)(p), vl)
#define VSE(p, v, vl)       __riscv_vse16_v_i16m1(p, v, vl)
#define VZERO(vl)           __riscv_vmv_v_x_i16m1(0, vl)
#define VSLL(v, s, vl)      __riscv_vsll_vx_i16m1(v, s, vl)
#define VAND(v, x, vl)      __riscv_vand_vx_i16m1(v, x, vl)
#define VMULH(v, x, vl)     __riscv_vmulh_vx_i16m1(v, x, vl)
#define VADD(a, b, vl)      __riscv_vadd_vv_i16m1(a, b, vl)
#define HIGH_HALF_MASK      ((acc_t)0xff00)
#else
#define INPUT_WIDTH 16
typedef int32_t acc_t;
typedef vint32m1_t vacc_t;
#define VSETVL(n)           __riscv_vsetvl_e32m1(n)
#define VLE(p, vl)          __riscv_vle32_v_i32m1((const int32_t *)(p), vl)
#define VSE(p, v, vl)       __riscv_vse32_v_i32m1(p, v, vl)
#define VZERO(vl)           __riscv_vmv_v_x_i32m1(0, vl)
#define VSLL(v, s, vl)      __riscv_vsll_vx_i32m1(v, s, vl)
#define VAND(v, x, vl)      __riscv_vand_vx_i32m1(v, x, vl)
#define VMULH(v, x, vl)     __riscv_vmulh_vx_i32m1(v, x, vl)
#define VADD(a, b, vl)      __riscv_vadd_vv_i32m1(a, b, vl)
#define HIGH_HALF_MASK      ((acc_t)0xffff0000)
#endif

#define A_SCALE_SHIFT (INPUT_WIDTH + 1 - GEMM_INPUT_BITS)
#define HALF_SIZE     (GEMM_SIZE / 2)
// Rows of C computed together, so each loaded and split B row is reused.
#define ROW_BLOCK     4

// Results of the even and odd columns: C[m][2j] and C[m][2j+1].
static acc_t vfu_even[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t vfu_odd[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static int32_t scalar_output[GEMM_SIZE * GEMM_SIZE];

// Keep GCC from specializing this call for the constant input arrays.
static void __attribute__((noipa, optimize("no-tree-vectorize"))) gemm_scalar(
    const gemm_input_t *a, const gemm_input_t *b, int32_t *c)
{
    for (unsigned m = 0; m < GEMM_SIZE; ++m) {
        for (unsigned n = 0; n < GEMM_SIZE; ++n) {
            int32_t sum = 0;
            for (unsigned k = 0; k < GEMM_SIZE; ++k) {
                int32_t product = (int32_t)a[m * GEMM_SIZE + k] * b[k * GEMM_SIZE + n];
                // RISC-V GCC's signed shift rounds each product toward minus infinity.
                sum += product >> (GEMM_INPUT_BITS - 1);
            }
            c[m * GEMM_SIZE + n] = sum;
        }
    }
}

#define ROW_PRODUCTS(r)                                                    \
    do {                                                                   \
        acc_t a_k = (acc_t)(a[(m + r) * GEMM_SIZE + k] * (1 << A_SCALE_SHIFT)); \
        acc_e##r = VADD(acc_e##r, VMULH(b_even, a_k, vl), vl);             \
        acc_o##r = VADD(acc_o##r, VMULH(b_odd, a_k, vl), vl);              \
    } while (0)

#define ROW_STORE(r)                                                       \
    do {                                                                   \
        VSE(&c_even[(m + r) * HALF_SIZE + j], acc_e##r, vl);               \
        VSE(&c_odd[(m + r) * HALF_SIZE + j], acc_o##r, vl);                \
    } while (0)

static void __attribute__((noipa)) gemm_vfu(
    const gemm_input_t *a, const gemm_input_t *b, acc_t *c_even, acc_t *c_odd)
{
    // Strip-mine the column pairs in case VLEN cannot hold all of them.
    for (size_t j = 0, vl; j < HALF_SIZE; j += vl) {
        vl = VSETVL(HALF_SIZE - j);
        for (unsigned m = 0; m < GEMM_SIZE; m += ROW_BLOCK) {
            vacc_t acc_e0 = VZERO(vl), acc_o0 = VZERO(vl);
            vacc_t acc_e1 = VZERO(vl), acc_o1 = VZERO(vl);
            vacc_t acc_e2 = VZERO(vl), acc_o2 = VZERO(vl);
            vacc_t acc_e3 = VZERO(vl), acc_o3 = VZERO(vl);
            for (unsigned k = 0; k < GEMM_SIZE; ++k) {
                vacc_t b_pair = VLE(&b[k * GEMM_SIZE + 2 * j], vl);
                vacc_t b_even = VSLL(b_pair, INPUT_WIDTH, vl);
                vacc_t b_odd = VAND(b_pair, HIGH_HALF_MASK, vl);
                ROW_PRODUCTS(0);
                ROW_PRODUCTS(1);
                ROW_PRODUCTS(2);
                ROW_PRODUCTS(3);
            }
            ROW_STORE(0);
            ROW_STORE(1);
            ROW_STORE(2);
            ROW_STORE(3);
        }
    }
}

static uint64_t cycles_now(void)
{
    uint32_t hi, lo, again;
    do {
        asm volatile("csrr %0, mcycleh" : "=r"(hi) :: "memory");
        asm volatile("csrr %0, mcycle" : "=r"(lo) :: "memory");
        asm volatile("csrr %0, mcycleh" : "=r"(again) :: "memory");
    } while (hi != again);
    return ((uint64_t)hi << 32) | lo;
}

// Keep the setup and checking loops scalar: autovectorizing them emits
// instructions VERMU does not implement (e.g. vlseg2e32.v, vmsne.vv, vsext.vf2).
int __attribute__((optimize("no-tree-vectorize"))) main(void)
{
    for (unsigned i = 0; i < GEMM_SIZE * HALF_SIZE; ++i) {
        vfu_even[i] = (acc_t)0xa5a5a5a5u;
        vfu_odd[i] = (acc_t)0xa5a5a5a5u;
    }
    asm volatile("csrci mcountinhibit, 1" ::: "memory");
    uint64_t start = cycles_now();
    gemm_scalar(scalar_a, scalar_b, scalar_output);
    uint64_t scalar_cycles = cycles_now() - start;

    start = cycles_now();
    gemm_vfu(scalar_a, scalar_b, vfu_even, vfu_odd);

    // Include the final store and its readback in the measured region.
    asm volatile("fence iorw, iorw" ::: "memory");
    volatile acc_t last = ((volatile acc_t *)vfu_odd)[GEMM_SIZE * HALF_SIZE - 1];
    (void)last;
    uint64_t vfu_cycles = cycles_now() - start;

    unsigned scalar_errors = 0, vfu_errors = 0;
    for (unsigned i = 0; i < GEMM_SIZE * GEMM_SIZE; ++i)
        scalar_errors += scalar_output[i] != scalar_expected[i];
    for (unsigned i = 0; i < GEMM_SIZE * HALF_SIZE; ++i) {
        vfu_errors += vfu_even[i] != scalar_expected[2 * i];
        vfu_errors += vfu_odd[i] != scalar_expected[2 * i + 1];
    }
    unsigned vlen = (unsigned)__riscv_vsetvlmax_e8m1() * 8;
    unsigned speedup = vfu_cycles ? (unsigned)(scalar_cycles * 100 / vfu_cycles) : 0;
    printf("RESULT size=%u bits=%u container=%u vlen=%u "
           "scalar_cycles=0x%08x%08x vfu_cycles=0x%08x%08x "
           "speedup=%u.%02u scalar_errors=%u vfu_errors=%u\n",
           GEMM_SIZE, GEMM_INPUT_BITS, INPUT_WIDTH, vlen,
           (unsigned)(scalar_cycles >> 32), (unsigned)scalar_cycles,
           (unsigned)(vfu_cycles >> 32), (unsigned)vfu_cycles,
           speedup / 100, speedup % 100, scalar_errors, vfu_errors);
    return scalar_errors != 0 || vfu_errors != 0;
}

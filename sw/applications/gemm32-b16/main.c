// VERMU counterpart of the HEEPtor gemm32-b*-dp8-vwr2 applications.
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
#define VSEW                "e16"
#define VLE                 "vle16.v"
#define VSE                 "vse16.v"
#define HIGH_HALF_MASK      ((acc_t)0xff00)
#else
#define INPUT_WIDTH 16
typedef int32_t acc_t;
#define VSEW                "e32"
#define VLE                 "vle32.v"
#define VSE                 "vse32.v"
#define HIGH_HALF_MASK      ((acc_t)0xffff0000)
#endif

#define A_SCALE_SHIFT (INPUT_WIDTH + 1 - GEMM_INPUT_BITS)
#define HALF_SIZE     (GEMM_SIZE / 2)

// At VLEN=96 a vector holds 6 e16 or 3 e32 elements, neither of which divides
// the 16 column pairs of a row of C. The kernel therefore vectorizes over the
// rows of C: a vector holds column k of A for VEC_ELEMS rows, and vmulh.vx
// multiplies it by one B[k][n], so the roles of A and B in the high-half
// product above swap. The 32 rows take 6 e16 or 11 e32 strips (the last one
// 2 rows), against 3 e16 or 6 e32 strips per row of C (the last one 4 or 1
// pairs) when vectorizing over the column pairs.
#define VLEN_BITS     96
#define VEC_ELEMS     (VLEN_BITS / (2 * INPUT_WIDTH))
// Columns of C computed together, so each loaded A strip is reused: one
// accumulator per column in the 4 registers left after the A strip and product.
#define COL_BLOCK     4
_Static_assert(GEMM_SIZE % COL_BLOCK == 0, "column blocks must tile C");

// Results of the even and odd columns, C[m][2j] and C[m][2j+1], at
// j * GEMM_SIZE + m, so that every strip of rows is contiguous.
static acc_t vfu_even[HALF_SIZE * GEMM_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t vfu_odd[HALF_SIZE * GEMM_SIZE] __attribute__((section(".xheep_data_interleaved")));
static int32_t scalar_output[GEMM_SIZE * GEMM_SIZE];

// B split once into its even and odd columns, already in the high half, and
// A transposed and already scaled, so the inner loop is one vector load, 4
// scalar loads and a vmulh + vadd per column.
static acc_t b_even[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t b_odd[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t a_t[GEMM_SIZE * GEMM_SIZE] __attribute__((section(".xheep_data_interleaved")));

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

// vsetvli with the remaining element count: VERMU sets vl to min(avl, VLMAX),
// so the last strip is the shorter one.
#define SET_VL(vl, avl) \
    asm volatile("vsetvli %0, %1, " VSEW ", m1, ta, ma" : "=r"(vl) : "r"(avl))

// The kernel may use only the vector registers v0..v5, so it is written in
// inline assembly with fixed registers instead of intrinsics, whose register
// allocation GCC is free to choose. The registers keep their values from one
// asm statement to the next: the function has no intrinsics and is not
// autovectorized, so GCC emits no vector instructions of its own in between.
//   v0: A strip   v1: product   v2..v5: accumulators of columns 2j..2j + 3
static void __attribute__((noipa, optimize("no-tree-vectorize"))) gemm_vfu(
    const gemm_input_t *a, const gemm_input_t *b, acc_t *c_even, acc_t *c_odd)
{
    // The column pairs of B rows are contiguous, so B is split in one pass.
    const acc_t *b_pairs = (const acc_t *)b;
    for (unsigned i = 0, vl; i < GEMM_SIZE * HALF_SIZE; i += vl) {
        SET_VL(vl, GEMM_SIZE * HALF_SIZE - i);
        asm volatile(VLE " v0, (%0)\n\t"
                     "vsll.vi v1, v0, %3\n\t"
                     VSE " v1, (%1)\n\t"
                     "vand.vx v1, v0, %4\n\t"
                     VSE " v1, (%2)"
                     :: "r"(&b_pairs[i]), "r"(&b_even[i]), "r"(&b_odd[i]),
                        "i"(INPUT_WIDTH), "r"(HIGH_HALF_MASK)
                     : "v0", "v1", "memory");
    }
    for (unsigned m = 0; m < GEMM_SIZE; ++m)
        for (unsigned k = 0; k < GEMM_SIZE; ++k)
            a_t[k * GEMM_SIZE + m] = (acc_t)(a[m * GEMM_SIZE + k] * (1 << A_SCALE_SHIFT));

    for (unsigned m = 0, vl; m < GEMM_SIZE; m += vl) {
        SET_VL(vl, GEMM_SIZE - m);
        for (unsigned j = 0; j < HALF_SIZE; j += COL_BLOCK / 2) {
            const acc_t *a_k = &a_t[m];
            const acc_t *be = &b_even[j];
            const acc_t *bo = &b_odd[j];
            // The first k sets the accumulators and the next ones add to them.
            asm volatile(VLE " v0, (%0)\n\t"
                         "vmulh.vx v2, v0, %1\n\t"
                         "vmulh.vx v3, v0, %2\n\t"
                         "vmulh.vx v4, v0, %3\n\t"
                         "vmulh.vx v5, v0, %4"
                         :: "r"(a_k), "r"(be[0]), "r"(bo[0]), "r"(be[1]), "r"(bo[1])
                         : "v0", "v2", "v3", "v4", "v5", "memory");
            for (unsigned k = 1; k < GEMM_SIZE; ++k) {
                a_k += GEMM_SIZE;
                be += HALF_SIZE;
                bo += HALF_SIZE;
                asm volatile(VLE " v0, (%0)\n\t"
                             "vmulh.vx v1, v0, %1\n\t"
                             "vadd.vv v2, v2, v1\n\t"
                             "vmulh.vx v1, v0, %2\n\t"
                             "vadd.vv v3, v3, v1\n\t"
                             "vmulh.vx v1, v0, %3\n\t"
                             "vadd.vv v4, v4, v1\n\t"
                             "vmulh.vx v1, v0, %4\n\t"
                             "vadd.vv v5, v5, v1"
                             :: "r"(a_k), "r"(be[0]), "r"(bo[0]), "r"(be[1]), "r"(bo[1])
                             : "v0", "v1", "v2", "v3", "v4", "v5", "memory");
            }
            acc_t *ce = &c_even[j * GEMM_SIZE + m];
            acc_t *co = &c_odd[j * GEMM_SIZE + m];
            asm volatile(VSE " v2, (%0)\n\t"
                         VSE " v3, (%1)\n\t"
                         VSE " v4, (%2)\n\t"
                         VSE " v5, (%3)"
                         :: "r"(ce), "r"(co), "r"(ce + GEMM_SIZE), "r"(co + GEMM_SIZE)
                         : "memory");
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
    unsigned vlen = (unsigned)__riscv_vsetvlmax_e8m1() * 8;
    if (vlen != VLEN_BITS) {
        printf("ERROR kernel built for vlen=%u, VPU has vlen=%u\n", VLEN_BITS, vlen);
        return 1;
    }
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
    for (unsigned m = 0; m < GEMM_SIZE; ++m) {
        for (unsigned j = 0; j < HALF_SIZE; ++j) {
            vfu_errors += vfu_even[j * GEMM_SIZE + m] != scalar_expected[m * GEMM_SIZE + 2 * j];
            vfu_errors += vfu_odd[j * GEMM_SIZE + m] != scalar_expected[m * GEMM_SIZE + 2 * j + 1];
        }
    }
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

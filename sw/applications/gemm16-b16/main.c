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
#define VLE(p, vl)          __riscv_vle16_v_i16m1((const int16_t *)(p), vl)
#define VSE(p, v, vl)       __riscv_vse16_v_i16m1(p, v, vl)
#define VSLL(v, s, vl)      __riscv_vsll_vx_i16m1(v, s, vl)
#define VAND(v, x, vl)      __riscv_vand_vx_i16m1(v, x, vl)
#define VMV(x, vl)          __riscv_vmv_v_x_i16m1(x, vl)
#define VMACC(d, x, v, vl)  __riscv_vmacc_vx_i16m1(d, x, v, vl)
#define VMULH_VX(v, x, vl)  __riscv_vmulh_vx_i16m1(v, x, vl)
#define VMULH_VV(a, b, vl)  __riscv_vmulh_vv_i16m1(a, b, vl)
#define VADD(a, b, vl)      __riscv_vadd_vv_i16m1(a, b, vl)
#define HIGH_HALF_MASK      ((acc_t)0xff00)
#else
#define INPUT_WIDTH 16
typedef int32_t acc_t;
typedef vint32m1_t vacc_t;
#define VLE(p, vl)          __riscv_vle32_v_i32m1((const int32_t *)(p), vl)
#define VSE(p, v, vl)       __riscv_vse32_v_i32m1(p, v, vl)
#define VSLL(v, s, vl)      __riscv_vsll_vx_i32m1(v, s, vl)
#define VAND(v, x, vl)      __riscv_vand_vx_i32m1(v, x, vl)
#define VMV(x, vl)          __riscv_vmv_v_x_i32m1(x, vl)
#define VMACC(d, x, v, vl)  __riscv_vmacc_vx_i32m1(d, x, v, vl)
#define VMULH_VX(v, x, vl)  __riscv_vmulh_vx_i32m1(v, x, vl)
#define VMULH_VV(a, b, vl)  __riscv_vmulh_vv_i32m1(a, b, vl)
#define VADD(a, b, vl)      __riscv_vadd_vv_i32m1(a, b, vl)
#define HIGH_HALF_MASK      ((acc_t)0xffff0000)
#endif

#define A_SCALE_SHIFT (INPUT_WIDTH + 1 - GEMM_INPUT_BITS)
#define HALF_SIZE     (GEMM_SIZE / 2)

// At VLEN=192 a vector holds 6 of the 8 column pairs of a row. The kernel
// computes the first 6 pairs one row per vector (head) and packs the last 2
// pairs of ROWS_PER_VEC rows into one full vector (tail), so the tail does not
// cost a mostly empty vector per row.
#define VLEN_BITS     192
#define VEC_ELEMS     (VLEN_BITS / (2 * INPUT_WIDTH))
#define HEAD_PAIRS    VEC_ELEMS
#define TAIL_PAIRS    (HALF_SIZE - HEAD_PAIRS)
#define ROWS_PER_VEC  (VEC_ELEMS / TAIL_PAIRS)
#define LAST_ROW      (GEMM_SIZE - 1)
// Rows of C computed together in the head, so each loaded and split B row is reused.
#define ROW_BLOCK     4
_Static_assert(TAIL_PAIRS > 0 && ROWS_PER_VEC * TAIL_PAIRS == VEC_ELEMS, "tail rows must fill the vector");
_Static_assert(ROWS_PER_VEC == 3 && 5 * ROWS_PER_VEC == LAST_ROW, "tail computes 5 full blocks plus 1 row");
_Static_assert(GEMM_SIZE % ROW_BLOCK == 0, "head row blocks must tile C");

// Results of the even and odd columns, C[m][2j] and C[m][2j+1]: the head
// pairs row-major, then the tail pairs row-major, so that every vector store
// is contiguous.
#define C_INDEX(m, j)                                                      \
    ((j) < HEAD_PAIRS ? (m) * HEAD_PAIRS + (j)                             \
                      : GEMM_SIZE * HEAD_PAIRS + (m) * TAIL_PAIRS + (j) - HEAD_PAIRS)
static acc_t vfu_even[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t vfu_odd[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static int32_t scalar_output[GEMM_SIZE * GEMM_SIZE];

// The tail pairs of each B row repeated ROWS_PER_VEC times, so that one load
// gives the B operand of all the rows of a tail block.
static acc_t b_rep[GEMM_SIZE * VEC_ELEMS] __attribute__((section(".xheep_data_interleaved")));

// 0/1 steps at the second and third row of a tail block. VERMU has no slides,
// gathers or masked merges, so A is spread over the rows of a block as
//   a0 + (a1 - a0) * row_ge1 + (a2 - a1) * row_ge2
static const acc_t row_ge1[VEC_ELEMS] = {[TAIL_PAIRS ... VEC_ELEMS - 1] = 1};
static const acc_t row_ge2[VEC_ELEMS] = {[2 * TAIL_PAIRS ... VEC_ELEMS - 1] = 1};

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

#define A_SCALED(m) ((acc_t)(a[(m) * GEMM_SIZE + k] * (1 << A_SCALE_SHIFT)))

// The first k sets the accumulators and the next ones add to them. Zeroing
// them first makes GCC copy one zero vector with vmv1r.v, which VERMU limits
// to the current vl (accumulators used at a longer vl would get 0xff tails).
#define ACC_SET(acc, p, vl) acc = (p)
#define ACC_ADD(acc, p, vl) acc = VADD(acc, p, vl)

// Head: accumulate row m + r of C for this k.
#define ROW_PRODUCTS(r, ACC)                                               \
    do {                                                                   \
        acc_t a_k = A_SCALED(m + (r));                                     \
        ACC(acc_e##r, VMULH_VX(b_even, a_k, vl), vl);                      \
        ACC(acc_o##r, VMULH_VX(b_odd, a_k, vl), vl);                       \
    } while (0)

#define HEAD_K_STEP(ACC)                                                   \
    do {                                                                   \
        vacc_t b_pair = VLE(&b[k * GEMM_SIZE], vl);                        \
        vacc_t b_even = VSLL(b_pair, INPUT_WIDTH, vl);                     \
        vacc_t b_odd = VAND(b_pair, HIGH_HALF_MASK, vl);                   \
        ROW_PRODUCTS(0, ACC);                                              \
        ROW_PRODUCTS(1, ACC);                                              \
        ROW_PRODUCTS(2, ACC);                                              \
        ROW_PRODUCTS(3, ACC);                                              \
    } while (0)

#define ROW_STORE(r)                                                       \
    do {                                                                   \
        VSE(&c_even[C_INDEX(m + (r), 0)], acc_e##r, vl);                   \
        VSE(&c_odd[C_INDEX(m + (r), 0)], acc_o##r, vl);                    \
    } while (0)

// Tail: accumulate rows 3i..3i+2 of C for this k.
#define BLOCK_PRODUCTS(i, ACC)                                             \
    do {                                                                   \
        acc_t a0 = A_SCALED(3 * (i));                                      \
        acc_t a1 = A_SCALED(3 * (i) + 1);                                  \
        acc_t a2 = A_SCALED(3 * (i) + 2);                                  \
        vacc_t a_rows = VMV(a0, vl);                                       \
        a_rows = VMACC(a_rows, (acc_t)(a1 - a0), ge1, vl);                 \
        a_rows = VMACC(a_rows, (acc_t)(a2 - a1), ge2, vl);                 \
        ACC(acc_e##i, VMULH_VV(b_even, a_rows, vl), vl);                   \
        ACC(acc_o##i, VMULH_VV(b_odd, a_rows, vl), vl);                    \
    } while (0)

// Rows 0..14 in five full blocks, row 15 alone.
#define TAIL_K_STEP(ACC)                                                   \
    do {                                                                   \
        vacc_t b_pair = VLE(&b_rep[k * VEC_ELEMS], vl);                    \
        vacc_t b_even = VSLL(b_pair, INPUT_WIDTH, vl);                     \
        vacc_t b_odd = VAND(b_pair, HIGH_HALF_MASK, vl);                   \
        BLOCK_PRODUCTS(0, ACC);                                            \
        BLOCK_PRODUCTS(1, ACC);                                            \
        BLOCK_PRODUCTS(2, ACC);                                            \
        BLOCK_PRODUCTS(3, ACC);                                            \
        BLOCK_PRODUCTS(4, ACC);                                            \
        acc_t a_last = A_SCALED(LAST_ROW);                                 \
        ACC(acc_e5, VMULH_VX(b_even, a_last, vl_row), vl_row);             \
        ACC(acc_o5, VMULH_VX(b_odd, a_last, vl_row), vl_row);              \
    } while (0)

#define BLOCK_STORE(i)                                                     \
    do {                                                                   \
        VSE(&c_even[C_INDEX(3 * (i), HEAD_PAIRS)], acc_e##i, vl);          \
        VSE(&c_odd[C_INDEX(3 * (i), HEAD_PAIRS)], acc_o##i, vl);           \
    } while (0)

// Constant vl values, valid because main checks VLEN. Calling vsetvl instead
// lets GCC fold it into an e8,mf2 vtype for the B copies, which is legal (loads
// and stores only demand the SEW/LMUL ratio) but VERMU stores vl * SEW bits.
static void __attribute__((noipa)) gemm_vfu(
    const gemm_input_t *a, const gemm_input_t *b, acc_t *c_even, acc_t *c_odd)
{
    const size_t vl = VEC_ELEMS;
    const size_t vl_row = TAIL_PAIRS;

    // Head: pairs 0..HEAD_PAIRS-1, one row per vector.
    for (unsigned m = 0; m < GEMM_SIZE; m += ROW_BLOCK) {
        vacc_t acc_e0, acc_o0, acc_e1, acc_o1, acc_e2, acc_o2, acc_e3, acc_o3;
        unsigned k = 0;
        HEAD_K_STEP(ACC_SET);
        for (k = 1; k < GEMM_SIZE; ++k)
            HEAD_K_STEP(ACC_ADD);
        ROW_STORE(0);
        ROW_STORE(1);
        ROW_STORE(2);
        ROW_STORE(3);
    }

    // Tail: the last TAIL_PAIRS pairs of ROWS_PER_VEC rows per vector.
    for (unsigned k = 0; k < GEMM_SIZE; ++k) {
        vacc_t b_seg = VLE(&b[k * GEMM_SIZE + 2 * HEAD_PAIRS], vl_row);
        for (unsigned r = 0; r < ROWS_PER_VEC; ++r)
            VSE(&b_rep[k * VEC_ELEMS + r * TAIL_PAIRS], b_seg, vl_row);
    }
    vacc_t ge1 = VLE(row_ge1, vl);
    vacc_t ge2 = VLE(row_ge2, vl);
    vacc_t acc_e0, acc_o0, acc_e1, acc_o1, acc_e2, acc_o2;
    vacc_t acc_e3, acc_o3, acc_e4, acc_o4, acc_e5, acc_o5;
    unsigned k = 0;
    TAIL_K_STEP(ACC_SET);
    for (k = 1; k < GEMM_SIZE; ++k)
        TAIL_K_STEP(ACC_ADD);
    BLOCK_STORE(0);
    BLOCK_STORE(1);
    BLOCK_STORE(2);
    BLOCK_STORE(3);
    BLOCK_STORE(4);
    VSE(&c_even[C_INDEX(LAST_ROW, HEAD_PAIRS)], acc_e5, vl_row);
    VSE(&c_odd[C_INDEX(LAST_ROW, HEAD_PAIRS)], acc_o5, vl_row);
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
    volatile acc_t last = ((volatile acc_t *)vfu_odd)[C_INDEX(LAST_ROW, HALF_SIZE - 1)];
    (void)last;
    uint64_t vfu_cycles = cycles_now() - start;

    unsigned scalar_errors = 0, vfu_errors = 0;
    for (unsigned i = 0; i < GEMM_SIZE * GEMM_SIZE; ++i)
        scalar_errors += scalar_output[i] != scalar_expected[i];
    for (unsigned m = 0; m < GEMM_SIZE; ++m) {
        for (unsigned j = 0; j < HALF_SIZE; ++j) {
            vfu_errors += vfu_even[C_INDEX(m, j)] != scalar_expected[m * GEMM_SIZE + 2 * j];
            vfu_errors += vfu_odd[C_INDEX(m, j)] != scalar_expected[m * GEMM_SIZE + 2 * j + 1];
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

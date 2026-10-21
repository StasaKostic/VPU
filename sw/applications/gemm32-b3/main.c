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

// At VLEN=288 a vector holds 18 e16 or 9 e32 elements, which divide neither
// the 16 column pairs of a row of C nor its 32 rows. The kernel computes the
// first ROW_ROWS rows vectorized over the rows of C, in full vectors (row
// part): a vector holds column k of A for VEC_ELEMS rows, and vmulh.vx
// multiplies it by one B[k][n], so the roles of A and B in the high-half
// product above swap. The last PAIR_ROWS rows are vectorized over the column
// pairs of a row of C (pair part), in strips of up to VEC_ELEMS pairs. That
// takes 60 e16 or 116 e32 vmulh + vadd per k, against 64 or 128 with either
// vectorization alone.
#define VLEN_BITS     288
#define VEC_ELEMS     (VLEN_BITS / (2 * INPUT_WIDTH))
#define ROW_ROWS      (GEMM_SIZE / VEC_ELEMS * VEC_ELEMS)
#define PAIR_ROWS     (GEMM_SIZE - ROW_ROWS)
// Columns of C computed together in the row part, so each loaded A strip is
// reused, and rows of C computed together in the pair part, so each loaded B
// strip is reused: one accumulator each in the 4 registers left after the
// strip and product. The pair part ends with a shorter block of PAIR_REM rows.
#define BLOCK         4
#define PAIR_REM      (PAIR_ROWS % BLOCK)
_Static_assert(GEMM_SIZE % BLOCK == 0, "row part column blocks must tile C");

// Results of the even and odd columns, C[m][2j] and C[m][2j+1], so that every
// vector store is contiguous: the row part at j * ROW_ROWS + m, the pair part
// at m * HALF_SIZE + j (row-major after the row part).
#define C_INDEX(m, j) ((m) < ROW_ROWS ? (j) * ROW_ROWS + (m) : (m) * HALF_SIZE + (j))
static acc_t vfu_even[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static acc_t vfu_odd[GEMM_SIZE * HALF_SIZE] __attribute__((section(".xheep_data_interleaved")));
static int32_t scalar_output[GEMM_SIZE * GEMM_SIZE];

// B split once into its even and odd columns, already in the high half, and
// A transposed and already scaled: the row part loads an A strip and takes B
// from 4 scalar loads, the pair part loads a B strip and takes A from them.
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

// The first k sets the accumulators and the next ones add to them: the
// strip in v0 is multiplied by the scalars x0.. into v2.. (n of them), then
// the accumulators are stored to p0... Unused operands are left unread.
#define MULH_SET(d, x)  "vmulh.vx v" #d ", v0, %" #x "\n\t"
#define MULH_ADD(d, x)  "vmulh.vx v1, v0, %" #x "\n\tvadd.vv v" #d ", v" #d ", v1\n\t"
#define STORE_ACC(d, p) VSE " v" #d ", (%" #p ")\n\t"
#define SET_1  MULH_SET(2, 1)
#define SET_2  SET_1 MULH_SET(3, 2)
#define SET_4  SET_2 MULH_SET(4, 3) MULH_SET(5, 4)
#define ADD_1  MULH_ADD(2, 1)
#define ADD_2  ADD_1 MULH_ADD(3, 2)
#define ADD_4  ADD_2 MULH_ADD(4, 3) MULH_ADD(5, 4)
#define ST_1   STORE_ACC(2, 0)
#define ST_2   ST_1 STORE_ACC(3, 1)
#define ST_4   ST_2 STORE_ACC(4, 2) STORE_ACC(5, 3)
#define FIRST_K(n, strip, x0, x1, x2, x3)                                  \
    asm volatile(VLE " v0, (%0)\n\t" SET_##n                               \
                 :: "r"(strip), "r"(x0), "r"(x1), "r"(x2), "r"(x3)         \
                 : "v0", "v2", "v3", "v4", "v5", "memory")
#define NEXT_K(n, strip, x0, x1, x2, x3)                                   \
    asm volatile(VLE " v0, (%0)\n\t" ADD_##n                               \
                 :: "r"(strip), "r"(x0), "r"(x1), "r"(x2), "r"(x3)         \
                 : "v0", "v1", "v2", "v3", "v4", "v5", "memory")
#define STORE_ACCS(n, p0, p1, p2, p3)                                      \
    asm volatile(ST_##n :: "r"(p0), "r"(p1), "r"(p2), "r"(p3) : "memory")

// Pair part: pairs j..j + vl - 1 of the n rows m.., A[m + r][k] = a_k[r].
// Rows past the block repeat a_k[0], so no operand reads outside a_t.
#define PAIR_BLOCK(n, x1, x2, x3)                                          \
    do {                                                                   \
        const acc_t *a_k = &a_t[m];                                        \
        const acc_t *b_k = &b_half[j];                                     \
        FIRST_K(n, b_k, a_k[0], x1, x2, x3);                               \
        for (unsigned k = 1; k < GEMM_SIZE; ++k) {                         \
            a_k += GEMM_SIZE;                                              \
            b_k += HALF_SIZE;                                              \
            NEXT_K(n, b_k, a_k[0], x1, x2, x3);                            \
        }                                                                  \
        acc_t *c_m = &c[C_INDEX(m, j)];                                    \
        STORE_ACCS(n, c_m, c_m + HALF_SIZE, c_m + 2 * HALF_SIZE, c_m + 3 * HALF_SIZE); \
    } while (0)

// The kernel may use only the vector registers v0..v5, so it is written in
// inline assembly with fixed registers instead of intrinsics, whose register
// allocation GCC is free to choose. The registers keep their values from one
// asm statement to the next: the function has no intrinsics and is not
// autovectorized, so GCC emits no vector instructions of its own in between.
//   row part:  v0: A strip   v1: product   v2..v5: accumulators of columns 2j..2j + 3
//   pair part: v0: B strip   v1: product   v2..v5: accumulators of rows m..m + 3
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

    // Row part: columns 2j..2j + 3 of rows m..m + VEC_ELEMS - 1.
    unsigned vl;
    SET_VL(vl, VEC_ELEMS);
    for (unsigned m = 0; m < ROW_ROWS; m += VEC_ELEMS) {
        for (unsigned j = 0; j < HALF_SIZE; j += BLOCK / 2) {
            const acc_t *a_k = &a_t[m];
            const acc_t *be = &b_even[j];
            const acc_t *bo = &b_odd[j];
            FIRST_K(4, a_k, be[0], bo[0], be[1], bo[1]);
            for (unsigned k = 1; k < GEMM_SIZE; ++k) {
                a_k += GEMM_SIZE;
                be += HALF_SIZE;
                bo += HALF_SIZE;
                NEXT_K(4, a_k, be[0], bo[0], be[1], bo[1]);
            }
            acc_t *ce = &c_even[C_INDEX(m, j)];
            acc_t *co = &c_odd[C_INDEX(m, j)];
            STORE_ACCS(4, ce, co, ce + ROW_ROWS, co + ROW_ROWS);
        }
    }

    // Pair part: rows ROW_ROWS.. in blocks of BLOCK and a last one of
    // PAIR_REM, the last strip of pairs shorter.
    for (unsigned parity = 0; parity < 2; ++parity) {
        const acc_t *b_half = parity ? b_odd : b_even;
        acc_t *c = parity ? c_odd : c_even;
        for (unsigned j = 0; j < HALF_SIZE; j += vl) {
            SET_VL(vl, HALF_SIZE - j);
            unsigned m = ROW_ROWS;
            for (; m + BLOCK <= GEMM_SIZE; m += BLOCK)
                PAIR_BLOCK(4, a_k[1], a_k[2], a_k[3]);
#if PAIR_REM == 1
            PAIR_BLOCK(1, a_k[0], a_k[0], a_k[0]);
#elif PAIR_REM == 2
            PAIR_BLOCK(2, a_k[1], a_k[0], a_k[0]);
#elif PAIR_REM != 0
#error "no pair block for PAIR_REM rows"
#endif
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

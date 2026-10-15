// common.h

#include <stdint.h>

// Stateless 32-bit hash (lowbias32), used as a counter-based random generator:
// any matrix element can be regenerated from (seed, row, col) on demand, so the
// matrices never have to be stored.
static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Random signed value that fits in `bits` bits: [-(2^(bits-1)), 2^(bits-1) - 1].
// The value is sign-extended (padded) to the width of the container type.
static inline int32_t rand_bits(uint32_t seed, int row, int col, int bits) {
  uint32_t h = mix32(seed ^ mix32((uint32_t)row * 0x9e3779b9U + (uint32_t)col));
  return (int32_t)(h & ((1U << bits) - 1)) - (1 << (bits - 1));
}

// Fills `row` with row `r` of the matrix identified by `seed`. The same
// (seed, r) always produces the same values. Kept scalar: autovectorizing it
// emits instructions vermu does not implement (e.g. vnsrl.wi).
#define DEFINE_GEN_ROW(type_name, T)                                        \
  __attribute__((noinline, optimize("no-tree-vectorize")))                  \
  void gen_row_##type_name(T *row, int o, uint32_t seed, int r, int bits) { \
    for (int k = 0; k < o; ++k)                                             \
      row[k] = (T)rand_bits(seed, r, k, bits);                              \
  }

DEFINE_GEN_ROW(int32, int32_t)
DEFINE_GEN_ROW(int16, int16_t)
DEFINE_GEN_ROW(int8, int8_t)
